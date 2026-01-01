"""Tests for Python-based activity writer."""

import pytest
from sqlalchemy import func, select

from tests.python_activity_writer.flask_app import AuditLogActivity, User, db


@pytest.mark.usefixtures("test_client")
class TestPythonActivityWriter:
    """Test Python-based activity writer instead of PostgreSQL triggers."""

    def test_insert(self, user):
        """Test that INSERT operations create activity records."""
        activity = db.session.scalar(select(AuditLogActivity).limit(1))
        assert activity is not None
        assert activity.old_data == {}
        assert activity.changed_data == {"id": user.id, "name": "Jan", "age": 15}
        assert activity.table_name == "user"
        assert activity.native_transaction_id > 0
        assert activity.verb == "insert"

    def test_update(self, user):
        """Test that UPDATE operations create activity records with changed data."""
        user.name = "Luke"
        user.age = 20
        db.session.commit()

        activities = db.session.scalars(
            select(AuditLogActivity)
            .where(AuditLogActivity.table_name == "user")
            .order_by(AuditLogActivity.id)
        ).all()

        # Should have 2 activities: one for insert, one for update
        assert len(activities) == 2

        # Check the update activity
        update_activity = activities[1]
        assert update_activity.verb == "update"
        # old_data should contain all fields (including id)
        assert update_activity.old_data == {"id": 1, "name": "Jan", "age": 15}
        # changed_data should only contain the modified fields
        assert update_activity.changed_data == {"name": "Luke", "age": 20}

    def test_delete(self, user):
        """Test that DELETE operations create activity records."""
        user_id = user.id
        user_name = user.name
        user_age = user.age

        db.session.delete(user)
        db.session.commit()

        activities = db.session.scalars(
            select(AuditLogActivity)
            .where(AuditLogActivity.table_name == "user")
            .order_by(AuditLogActivity.id)
        ).all()

        # Should have 2 activities: one for insert, one for delete
        assert len(activities) == 2

        # Check the delete activity
        delete_activity = activities[1]
        assert delete_activity.verb == "delete"
        assert delete_activity.old_data == {"id": user_id, "name": user_name, "age": user_age}
        assert delete_activity.changed_data == {}

    def test_activity_after_commit(self):
        """Test that activities are only created after commit."""
        user = User(id=1, name="Jack")
        db.session.add(user)
        db.session.commit()
        user = User(id=2, name="Jill")
        db.session.add(user)
        db.session.commit()
        assert db.session.scalar(select(func.count()).select_from(AuditLogActivity)) == 2

    def test_activity_after_rollback(self):
        """Test that rolled back transactions don't create activity records."""
        user = User(id=1, name="Jack")
        db.session.add(user)
        db.session.rollback()
        user = User(id=2, name="Jill")
        db.session.add(user)
        db.session.commit()
        assert db.session.scalar(select(func.count()).select_from(AuditLogActivity)) == 1

    def test_audit_logger_no_actor(self):
        """Test that activity records are created without actor."""
        user = User(id=1, name="Jack")
        db.session.add(user)
        db.session.commit()
        activity = db.session.scalar(select(AuditLogActivity).limit(1))
        assert activity.transaction.actor_id is None

    def test_audit_logger_with_actor(self, test_client, logged_in_user):
        """Test that activity records include actor ID when user is logged in."""
        resp = test_client.post("/article")
        assert resp.status_code == 200
        activity = db.session.scalar(
            select(AuditLogActivity).where(AuditLogActivity.table_name == "article").limit(1)
        )
        assert activity.transaction.actor_id == logged_in_user.id

    def test_raw_inserts(self):
        """Test that raw SQL inserts also create activity records.

        Note: Raw SQL inserts through session.execute() don't trigger after_flush,
        so this is a known limitation of the Python-based activity writer.
        Use the ORM (session.add) or PostgreSQL triggers for raw SQL support.
        """
        import pytest

        pytest.skip("Raw SQL inserts not supported by Python activity writer")

    def test_partial_update(self, user):
        """Test that only changed fields are recorded in activity."""
        # Only update name, not age
        user.name = "Updated Name"
        db.session.commit()

        activities = db.session.scalars(
            select(AuditLogActivity)
            .where(AuditLogActivity.table_name == "user")
            .order_by(AuditLogActivity.id)
        ).all()

        # Should have 2 activities: one for insert, one for update
        assert len(activities) == 2

        # Check the update activity - should only have name in changed_data
        update_activity = activities[1]
        assert update_activity.verb == "update"
        assert "name" in update_activity.changed_data
        assert update_activity.changed_data["name"] == "Updated Name"
        # Old data should contain all fields including the unchanged age
        assert "name" in update_activity.old_data
        assert update_activity.old_data["name"] == "Jan"
        assert "age" in update_activity.old_data
        assert update_activity.old_data["age"] == 15

    def test_data_expression(self, user):
        """Test that the data hybrid property works correctly."""
        user.name = "Luke"
        db.session.commit()
        activities = db.session.scalars(
            select(AuditLogActivity).where(
                AuditLogActivity.table_name == "user",
                AuditLogActivity.data["id"].astext == str(user.id),
            )
        ).all()
        assert len(activities) == 2

    def test_update_with_sqlalchemy_function(self, user):
        """Test that SQLAlchemy functions in updates are handled properly.
        
        This tests the fix for the issue where SQLAlchemy Function objects
        (like func.jsonb_set) in changed_data cause JSON serialization errors.
        The actual computed values should be captured, not placeholder strings.
        """
        # Set initial metadata
        user.metadata = {"key": "value"}
        db.session.commit()

        # Update using a SQLAlchemy function (jsonb_set)
        # This simulates the user's scenario where func.jsonb_set is used
        user.metadata = func.jsonb_set(
            User.metadata,
            ["new_key"],
            '"new_value"',
            True
        )
        db.session.commit()

        # Should have 3 activities: initial insert, first update, second update
        activities = db.session.scalars(
            select(AuditLogActivity)
            .where(AuditLogActivity.table_name == "user")
            .order_by(AuditLogActivity.id)
        ).all()

        assert len(activities) == 3
        
        # The third activity (second update) should contain the ACTUAL computed value
        # from the jsonb_set function, not a placeholder string
        last_activity = activities[2]
        assert last_activity.verb == "update"
        assert "metadata" in last_activity.changed_data
        
        # The metadata should be the actual JSONB result with both keys
        actual_metadata = last_activity.changed_data["metadata"]
        assert isinstance(actual_metadata, dict), "Should be a dict, not a string placeholder"
        assert actual_metadata["key"] == "value", "Should preserve original key"
        assert actual_metadata["new_key"] == "new_value", "Should have new key from jsonb_set"
        
        # Old data should have the original metadata
        assert "metadata" in last_activity.old_data
        assert last_activity.old_data["metadata"] == {"key": "value"}
