"""Tests for Python-based activity writer with separate audit database."""

import pytest
from sqlalchemy import func, select

from tests.python_activity_writer_separate_db.flask_app import (
    AuditLogActivity,
    AuditLogTransaction,
    User,
    audit_logger,
    db,
)


@pytest.mark.usefixtures("test_client")
class TestPythonActivityWriterSeparateDatabase:
    """Test that both transactions and activities are written to the secondary database."""

    def test_transaction_written_to_secondary_database(self, user):
        """Test that transaction records are written to the secondary audit database."""
        # Query the secondary database directly for transaction records
        with audit_logger._audit_session_factory() as audit_session:
            transaction_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
            )
            assert transaction_count == 1, "Transaction should be written to secondary database"

            # Verify we can read the transaction from secondary database
            transaction = audit_session.scalar(select(AuditLogTransaction).limit(1))
            assert transaction is not None
            assert transaction.native_transaction_id > 0

    def test_activity_written_to_secondary_database(self, user):
        """Test that activity records are also written to the secondary audit database."""
        # Query the secondary database directly for activity records
        with audit_logger._audit_session_factory() as audit_session:
            activity_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogActivity)
            )
            assert activity_count == 1, "Activity should be written to secondary database"

            # Verify we can read the activity from secondary database
            activity = audit_session.scalar(select(AuditLogActivity).limit(1))
            assert activity is not None
            assert activity.table_name == "user"
            assert activity.verb == "insert"

    def test_main_database_does_not_have_audit_records(self, user):
        """Test that no audit records are written to the main database.

        Note: The tables might exist in main database due to metadata registration,
        but no data should be written to them.
        """
        # Check that NO data exists in main database audit tables
        activity_count_main = db.session.scalar(select(func.count()).select_from(AuditLogActivity))
        transaction_count_main = db.session.scalar(
            select(func.count()).select_from(AuditLogTransaction)
        )

        # All data should be in the secondary database, not main
        assert activity_count_main == 0, (
            "No activity records should be in main database "
            "when using secondary audit database with Python activity writer"
        )
        assert transaction_count_main == 0, (
            "No transaction records should be in main database "
            "when using secondary audit database with Python activity writer"
        )

    def test_update_activity_in_secondary_database(self, user):
        """Test that UPDATE operations create activity records in secondary database."""
        user.name = "Updated"
        user.age = 20
        db.session.commit()

        # Query the secondary database for activities
        with audit_logger._audit_session_factory() as audit_session:
            activities = audit_session.scalars(
                select(AuditLogActivity)
                .where(AuditLogActivity.table_name == "user")
                .order_by(AuditLogActivity.id)
            ).all()

            assert len(activities) == 2, "Should have INSERT and UPDATE activities"
            assert activities[0].verb == "insert"
            assert activities[1].verb == "update"
            assert activities[1].changed_data == {"name": "Updated", "age": 20}

    def test_delete_activity_in_secondary_database(self, user):
        """Test that DELETE operations create activity records in secondary database."""
        user_id = user.id
        db.session.delete(user)
        db.session.commit()

        # Query the secondary database for activities
        with audit_logger._audit_session_factory() as audit_session:
            activities = audit_session.scalars(
                select(AuditLogActivity)
                .where(AuditLogActivity.table_name == "user")
                .order_by(AuditLogActivity.id)
            ).all()

            assert len(activities) == 2, "Should have INSERT and DELETE activities"
            assert activities[0].verb == "insert"
            assert activities[1].verb == "delete"
            assert activities[1].old_data["id"] == user_id

    def test_activity_references_transaction(self, user):
        """Test that activity records correctly reference transaction records."""
        with audit_logger._audit_session_factory() as audit_session:
            activity = audit_session.scalar(
                select(AuditLogActivity).where(AuditLogActivity.table_name == "user").limit(1)
            )

            assert activity is not None
            assert activity.transaction_id is not None

            # Verify the transaction exists
            transaction = audit_session.scalar(
                select(AuditLogTransaction).where(
                    AuditLogTransaction.id == activity.transaction_id
                )
            )
            assert transaction is not None
            assert transaction.native_transaction_id == activity.native_transaction_id

    def test_audit_failure_does_not_break_main_transaction(self, monkeypatch):
        """Test that audit log failures don't break the main transaction."""

        # Simulate audit database failure by breaking the session factory
        def failing_session_factory():
            raise Exception("Simulated audit database failure")

        monkeypatch.setattr(audit_logger, "_audit_session_factory", failing_session_factory)

        # This should succeed even though audit logging fails
        user = User(id=10, name="TestUser")
        db.session.add(user)
        db.session.commit()  # Should not raise

        # Verify the user was created in main database
        created_user = db.session.get(User, 10)
        assert created_user is not None
        assert created_user.name == "TestUser"

    def test_rollback_doesnt_create_audit_records(self):
        """Test that rolled back transactions don't create audit records."""
        with audit_logger._audit_session_factory() as audit_session:
            initial_transaction_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
            )
            initial_activity_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogActivity)
            )

        # Create and rollback
        user = User(id=99, name="RollbackUser")
        db.session.add(user)
        db.session.rollback()

        # Create and commit
        user2 = User(id=100, name="CommitUser")
        db.session.add(user2)
        db.session.commit()

        with audit_logger._audit_session_factory() as audit_session:
            final_transaction_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
            )
            final_activity_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogActivity)
            )

        # Only the committed transaction should be recorded
        assert final_transaction_count == initial_transaction_count + 1
        assert final_activity_count == initial_activity_count + 1
