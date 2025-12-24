"""Tests for fallback behavior when no secondary database is configured."""
import pytest
from sqlalchemy import func, select

from tests.defaults.flask_app import AuditLogActivity, User, db


@pytest.mark.usefixtures("test_client")
class TestAuditLoggerFallback:
    """Test that audit logger falls back to main database when no secondary DB is configured."""

    def test_transaction_in_main_database_without_secondary_config(self, user):
        """Test that transactions are written to main database when no secondary DB is configured."""
        # The defaults test app doesn't configure a secondary database
        # So transactions should be in the main database
        activity = db.session.scalar(select(AuditLogActivity).limit(1))
        assert activity is not None
        assert activity.transaction is not None
        assert activity.transaction.native_transaction_id > 0

    def test_normal_operation_without_secondary_database(self):
        """Test normal CRUD operations work without secondary database."""
        # Create
        user = User(id=1, name="TestUser", age=25)
        db.session.add(user)
        db.session.commit()
        
        # Read
        fetched_user = db.session.get(User, 1)
        assert fetched_user.name == "TestUser"
        
        # Update
        fetched_user.name = "UpdatedUser"
        db.session.commit()
        
        # Verify audit trail
        activities = db.session.scalars(select(AuditLogActivity)).all()
        assert len(activities) >= 2  # At least insert and update
        
        # Delete
        db.session.delete(fetched_user)
        db.session.commit()
        
        # Verify delete was audited
        activities = db.session.scalars(select(AuditLogActivity)).all()
        assert len(activities) >= 3  # insert, update, delete

    def test_activity_count_matches_operations(self):
        """Test that activity records match the number of operations."""
        # Create multiple users
        for i in range(5):
            user = User(id=i+1, name=f"User{i}")
            db.session.add(user)
            db.session.commit()
        
        # Check activity count
        activity_count = db.session.scalar(
            select(func.count()).select_from(AuditLogActivity)
        )
        assert activity_count == 5
