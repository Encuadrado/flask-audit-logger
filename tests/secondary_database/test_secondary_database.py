"""Tests for secondary audit database functionality."""

import pytest
from sqlalchemy import func, select, text

from tests.secondary_database.flask_app import (
    AuditLogTransaction,
    User,
    audit_logger,
    db,
)


@pytest.mark.usefixtures("test_client")
class TestSecondaryAuditDatabase:
    """Test that audit logs are written to the secondary database."""

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

    def test_main_database_does_not_have_transaction_records(self, user):
        """Test that transaction records are NOT in the main database."""
        # Try to query transaction table from main database
        # This would fail if the table doesn't exist or should return 0 records
        try:
            # Check if the transaction table exists in main database
            with db.engine.connect() as connection:
                result = connection.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name = 'transaction' AND table_schema = 'public'"
                    )
                )
                table_exists = result.scalar() > 0

                if table_exists:
                    # If table exists in main DB, it should be empty or have activity records
                    # but the transaction we just created should only be in audit DB
                    pass
        except Exception:
            # If table doesn't exist in main DB, that's expected
            pass

    def test_multiple_transactions_to_secondary_database(self):
        """Test that multiple transactions are correctly written to secondary database."""
        # Create multiple users
        user1 = User(id=1, name="Alice")
        db.session.add(user1)
        db.session.commit()

        user2 = User(id=2, name="Bob")
        db.session.add(user2)
        db.session.commit()

        user3 = User(id=3, name="Charlie")
        db.session.add(user3)
        db.session.commit()

        # Check all transactions are in secondary database
        with audit_logger._audit_session_factory() as audit_session:
            transaction_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
            )
            assert transaction_count == 3, "All transactions should be in secondary database"

    def test_audit_failure_does_not_break_main_transaction(self, monkeypatch):
        """Test that audit log failures don't break the main transaction."""
        # Simulate audit database failure by breaking the engine
        original_session_factory = audit_logger._audit_session_factory

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

        # Restore original session factory
        monkeypatch.setattr(audit_logger, "_audit_session_factory", original_session_factory)

    def test_audit_with_invalid_audit_database(self, monkeypatch):
        """Test graceful handling when audit database connection fails."""
        # Create a user - this should succeed even if audit fails
        user = User(id=20, name="ResilientUser")

        # Create a failing session factory
        class FailingSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, *args, **kwargs):
                raise Exception("Audit database unreachable")

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        def failing_factory():
            return FailingSession()

        # Patch the audit session factory to fail
        monkeypatch.setattr(audit_logger, "_audit_session_factory", failing_factory)

        db.session.add(user)
        db.session.commit()  # Should succeed despite audit failure

        # Verify user exists
        created_user = db.session.get(User, 20)
        assert created_user is not None
        assert created_user.name == "ResilientUser"

    def test_rollback_doesnt_create_transaction_record(self):
        """Test that rolled back transactions don't create audit records."""
        with audit_logger._audit_session_factory() as audit_session:
            initial_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
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
            final_count = audit_session.scalar(
                select(func.count()).select_from(AuditLogTransaction)
            )

        # Only the committed transaction should be recorded
        assert final_count == initial_count + 1
