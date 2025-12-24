import pytest
from flask_login import login_user
from sqlalchemy import text
from sqlalchemy.orm import scoped_session, sessionmaker

from tests.python_activity_writer_separate_db.flask_app import (
    Article,
    AuditLogActivity,
    AuditLogTransaction,
    User,
    app,
    audit_db_conn_str,
    audit_logger,
    db,
)
from tests.utils import REPO_ROOT

ALEMBIC_CONFIG = REPO_ROOT / "tests" / "python_activity_writer_separate_db" / "alembic_config"


@pytest.fixture(scope="session")
def test_client():
    """Set up test client with both main and audit databases."""
    from contextlib import contextmanager

    from tests.utils import clear_alembic_migrations, run_alembic_command

    test_client = app.test_client()

    @contextmanager
    def run_dual_database_migrations():
        """Run migrations on both main and audit databases."""
        # Clear main database
        clear_alembic_migrations(db, ALEMBIC_CONFIG)

        # Clear audit database
        with audit_logger._audit_engine.begin() as connection:
            connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"))

        # Create migrations for main database
        run_alembic_command(
            engine=db.engine,
            command="revision",
            command_kwargs={"autogenerate": True, "rev_id": "1", "message": "create"},
            alembic_config=ALEMBIC_CONFIG,
        )

        # Run migrations on main database
        run_alembic_command(
            engine=db.engine,
            command="upgrade",
            command_kwargs={"revision": "head"},
            alembic_config=ALEMBIC_CONFIG,
        )

        # Create audit tables on secondary database
        with audit_logger._audit_engine.begin() as connection:
            # Enable btree_gist extension on audit database
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gist"))
            # Create the transaction and activity tables on the audit database
            AuditLogTransaction.metadata.create_all(bind=connection)
            AuditLogActivity.metadata.create_all(bind=connection)

        yield

        # Cleanup
        run_alembic_command(
            engine=db.engine,
            command="downgrade",
            command_kwargs={"revision": "base"},
            alembic_config=ALEMBIC_CONFIG,
        )
        clear_alembic_migrations(db, ALEMBIC_CONFIG)

        with audit_logger._audit_engine.begin() as connection:
            connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))

    with app.app_context():
        with run_dual_database_migrations():
            yield test_client


@pytest.fixture(autouse=True)
def enable_transactional_tests(test_client):
    """Enable transactional tests for main database."""
    connection = db.engine.connect()
    transaction = connection.begin()

    db.session = scoped_session(
        session_factory=sessionmaker(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )
    )

    yield

    db.session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def user():
    user = User(id=1, name="Jan", age=15)
    db.session.add(user)
    db.session.commit()
    # Expire the user to force reload on next access
    # This ensures history tracking works correctly
    db.session.expire(user)
    # Access an attribute to trigger reload
    _ = user.name
    yield user


@pytest.fixture
def logged_in_user(test_client):
    user = User(id=100, name="George")
    db.session.add(user)
    with audit_logger.disable(db.session):
        db.session.commit()

    with test_client.application.test_request_context():
        login_user(user)
        yield user


@pytest.fixture
def article():
    a = Article(id=1, name="Wilson, King of Prussia, I lay this hate on you")
    db.session.add(a)
    db.session.commit()
    yield a
