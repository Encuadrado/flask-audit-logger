import logging
import os
import string
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import cached_property
from uuid import UUID

from flask import request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    DDL,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Table,
    Text,
    create_engine,
    event,
    func,
    inspect,
    literal_column,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, ExcludeConstraint, insert
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import ColumnProperty, relationship, sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.elements import TextClause, ClauseElement

from flask_audit_logger import alembic_hooks

HERE = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)


def _make_json_serializable(data):
    """Convert data to JSON serializable format.

    Handles datetime, date, Decimal, UUID, SQLAlchemy expressions and other non-serializable types.
    """
    if isinstance(data, dict):
        return {key: _make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_make_json_serializable(item) for item in data]
    elif isinstance(data, ClauseElement):
        # Handle SQLAlchemy expressions (like func.jsonb_set, etc.)
        # Return a string representation to indicate it's a SQL expression
        return f"<SQL Expression: {type(data).__name__}>"
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, UUID):
        return str(data)
    elif isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    else:
        return data


class ImproperlyConfigured(Exception):
    pass


@dataclass
class PGExtension:
    schema: str
    signature: str

    @property
    def create_sql(self):
        return text(f"CREATE EXTENSION IF NOT EXISTS {self.signature} WITH SCHEMA {self.schema}")

    @property
    def drop_sql(self):
        return text(f"DROP EXTENSION IF EXISTS {self.signature}")


@dataclass
class PGFunction:
    schema: str
    signature: str
    create_sql: TextClause

    @property
    def drop_sql(self):
        return text(f'DROP FUNCTION IF EXISTS "{self.schema}"."{self.signature}" CASCADE')


@dataclass
class PGTrigger:
    schema: str
    signature: str
    table_name: str
    create_sql: TextClause

    @property
    def drop_sql(self):
        return text(f'DROP TRIGGER IF EXISTS "{self.signature}" ON "{self.table_name}"')


class AuditLogger(object):
    _actor_cls = None
    writer = None
    _audit_engine = None
    _audit_session_factory = None

    def __init__(
        self,
        db,
        get_actor_id=None,
        get_client_addr=None,
        actor_cls=None,
        schema=None,
        audit_db_uri=None,
        use_python_activity_writer=False,
    ):
        self._actor_cls = actor_cls
        self.get_actor_id = get_actor_id or _default_actor_id
        self.get_client_addr = get_client_addr or _default_client_addr
        self.schema = schema or "public"
        self.audit_logger_disabled = False
        self.db = db
        self.audit_db_uri = audit_db_uri
        self.use_python_activity_writer = use_python_activity_writer
        # Cache for table OIDs (relid) - these don't change during app lifetime
        self._relid_cache = {}

        # Set up secondary database connection if configured
        if audit_db_uri:
            self._setup_audit_database(audit_db_uri)

        self.transaction_cls = _transaction_model_factory(db.Model, schema, self.actor_cls)
        self.activity_cls = _activity_model_factory(db.Model, schema, self.transaction_cls)
        self.versioned_tables = _detect_versioned_tables(db)
        self.attach_listeners()
        self.initialize_alembic_hooks()

    def _setup_audit_database(self, audit_db_uri):
        """Set up the secondary database connection for audit logs."""
        try:
            # Create engine for the audit database
            self._audit_engine = create_engine(audit_db_uri)

            # Create session factory for audit database
            self._audit_session_factory = sessionmaker(bind=self._audit_engine)

            # Log configuration without exposing credentials
            # Parse the URI to get just the database name
            try:
                from sqlalchemy.engine.url import make_url

                url = make_url(audit_db_uri)
                safe_info = f"database={url.database}, host={url.host or 'localhost'}"
            except Exception:
                safe_info = "configured"

            logger.info(f"Audit database {safe_info}")
        except Exception as e:
            logger.error(f"Failed to setup audit database: {e}")
            raise ImproperlyConfigured(f"Could not configure audit database: {e}")

    def attach_listeners(self):
        """Listeners save transaction records with actor_ids when versioned tables are affected.
        Flush events occur when a mapped object is created or modified. ORM Execute events occur
        when an insert()/update()/delete() is passed to session.execute()."""
        event.listen(Session, "before_flush", self.receive_before_flush)
        event.listen(Session, "do_orm_execute", self.receive_do_orm_execute)
        # Add after_flush listener for Python-based activity writer
        if self.use_python_activity_writer:
            event.listen(Session, "after_flush", self.receive_after_flush)

    def initialize_alembic_hooks(self):
        alembic_hooks.setup_schema(self)
        alembic_hooks.setup_functions_and_triggers(self)
        self.writer = alembic_hooks.init_migration_ops(self.schema)

    def process_revision_directives(self, context, revision, directives):
        if self.writer:
            self.writer.process_revision_directives(context, revision, directives)

    @property
    def prefix(self):
        return f"{self.schema}." if self.schema != "public" else ""

    @cached_property
    def pg_entities(self):
        return [
            self.pg_btree_gist_extension,
            *self.pg_functions,
            *self.pg_triggers,
        ]

    @cached_property
    def pg_functions(self):
        return [
            self.pg_get_setting,
            self.pg_jsonb_subtract,
            self.pg_jsonb_change_key_name,
            self.pg_create_activity,
        ]

    @property
    def pg_triggers(self):
        return [t for triggers in self.pg_triggers_per_table.values() for t in triggers]

    @cached_property
    def pg_triggers_per_table(self):
        triggers_per_table = {}
        for table in self.versioned_tables:
            target_schema = table.schema or "public"
            versioned = table.info.get("versioned", {})
            excluded_columns = ""
            if "exclude" in versioned:
                joined_excludes = ",".join(versioned["exclude"])
                excluded_columns = "'{" + joined_excludes + "}'"

            triggers_per_table[table.name] = [
                PGTrigger(
                    schema=target_schema,
                    table_name=table.name,
                    signature="audit_trigger_insert",
                    create_sql=self.render_sql_template(
                        "audit_trigger_insert.sql",
                        table_name=table.name,
                        excluded_columns=excluded_columns,
                    ),
                ),
                PGTrigger(
                    schema=target_schema,
                    table_name=table.name,
                    signature="audit_trigger_update",
                    create_sql=self.render_sql_template(
                        "audit_trigger_update.sql",
                        table_name=table.name,
                        excluded_columns=excluded_columns,
                    ),
                ),
                PGTrigger(
                    schema=target_schema,
                    table_name=table.name,
                    signature="audit_trigger_delete",
                    create_sql=self.render_sql_template(
                        "audit_trigger_delete.sql",
                        table_name=table.name,
                        excluded_columns=excluded_columns,
                    ),
                ),
            ]

        return triggers_per_table

    @property
    def functions_by_signature(self) -> dict[str, PGFunction]:
        return {pg_func.signature: pg_func for pg_func in self.pg_functions}

    @cached_property
    def pg_btree_gist_extension(self) -> PGExtension:
        return PGExtension(schema="public", signature="btree_gist")

    @property
    def pg_get_setting(self) -> PGFunction:
        return PGFunction(
            schema=self.schema,
            signature="get_setting(setting text, fallback text)",
            create_sql=self.render_sql_template("get_setting.sql"),
        )

    @property
    def pg_jsonb_subtract(self) -> PGFunction:
        return PGFunction(
            schema=self.schema,
            signature="jsonb_subtract(arg1 jsonb, arg2 jsonb)",
            create_sql=self.render_sql_template("jsonb_subtract.sql"),
        )

    @property
    def pg_jsonb_change_key_name(self) -> PGFunction:
        return PGFunction(
            schema=self.schema,
            signature="jsonb_change_key_name(data jsonb, old_key text, new_key text)",
            create_sql=self.render_sql_template("jsonb_change_key_name.sql"),
        )

    @property
    def pg_create_activity(self) -> PGFunction:
        return PGFunction(
            schema=self.schema,
            signature="create_activity()",
            create_sql=self.render_sql_template("create_activity.sql"),
        )

    @contextmanager
    def disable(self, session):
        session.execute(text("SET LOCAL flask_audit_logger.enable_versioning = 'false'"))
        self.audit_logger_disabled = True
        try:
            yield
        finally:
            self.audit_logger_disabled = False
            session.execute(text("SET LOCAL flask_audit_logger.enable_versioning = 'true'"))

    def render_sql_template(
        self, tmpl_name: str, as_text: bool = True, **kwargs
    ) -> TextClause | DDL:
        file_contents = _read_file(f"templates/{tmpl_name}").replace("$$", "$$$$")
        tmpl = string.Template(file_contents)
        context = dict(schema=self.schema)

        context["schema_prefix"] = "{}.".format(self.schema)
        context["revoke_cmd"] = ("REVOKE ALL ON {schema_prefix}activity FROM public;").format(
            **context
        )

        sql = tmpl.substitute(**context, **kwargs)

        if not as_text:
            return DDL(sql)

        return text(sql)

    def receive_do_orm_execute(self, orm_execute_state):
        is_write = (
            orm_execute_state.is_insert
            or orm_execute_state.is_update
            or orm_execute_state.is_delete
        )
        affects_versioned_table = any(
            m.local_table in self.versioned_tables for m in orm_execute_state.all_mappers
        )
        if is_write and affects_versioned_table:
            native_tx_id = orm_execute_state.session.execute(func.txid_current()).scalar()
            self.save_transaction(orm_execute_state.session, native_tx_id)

    def receive_before_flush(self, session, flush_context, instances):
        if _is_session_modified(session, self.versioned_tables):
            native_tx_id = session.execute(func.txid_current()).scalar()
            self.save_transaction(session, native_tx_id)
            # For Python-based activity writer, collect entity changes before flush
            if self.use_python_activity_writer:
                try:
                    # Store native_tx_id on flush_context for use in _collect_entity_changes
                    flush_context._audit_logger_native_tx_id = native_tx_id
                    self._collect_entity_changes(session, flush_context)
                except Exception as e:
                    logger.error(f"Error in audit logger before_flush: {e}", exc_info=True)
                    # Don't re-raise - audit logging failures should not break the application

    def receive_after_flush(self, session, flush_context):
        """Save activity records after flush when using Python-based activity writer."""
        if self.audit_logger_disabled:
            return

        # Retrieve the changes collected before flush
        if hasattr(flush_context, "_audit_logger_changes"):
            changes = flush_context._audit_logger_changes
            entity_map = changes.get("_entity_map", {})
            
            # For inserts, capture auto-generated IDs after flush
            for insert_record in changes["inserts"]:
                if insert_record.get("_needs_pk_capture"):
                    entity_id = insert_record.get("_entity_id")
                    entity = entity_map.get(entity_id)
                    if entity:
                        try:
                            # After flush, use inspect(entity).identity to get PK values safely
                            # This doesn't trigger attribute access that could cause ObjectDeletedError
                            entity_state = inspect(entity)
                            if entity_state.identity:
                                # identity is a tuple of primary key values
                                table = entity.__table__
                                pk_cols = list(table.primary_key.columns)
                                for i, pk_col in enumerate(pk_cols):
                                    if i < len(entity_state.identity):
                                        pk_value = entity_state.identity[i]
                                        if pk_value is not None:
                                            insert_record["changed_data"][pk_col.name] = pk_value
                        except Exception as e:
                            logger.warning(f"Failed to capture auto-generated primary key: {e}")
                    # Clean up temporary fields
                    insert_record.pop("_needs_pk_capture", None)
                    insert_record.pop("_entity_id", None)
            
            # Clean up entity map
            changes.pop("_entity_map", None)
            
            # Capture values for columns that used SQL expressions in updates
            for update_record in changes["updates"]:
                if update_record.get("columns_with_expressions"):
                    table = update_record.get("table")
                    if table is not None:
                        # Query database directly to get computed values without modifying entity state.
                        # This adds one extra query per update with SQL expressions, but:
                        # 1. SQL expressions in updates are rare
                        # 2. Avoids ObjectDeletedError and entity state issues
                        # 3. Ensures accurate audit trail with actual computed values
                        primary_key_cols = [col for col in table.primary_key.columns]
                        
                        # Build WHERE clause using primary key values from changed_data
                        where_clauses = []
                        for pk_col in primary_key_cols:
                            pk_value = update_record["changed_data"].get(pk_col.name)
                            if pk_value is not None:
                                where_clauses.append(pk_col == pk_value)
                        
                        if where_clauses:
                            try:
                                # Query only the columns with SQL expressions
                                columns_to_select = [
                                    table.c[col_name] 
                                    for col_name in update_record["columns_with_expressions"]
                                ]
                                
                                stmt = select(*columns_to_select).where(*where_clauses)
                                result = session.execute(stmt).first()
                                
                                if result:
                                    # Capture the actual computed values
                                    for i, column_name in enumerate(update_record["columns_with_expressions"]):
                                        update_record["changed_data"][column_name] = result[i]
                                else:
                                    logger.warning(
                                        f"Could not find row to capture SQL expression values. "
                                        f"Table: {table.name}, Primary key filter: {where_clauses}"
                                    )
                            except Exception as e:
                                # If query fails, log warning but don't break the transaction
                                logger.warning(
                                    f"Failed to query computed SQL expression values: {e}. "
                                    f"Audit log will not contain computed value for columns: {update_record['columns_with_expressions']}"
                                )
                
                # Clean up temporary fields from all update records
                update_record.pop("columns_with_expressions", None)
                update_record.pop("table", None)
            
            # Now serialize everything to make it JSON-safe
            for change_list in [changes["inserts"], changes["updates"], changes["deletes"]]:
                for change in change_list:
                    change["old_data"] = _make_json_serializable(change.get("old_data", {}))
                    change["changed_data"] = _make_json_serializable(change.get("changed_data", {}))
            
            self.save_activity_records_after_flush(session, changes)

    def _collect_entity_changes(self, session, flush_context):
        """Collect entity changes before they are flushed."""
        try:
            # Get native_tx_id from flush_context if available, otherwise query the session
            # This ensures we use the main database's transaction ID consistently
            if hasattr(flush_context, '_audit_logger_native_tx_id'):
                native_tx_id = flush_context._audit_logger_native_tx_id
            else:
                native_tx_id = session.execute(func.txid_current()).scalar()

            changes = {
                "native_transaction_id": native_tx_id,
                "inserts": [],
                "updates": [],
                "deletes": [],
                "_entity_map": {},  # Map entity memory address to entity for PK capture
            }

            # Collect new entities (INSERTs)
            # Note: Iterating directly over session.new without converting to list
            # to avoid potential identity tracking issues
            for entity in session.new:
                try:
                    if entity.__table__ not in self.versioned_tables:
                        continue
                    insert_data = self._capture_insert_data(entity)
                    changes["inserts"].append(insert_data)
                    # If this insert needs PK capture, map entity for later lookup
                    if insert_data.get("_needs_pk_capture"):
                        changes["_entity_map"][id(entity)] = entity
                except Exception as e:
                    logger.error(f"Error capturing insert data for {entity.__class__.__name__}: {e}", exc_info=True)

            # Collect modified entities (UPDATEs)
            for entity in session.dirty:
                try:
                    if entity.__table__ not in self.versioned_tables:
                        continue
                    if not _is_entity_modified(entity):
                        continue
                    changes["updates"].append(self._capture_update_data(entity))
                except Exception as e:
                    logger.error(f"Error capturing update data for {entity.__class__.__name__}: {e}", exc_info=True)

            # Collect deleted entities (DELETEs)
            for entity in session.deleted:
                try:
                    if entity.__table__ not in self.versioned_tables:
                        continue
                    changes["deletes"].append(self._capture_delete_data(entity))
                except Exception as e:
                    logger.error(f"Error capturing delete data for {entity.__class__.__name__}: {e}", exc_info=True)

            # Store changes on flush_context for retrieval after flush
            flush_context._audit_logger_changes = changes
        except Exception as e:
            logger.error(f"Error collecting entity changes: {e}", exc_info=True)
            # Don't re-raise - audit logging failures should not break the application

    def _get_table_relid(self, session, schema, table_name):
        """Get the PostgreSQL OID (relid) for a table with caching."""
        cache_key = (schema, table_name)
        
        # Check cache first
        if cache_key in self._relid_cache:
            return self._relid_cache[cache_key]
        
        # Query the database if not cached
        try:
            query = text("""
                SELECT c.oid 
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = :table_name AND n.nspname = :schema
            """)
            result = session.execute(query, {"table_name": table_name, "schema": schema})
            row = result.fetchone()
            relid = row[0] if row else None
            
            # Cache the result (even if None)
            self._relid_cache[cache_key] = relid
            return relid
        except Exception as e:
            logger.warning(f"Failed to get relid for {schema}.{table_name}: {e}")
            return None

    def _get_table_relids_batch(self, session, tables):
        """Get PostgreSQL OIDs (relids) for multiple tables in a single query.
        
        Args:
            session: Database session
            tables: List of (schema, table_name) tuples
            
        Returns:
            Dictionary mapping (schema, table_name) -> relid
        """
        # Filter out tables already in cache
        tables_to_query = [(schema, table_name) for schema, table_name in tables 
                          if (schema, table_name) not in self._relid_cache]
        
        if not tables_to_query:
            # All tables already cached
            return {key: self._relid_cache[key] for key in tables}
        
        try:
            # Build a query to get all relids at once
            # Using unnest to pass the table list
            query = text("""
                WITH table_list AS (
                    SELECT 
                        unnest(:schemas::text[]) AS schema,
                        unnest(:table_names::text[]) AS table_name
                )
                SELECT 
                    tl.schema,
                    tl.table_name,
                    c.oid
                FROM table_list tl
                JOIN pg_namespace n ON n.nspname = tl.schema
                JOIN pg_class c ON c.relname = tl.table_name AND c.relnamespace = n.oid
            """)
            
            schemas = [schema for schema, _ in tables_to_query]
            table_names = [table_name for _, table_name in tables_to_query]
            
            result = session.execute(query, {
                "schemas": schemas,
                "table_names": table_names
            })
            
            # Cache all results
            for row in result:
                cache_key = (row[0], row[1])
                self._relid_cache[cache_key] = row[2]
            
            # Return all requested relids (from cache)
            return {key: self._relid_cache.get(key) for key in tables}
            
        except Exception as e:
            logger.warning(f"Failed to batch get relids: {e}")
            # Fallback to individual queries
            result = {}
            for schema, table_name in tables:
                result[(schema, table_name)] = self._get_table_relid(session, schema, table_name)
            return result

    def _capture_insert_data(self, entity):
        """Capture data for an INSERT operation.
        
        Auto-generated primary keys will be captured after flush using inspect(entity).identity.
        """
        table = entity.__table__
        versioned_info = table.info.get("versioned", {})
        excluded_columns = set(versioned_info.get("exclude", []))

        changed_data = {}
        has_auto_pk = False
        
        for column in table.columns:
            if column.name in excluded_columns:
                continue
            value = getattr(entity, column.name, None)
            if value is not None:
                changed_data[column.name] = value
            elif column.primary_key:
                # Primary key is None - will be auto-generated
                has_auto_pk = True

        result = {
            "schema": table.schema or "public",
            "table_name": table.name,
            "verb": "insert",
            "old_data": {},
            "changed_data": changed_data,  # Don't serialize yet
        }
        
        # Mark that we need to capture PK after flush
        # Store the entity's memory address (id) for matching later - doesn't keep entity alive
        if has_auto_pk:
            result["_needs_pk_capture"] = True
            result["_entity_id"] = id(entity)  # Memory address for matching
        
        return result

    def _capture_update_data(self, entity):
        """Capture data for an UPDATE operation."""
        table = entity.__table__
        versioned_info = table.info.get("versioned", {})
        excluded_columns = set(versioned_info.get("exclude", []))

        # For updates, we capture old values of changed fields in old_data
        # and new values of changed fields in changed_data
        # However, we always include primary key columns in both for record identification
        old_data = {}
        changed_data = {}
        columns_with_expressions = set()  # Track columns that use SQL expressions

        insp = inspect(entity)
        
        # Get primary key column names
        primary_key_columns = {col.name for col in table.primary_key.columns}

        # Iterate through all columns to build old_data and changed_data
        for column in table.columns:
            if column.name in excluded_columns:
                continue

            # Get the attribute for this column
            attr = insp.attrs.get(column.name)
            if not attr:
                continue

            history = attr.history
            if history.has_changes():
                # This column was modified
                # For the old value, try deleted first, then unchanged
                if history.deleted:
                    old_data[column.name] = history.deleted[0]
                elif history.unchanged:
                    old_data[column.name] = history.unchanged[0]
                else:
                    # This shouldn't happen but handle it - maybe it's NULL -> value
                    old_data[column.name] = None

                if history.added:
                    added_value = history.added[0]
                    # Check if this is a SQL expression
                    if isinstance(added_value, ClauseElement):
                        # Mark this column to be captured after flush
                        columns_with_expressions.add(column.name)
                        # Don't set changed_data yet - will be populated after flush
                    else:
                        changed_data[column.name] = added_value
            elif column.name in primary_key_columns:
                # Always include primary key columns in both old_data and changed_data
                # even if unchanged (for record identification)
                value = getattr(entity, column.name, None)
                if value is not None:
                    old_data[column.name] = value
                    changed_data[column.name] = value

        return {
            "schema": table.schema or "public",
            "table_name": table.name,
            "verb": "update",
            "old_data": old_data,  # Don't serialize yet
            "changed_data": changed_data,  # Don't serialize yet
            "columns_with_expressions": columns_with_expressions,
            # Don't store entity reference - store table metadata instead
            "table": table if columns_with_expressions else None,
        }

    def _capture_delete_data(self, entity):
        """Capture data for a DELETE operation."""
        table = entity.__table__
        versioned_info = table.info.get("versioned", {})
        excluded_columns = set(versioned_info.get("exclude", []))

        old_data = {}
        for column in table.columns:
            if column.name in excluded_columns:
                continue
            value = getattr(entity, column.name, None)
            if value is not None:
                old_data[column.name] = value

        return {
            "schema": table.schema or "public",
            "table_name": table.name,
            "verb": "delete",
            "old_data": old_data,  # Don't serialize yet
            "changed_data": {},
        }

    def save_activity_records_after_flush(self, session, changes):
        """Save activity records after flush using direct SQL inserts."""
        if self.audit_logger_disabled:
            return

        try:
            all_changes = []
            all_changes.extend(changes["inserts"])
            all_changes.extend(changes["updates"])
            all_changes.extend(changes["deletes"])

            if not all_changes:
                return

            # Get the native transaction ID from the changes
            native_tx_id = changes["native_transaction_id"]

            # Batch fetch all unique table relids to minimize database queries
            unique_tables = set((change["schema"], change["table_name"]) for change in all_changes)
            relid_map = self._get_table_relids_batch(session, list(unique_tables))

            # Prepare activity records for insertion
            for change in all_changes:
                # Skip if no data changes
                if not change["changed_data"] and not change["old_data"]:
                    continue

                # Extract record_id from the data
                # For inserts/updates: prefer changed_data, fallback to old_data
                # For deletes: use old_data
                record_id = None
                if change["verb"] == "delete":
                    record_id = change["old_data"].get("id")
                else:
                    record_id = change["changed_data"].get("id") or change["old_data"].get("id")
                
                # Convert record_id to string if it exists
                if record_id is not None:
                    record_id = str(record_id)

                # Get the table relid (OID) from the batch result
                relid = relid_map.get((change["schema"], change["table_name"]))

                # Look up the transaction_id using the native_transaction_id
                # This query needs to be executed in the audit database context
                if self._audit_engine and self._audit_session_factory:
                    audit_session = self._audit_session_factory()
                    try:
                        # Get the transaction ID from the audit database
                        transaction_id = audit_session.scalar(
                            select(self.transaction_cls.id)
                            .where(self.transaction_cls.native_transaction_id == native_tx_id)
                            .order_by(self.transaction_cls.issued_at.desc())
                            .limit(1)
                        )

                        if transaction_id is None:
                            logger.warning(
                                f"Transaction not found for native_transaction_id={native_tx_id}. "
                                "Activity record will be saved without transaction_id."
                            )

                        # Use SQL insert statement
                        values = {
                            "schema": change["schema"],
                            "table_name": change["table_name"],
                            "relid": relid,
                            "issued_at": text("now() AT TIME ZONE 'UTC'"),
                            "native_transaction_id": native_tx_id,
                            "verb": change["verb"],
                            "old_data": change["old_data"],
                            "changed_data": change["changed_data"],
                            "transaction_id": transaction_id,
                            "record_id": record_id,
                        }

                        stmt = insert(self.activity_cls).values(**values)
                        audit_session.execute(stmt)
                        audit_session.commit()
                    except Exception as audit_error:
                        audit_session.rollback()
                        raise audit_error
                    finally:
                        audit_session.close()
                else:
                    # For main database, use subquery approach
                    values = {
                        "schema": change["schema"],
                        "table_name": change["table_name"],
                        "relid": relid,
                        "issued_at": text("now() AT TIME ZONE 'UTC'"),
                        "native_transaction_id": native_tx_id,
                        "verb": change["verb"],
                        "old_data": change["old_data"],
                        "changed_data": change["changed_data"],
                        "transaction_id": select(self.transaction_cls.id)
                        .where(self.transaction_cls.native_transaction_id == native_tx_id)
                        .order_by(self.transaction_cls.issued_at.desc())
                        .limit(1)
                        .scalar_subquery(),
                        "record_id": record_id,
                    }

                    stmt = insert(self.activity_cls).values(**values)
                    session.execute(stmt)

        except Exception as e:
            logger.error(f"Failed to save activity records: {e}", exc_info=True)

    def save_transaction(self, session, native_tx_id=None):
        if self.audit_logger_disabled:
            return

        # If native_tx_id is not provided, get it from the session
        # This ensures we use the main database's transaction ID even when writing to a secondary audit DB
        if native_tx_id is None:
            native_tx_id = session.execute(func.txid_current()).scalar()

        values = {
            "native_transaction_id": native_tx_id,
            "issued_at": text("now() AT TIME ZONE 'UTC'"),
            "client_addr": self.get_client_addr(),
            "actor_id": self.get_actor_id(),
        }

        stmt = (
            insert(self.transaction_cls)
            .values(**values)
            .on_conflict_do_nothing(constraint="transaction_unique_native_tx_id")
        )

        # Execute on secondary database if configured, otherwise use main session
        try:
            if self._audit_engine and self._audit_session_factory:
                # Use separate session for secondary audit database
                audit_session = self._audit_session_factory()
                try:
                    audit_session.execute(stmt)
                    audit_session.commit()
                except Exception as audit_error:
                    audit_session.rollback()
                    raise audit_error
                finally:
                    audit_session.close()
            else:
                # Use main database session
                session.execute(stmt)
        except Exception as e:
            # Log the error but don't fail the main transaction
            logger.error(f"Failed to save audit transaction: {e}", exc_info=True)
            # Don't re-raise - audit logging failures should not break the application

    @property
    def actor_cls(self):
        if isinstance(self._actor_cls, str):
            if not self.db.Model:
                raise ImproperlyConfigured("No SQLAlchemy db object")
            registry = self.db.Model.registry._class_registry
            try:
                return registry[self._actor_cls]
            except KeyError:
                raise ImproperlyConfigured(
                    f"""Could not build relationship between AuditLogActivity
                    and {self._actor_cls}. {self._actor_cls} was not found in
                    declarative class registry. Either configure AuditLogger to
                    use different actor class or disable this relationship by
                    setting it to None."""
                )
        return self._actor_cls


def _transaction_model_factory(base, schema, actor_cls):
    if actor_cls:
        actor_pk = inspect(actor_cls).primary_key[0]
        actor_fk = ForeignKey(f"{actor_cls.__table__.name}.{actor_pk.name}")

    class AuditLogTransaction(base):
        __tablename__ = "transaction"

        id = Column(BigInteger, primary_key=True)
        native_transaction_id = Column(BigInteger)
        issued_at = Column(DateTime)
        client_addr = Column(INET)
        if actor_cls:
            actor_id = Column(actor_pk.type, actor_fk)
            actor = relationship(actor_cls)
        else:
            actor_id = Column(Text)

        __table_args__ = (
            ExcludeConstraint(
                (literal_column("native_transaction_id"), "="),
                (
                    literal_column("tsrange(issued_at - INTERVAL '1 HOUR', issued_at)"),
                    "&&",
                ),
                name="transaction_unique_native_tx_id",
            ),
            {"schema": schema, "extend_existing": True},
        )

        def __repr__(self):
            return "<{cls} id={id!r} issued_at={issued_at!r}>".format(
                cls=self.__class__.__name__, id=self.id, issued_at=self.issued_at
            )

    return AuditLogTransaction


def _activity_model_factory(base, schema_name, transaction_cls):
    class AuditLogActivity(base):
        __tablename__ = "activity"
        __table_args__ = {"schema": schema_name, "extend_existing": True}

        id = Column(BigInteger, primary_key=True)
        schema = Column(Text)
        table_name = Column(Text)
        relid = Column(Integer)
        issued_at = Column(DateTime)
        native_transaction_id = Column(BigInteger, index=True)
        verb = Column(Text)
        old_data = Column(JSONB, default={}, server_default="{}")
        changed_data = Column(JSONB, default={}, server_default="{}")
        transaction_id = Column(BigInteger, ForeignKey(transaction_cls.id))
        record_id = Column(Text, index=True)

        transaction = relationship(transaction_cls, backref="activities")

        @hybrid_property
        def data(self):
            data = self.old_data.copy() if self.old_data else {}
            if self.changed_data:
                data.update(self.changed_data)
            return data

        @data.expression
        def data(cls):
            return cls.old_data + cls.changed_data

        def __repr__(self):
            return ("<{cls} table_name={table_name!r} " "id={id!r}>").format(
                cls=self.__class__.__name__, table_name=self.table_name, id=self.id
            )

    return AuditLogActivity


def _read_file(file):
    with open(os.path.join(HERE, file)) as f:
        s = f.read()
    return s


def _default_actor_id():
    try:
        from flask_login import current_user
    except ImportError:
        return None

    try:
        return current_user.id
    except AttributeError:
        return None


def _default_client_addr():
    # Return None if we are outside of request context.
    return (request and request.remote_addr) or None


def _detect_versioned_tables(db: SQLAlchemy) -> set[Table]:
    versioned_tables = set()

    for table in db.metadata.tables.values():
        if table.info.get("versioned") is not None:
            versioned_tables.add(table)

    return versioned_tables


def _is_session_modified(session: Session, versioned_tables: set[Table]) -> bool:
    return any(
        _is_entity_modified(entity) or entity in session.deleted
        for entity in session
        if entity.__table__ in versioned_tables
    )


def _is_entity_modified(entity) -> bool:
    versioned = entity.__table__.info.get("versioned")
    excluded_cols = set(versioned.get("exclude", []))
    modified_cols = {column.name for column in _modified_columns(entity)}

    return bool(modified_cols - excluded_cols)


def _modified_columns(obj):
    columns = set()
    mapper = inspect(obj.__class__)
    for key, attr in inspect(obj).attrs.items():
        if key in mapper.synonyms.keys():
            continue
        prop = getattr(obj.__class__, key).property
        if attr.history.has_changes():
            columns |= set(
                prop.columns
                if isinstance(prop, ColumnProperty)
                else [local for local, remote in prop.local_remote_pairs]
            )

    return columns
