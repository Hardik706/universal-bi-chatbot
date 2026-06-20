from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

def get_db_schema(engine_or_uri: Engine | str) -> str:
    """
    Dynamically extracts the schema of the connected database using SQLAlchemy inspection.
    Returns a markdown-formatted string listing tables, columns, types, primary keys, and foreign keys.
    """
    if isinstance(engine_or_uri, str):
        try:
            engine = create_engine(engine_or_uri)
        except Exception as e:
            logger.error(f"Failed to create engine from database URI: {e}")
            return f"Error connecting to database: {str(e)}"
    else:
        engine = engine_or_uri

    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if not table_names:
            return "No tables found in the connected database."

        schema_repr = []
        schema_repr.append("# Database Schema Reference\n")

        for table in table_names:
            schema_repr.append(f"## Table: `{table}`")
            
            # Columns
            columns = inspector.get_columns(table)
            schema_repr.append("### Columns:")
            for col in columns:
                col_name = col['name']
                col_type = str(col['type'])
                is_nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                default_val = f" DEFAULT {col.get('default')}" if col.get('default') is not None else ""
                
                schema_repr.append(f"- `{col_name}` ({col_type}) - {is_nullable}{default_val}")

            # Primary Keys
            pk_constraint = inspector.get_pk_constraint(table)
            pk_cols = pk_constraint.get("constrained_columns", [])
            if pk_cols:
                schema_repr.append(f"**Primary Key**: `{', '.join(pk_cols)}`")

            # Foreign Keys
            fks = inspector.get_foreign_keys(table)
            if fks:
                schema_repr.append("### Foreign Keys:")
                for fk in fks:
                    constrained = fk.get("constrained_columns", [])
                    referred_table = fk.get("referred_table")
                    referred_columns = fk.get("referred_columns", [])
                    schema_repr.append(
                        f"- `{', '.join(constrained)}` references `{referred_table}({', '.join(referred_columns)})`"
                    )
            
            schema_repr.append("\n" + "-"*40 + "\n")

        return "\n".join(schema_repr)

    except Exception as e:
        logger.exception("Error extracting database schema")
        return f"Error extracting schema: {str(e)}"
