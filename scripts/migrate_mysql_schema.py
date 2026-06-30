from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mysql_metadata import _mysql_connect_kwargs_from_env


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)

    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("PyMySQL is required. Install it in rag_env.") from exc

    conn = pymysql.connect(
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        **_mysql_connect_kwargs_from_env(),
    )
    try:
        with conn.cursor() as cursor:
            _migrate(cursor)
    finally:
        conn.close()


def _migrate(cursor: Any) -> None:
    _add_column(
        cursor,
        "rag_departments",
        "tenant_id",
        "tenant_id VARCHAR(64) NOT NULL DEFAULT 'default' AFTER department_id",
    )
    _add_index(
        cursor,
        "rag_departments",
        "idx_rag_departments_tenant",
        "INDEX idx_rag_departments_tenant (tenant_id)",
    )

    _add_column(
        cursor,
        "rag_users",
        "tenant_id",
        "tenant_id VARCHAR(64) NOT NULL DEFAULT 'default' AFTER user_id",
    )
    _add_index(
        cursor,
        "rag_users",
        "idx_rag_users_tenant",
        "INDEX idx_rag_users_tenant (tenant_id)",
    )

    _add_column(
        cursor,
        "rag_documents",
        "tenant_id",
        "tenant_id VARCHAR(64) NOT NULL DEFAULT 'default' AFTER document_id",
    )
    _add_column(
        cursor,
        "rag_documents",
        "file_id",
        "file_id VARCHAR(128) NULL AFTER tenant_id",
    )
    cursor.execute(
        "UPDATE rag_documents "
        "SET file_id = COALESCE(NULLIF(file_id, ''), title, document_id) "
        "WHERE file_id IS NULL OR file_id = ''"
    )
    cursor.execute("ALTER TABLE rag_documents MODIFY file_id VARCHAR(128) NOT NULL")
    _add_index(
        cursor,
        "rag_documents",
        "uk_rag_documents_tenant_file",
        "UNIQUE KEY uk_rag_documents_tenant_file (tenant_id, file_id)",
    )
    _add_index(
        cursor,
        "rag_documents",
        "idx_rag_documents_tenant",
        "INDEX idx_rag_documents_tenant (tenant_id)",
    )
    _add_index(
        cursor,
        "rag_documents",
        "idx_rag_documents_file",
        "INDEX idx_rag_documents_file (file_id)",
    )

    _add_column(
        cursor,
        "rag_text_units",
        "tenant_id",
        "tenant_id VARCHAR(64) NOT NULL DEFAULT 'default' AFTER text_unit_id",
    )
    _add_column(
        cursor,
        "rag_text_units",
        "file_id",
        "file_id VARCHAR(128) NULL AFTER tenant_id",
    )
    cursor.execute(
        """
        UPDATE rag_text_units tu
        LEFT JOIN rag_documents d ON d.document_id = tu.document_id
        SET tu.file_id = COALESCE(NULLIF(tu.file_id, ''), d.file_id, tu.document_id)
        WHERE tu.file_id IS NULL OR tu.file_id = ''
        """
    )
    cursor.execute("ALTER TABLE rag_text_units MODIFY file_id VARCHAR(128) NOT NULL")
    _add_index(
        cursor,
        "rag_text_units",
        "idx_rag_text_units_tenant",
        "INDEX idx_rag_text_units_tenant (tenant_id)",
    )
    _add_index(
        cursor,
        "rag_text_units",
        "idx_rag_text_units_file",
        "INDEX idx_rag_text_units_file (file_id)",
    )

    _add_column(
        cursor,
        "rag_resource_permissions",
        "tenant_id",
        "tenant_id VARCHAR(64) NOT NULL DEFAULT 'default' AFTER permission_id",
    )
    _add_index(
        cursor,
        "rag_resource_permissions",
        "idx_rag_resource_permissions_tenant",
        "INDEX idx_rag_resource_permissions_tenant (tenant_id)",
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_graph_element_mapping (
          mapping_id BIGINT PRIMARY KEY AUTO_INCREMENT,
          tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
          file_id VARCHAR(128) NOT NULL,
          document_id VARCHAR(128) NOT NULL,
          text_unit_id VARCHAR(128) NULL,
          element_type ENUM('node', 'relationship') NOT NULL,
          element_id VARCHAR(128) NOT NULL,
          element_name VARCHAR(500) NULL,
          source_name VARCHAR(500) NULL,
          target_name VARCHAR(500) NULL,
          relationship_description TEXT NULL,
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_rag_graph_mapping (
            tenant_id,
            file_id,
            element_type,
            element_id,
            text_unit_id
          ),
          INDEX idx_rag_graph_mapping_tenant_file (tenant_id, file_id),
          INDEX idx_rag_graph_mapping_element (element_type, element_id),
          INDEX idx_rag_graph_mapping_document (document_id),
          INDEX idx_rag_graph_mapping_text_unit (text_unit_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    print("ensured table rag_graph_element_mapping")


def _table_exists(cursor: Any, table: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table,),
    )
    return int(cursor.fetchone()["n"]) > 0


def _column_exists(cursor: Any, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return int(cursor.fetchone()["n"]) > 0


def _index_exists(cursor: Any, table: str, index: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.statistics
        WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s
        """,
        (table, index),
    )
    return int(cursor.fetchone()["n"]) > 0


def _add_column(cursor: Any, table: str, column: str, ddl: str) -> None:
    if not _table_exists(cursor, table) or _column_exists(cursor, table, column):
        print(f"kept column {table}.{column}")
        return
    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
    print(f"added column {table}.{column}")


def _add_index(cursor: Any, table: str, index: str, ddl: str) -> None:
    if not _table_exists(cursor, table) or _index_exists(cursor, table, index):
        print(f"kept index {table}.{index}")
        return
    cursor.execute(f"ALTER TABLE {table} ADD {ddl}")
    print(f"added index {table}.{index}")


if __name__ == "__main__":
    main()
