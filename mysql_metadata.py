from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


class MySQLPermissionStore:
    def __init__(self, enabled: bool, connect_kwargs: dict[str, Any] | None = None):
        self.enabled = enabled
        self.connect_kwargs = connect_kwargs or {}

    @classmethod
    def from_env(cls) -> "MySQLPermissionStore":
        if os.getenv("GRAPHRAG_PERMISSION_MODE", "off").lower() != "mysql":
            return cls(enabled=False)
        return cls(enabled=True, connect_kwargs=_mysql_connect_kwargs_from_env())

    def allowed_text_unit_ids(
        self,
        tenant_id: str | None,
        user_id: str | None,
        department_id: str | None,
        role_ids: list[str] | None = None,
    ) -> set[str]:
        if not self.enabled:
            return set()

        role_ids = role_ids or []
        principal_clauses = []
        principal_params: list[Any] = []

        if user_id:
            principal_clauses.append("(rp.principal_type = 'user' AND rp.principal_id = %s)")
            principal_params.append(user_id)
        if department_id:
            principal_clauses.append(
                "(rp.principal_type = 'department' AND rp.principal_id = %s)"
            )
            principal_params.append(department_id)
        for role_id in role_ids:
            principal_clauses.append("(rp.principal_type = 'role' AND rp.principal_id = %s)")
            principal_params.append(role_id)

        explicit_permission_clause = "FALSE"
        if principal_clauses:
            explicit_permission_clause = (
                "EXISTS ("
                "SELECT 1 FROM rag_resource_permissions rp "
                "WHERE rp.can_read = 1 "
                "AND rp.tenant_id = tu.tenant_id "
                "AND ("
                "  (rp.resource_type = 'text_unit' AND rp.resource_id = tu.text_unit_id) "
                "  OR (rp.resource_type = 'document' AND rp.resource_id = tu.document_id)"
                ") "
                f"AND ({' OR '.join(principal_clauses)})"
                ")"
            )

        visibility_clause = ["tu.visibility = 'public'"]
        visibility_params: list[Any] = []
        if department_id:
            visibility_clause.append(
                "(tu.visibility = 'department' AND tu.department_id = %s)"
            )
            visibility_params.append(department_id)
        if user_id:
            visibility_clause.append("(tu.visibility = 'private' AND tu.owner_user_id = %s)")
            visibility_params.append(user_id)

        tenant_clause = ""
        tenant_params: list[Any] = []
        if tenant_id:
            tenant_clause = "tu.tenant_id = %s AND "
            tenant_params.append(tenant_id)

        sql = (
            "SELECT tu.text_unit_id "
            "FROM rag_text_units tu "
            f"WHERE {tenant_clause}({' OR '.join(visibility_clause)} OR {explicit_permission_clause})"
        )

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, [*tenant_params, *visibility_params, *principal_params])
                return {str(row[0]) for row in cursor.fetchall()}

    def _connect(self) -> Any:
        try:
            import pymysql
        except ImportError as exc:
            msg = (
                "PyMySQL is required for MySQL permission mode. Install it with "
                "`pip install pymysql` in rag_env."
            )
            raise RuntimeError(msg) from exc

        return pymysql.connect(
            charset="utf8mb4",
            cursorclass=pymysql.cursors.Cursor,
            autocommit=True,
            **self.connect_kwargs,
        )


def sync_graphrag_metadata_to_mysql(
    root: Path,
    default_tenant_id: str = "default",
    default_department_id: str = "default",
    default_visibility: str = "department",
    overrides_path: Path | None = None,
) -> dict[str, int]:
    store = MySQLPermissionStore(enabled=True, connect_kwargs=_mysql_connect_kwargs_from_env())
    overrides = _load_overrides(overrides_path)

    output_dir = root / "output"
    documents = pd.read_parquet(output_dir / "documents.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")

    document_entities = _document_entities(documents, text_units, entities)
    text_unit_entities = _text_unit_entities(text_units, entities)
    document_overrides = _document_override_map(documents, overrides.get("documents", {}))
    document_file_ids = _document_file_id_map(documents, document_overrides)

    with store._connect() as conn:
        with conn.cursor() as cursor:
            doc_count = _upsert_documents(
                cursor=cursor,
                root=root,
                documents=documents,
                document_entities=document_entities,
                document_overrides=document_overrides,
                document_file_ids=document_file_ids,
                default_tenant_id=default_tenant_id,
                default_department_id=default_department_id,
                default_visibility=default_visibility,
            )
            text_count = _upsert_text_units(
                cursor=cursor,
                text_units=text_units,
                text_unit_entities=text_unit_entities,
                document_overrides=document_overrides,
                text_unit_overrides=overrides.get("text_units", {}),
                document_file_ids=document_file_ids,
                default_tenant_id=default_tenant_id,
                default_department_id=default_department_id,
                default_visibility=default_visibility,
            )
            graph_count = _sync_graph_element_mapping(
                cursor=cursor,
                documents=documents,
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                document_file_ids=document_file_ids,
                document_overrides=document_overrides,
                default_tenant_id=default_tenant_id,
            )
            prune_stats = _prune_stale_metadata(
                cursor=cursor,
                documents=documents,
                text_units=text_units,
                document_overrides=document_overrides,
                default_tenant_id=default_tenant_id,
            )

    return {
        "documents": doc_count,
        "text_units": text_count,
        "graph_elements": graph_count,
        **prune_stats,
    }


def _mysql_connect_kwargs_from_env() -> dict[str, Any]:
    dsn = os.getenv("GRAPHRAG_MYSQL_DSN")
    if dsn:
        parsed = urlparse(dsn)
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": parsed.username or "",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/"),
        }

    return {
        "host": os.getenv("GRAPHRAG_MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("GRAPHRAG_MYSQL_PORT", "3306")),
        "user": os.getenv("GRAPHRAG_MYSQL_USER", "root"),
        "password": os.getenv("GRAPHRAG_MYSQL_PASSWORD", ""),
        "database": os.getenv("GRAPHRAG_MYSQL_DATABASE", "graphrag"),
    }


def _load_overrides(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"documents": {}, "text_units": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _document_override_map(
    documents: pd.DataFrame,
    raw_overrides: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in documents.to_dict(orient="records"):
        doc_id = str(row["id"])
        title = str(row["title"])
        result[doc_id] = raw_overrides.get(doc_id) or raw_overrides.get(title, {})
    return result


def _document_file_id_map(
    documents: pd.DataFrame,
    document_overrides: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in documents.to_dict(orient="records"):
        doc_id = str(row["id"])
        title = str(row["title"])
        override = document_overrides.get(doc_id, {})
        result[doc_id] = str(override.get("file_id", title))
    return result


def _document_entities(
    documents: pd.DataFrame,
    text_units: pd.DataFrame,
    entities: pd.DataFrame,
) -> dict[str, list[str]]:
    text_unit_to_doc: dict[str, str] = {}
    for row in text_units.to_dict(orient="records"):
        text_unit_to_doc[str(row["id"])] = str(row["document_id"])

    result: dict[str, set[str]] = {str(row["id"]): set() for row in documents.to_dict(orient="records")}
    for row in entities.to_dict(orient="records"):
        title = str(row.get("title", ""))
        for text_unit_id in _as_list(row.get("text_unit_ids")):
            doc_id = text_unit_to_doc.get(str(text_unit_id))
            if doc_id:
                result.setdefault(doc_id, set()).add(title)
    return {doc_id: sorted(values) for doc_id, values in result.items()}


def _text_unit_entities(
    text_units: pd.DataFrame,
    entities: pd.DataFrame,
) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {str(row["id"]): set() for row in text_units.to_dict(orient="records")}
    for row in entities.to_dict(orient="records"):
        title = str(row.get("title", ""))
        for text_unit_id in _as_list(row.get("text_unit_ids")):
            result.setdefault(str(text_unit_id), set()).add(title)
    return {text_unit_id: sorted(values) for text_unit_id, values in result.items()}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _upsert_documents(
    cursor: Any,
    root: Path,
    documents: pd.DataFrame,
    document_entities: dict[str, list[str]],
    document_overrides: dict[str, Any],
    document_file_ids: dict[str, str],
    default_tenant_id: str,
    default_department_id: str,
    default_visibility: str,
) -> int:
    sql = """
        INSERT INTO rag_documents (
            document_id, tenant_id, file_id, human_readable_id, title, file_path, department_id,
            visibility, owner_user_id, keywords, entity_names, tags, search_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            tenant_id = VALUES(tenant_id),
            file_id = VALUES(file_id),
            human_readable_id = VALUES(human_readable_id),
            title = VALUES(title),
            file_path = VALUES(file_path),
            department_id = VALUES(department_id),
            visibility = VALUES(visibility),
            owner_user_id = VALUES(owner_user_id),
            keywords = VALUES(keywords),
            entity_names = VALUES(entity_names),
            tags = VALUES(tags),
            search_text = VALUES(search_text)
    """
    count = 0
    for row in documents.to_dict(orient="records"):
        doc_id = str(row["id"])
        override = document_overrides.get(doc_id, {})
        entity_names = document_entities.get(doc_id, [])
        keywords = override.get("keywords", entity_names[:20])
        tags = override.get("tags", [])
        title = str(row["title"])
        cursor.execute(
            sql,
            (
                doc_id,
                str(override.get("tenant_id", default_tenant_id)),
                document_file_ids.get(doc_id, title),
                str(row.get("human_readable_id", "")),
                title,
                str(root / "input" / title),
                str(override.get("department_id", default_department_id)),
                str(override.get("visibility", default_visibility)),
                override.get("owner_user_id"),
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(entity_names, ensure_ascii=False),
                json.dumps(tags, ensure_ascii=False),
                "\n".join([title, str(row.get("text", "")), " ".join(entity_names)]),
            ),
        )
        count += 1
    return count


def _upsert_text_units(
    cursor: Any,
    text_units: pd.DataFrame,
    text_unit_entities: dict[str, list[str]],
    document_overrides: dict[str, Any],
    text_unit_overrides: dict[str, Any],
    document_file_ids: dict[str, str],
    default_tenant_id: str,
    default_department_id: str,
    default_visibility: str,
) -> int:
    sql = """
        INSERT INTO rag_text_units (
            text_unit_id, tenant_id, file_id, human_readable_id, document_id, department_id,
            visibility, owner_user_id, keywords, entity_names, search_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            tenant_id = VALUES(tenant_id),
            file_id = VALUES(file_id),
            human_readable_id = VALUES(human_readable_id),
            document_id = VALUES(document_id),
            department_id = VALUES(department_id),
            visibility = VALUES(visibility),
            owner_user_id = VALUES(owner_user_id),
            keywords = VALUES(keywords),
            entity_names = VALUES(entity_names),
            search_text = VALUES(search_text)
    """
    count = 0
    for row in text_units.to_dict(orient="records"):
        text_unit_id = str(row["id"])
        document_id = str(row["document_id"])
        doc_override = document_overrides.get(document_id, {})
        override = text_unit_overrides.get(text_unit_id, {})
        entity_names = text_unit_entities.get(text_unit_id, [])
        keywords = override.get("keywords", doc_override.get("keywords", entity_names[:20]))
        cursor.execute(
            sql,
            (
                text_unit_id,
                str(override.get("tenant_id", doc_override.get("tenant_id", default_tenant_id))),
                document_file_ids.get(document_id, document_id),
                str(row.get("human_readable_id", "")),
                document_id,
                str(override.get("department_id", doc_override.get("department_id", default_department_id))),
                str(override.get("visibility", doc_override.get("visibility", default_visibility))),
                override.get("owner_user_id", doc_override.get("owner_user_id")),
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(entity_names, ensure_ascii=False),
                "\n".join([str(row.get("text", "")), " ".join(entity_names)]),
            ),
        )
        count += 1
    return count


def _sync_graph_element_mapping(
    cursor: Any,
    documents: pd.DataFrame,
    text_units: pd.DataFrame,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    document_file_ids: dict[str, str],
    document_overrides: dict[str, Any],
    default_tenant_id: str,
) -> int:
    text_unit_to_document = {
        str(row["id"]): str(row["document_id"])
        for row in text_units.to_dict(orient="records")
    }
    document_tenant_ids = {
        str(row["id"]): str(document_overrides.get(str(row["id"]), {}).get("tenant_id", default_tenant_id))
        for row in documents.to_dict(orient="records")
    }
    tenant_ids = sorted(set(document_tenant_ids.values()) or {default_tenant_id})
    delete_placeholders = ", ".join(["%s"] * len(tenant_ids))
    cursor.execute(
        f"DELETE FROM rag_graph_element_mapping WHERE tenant_id IN ({delete_placeholders})",
        tenant_ids,
    )

    sql = """
        INSERT IGNORE INTO rag_graph_element_mapping (
            tenant_id, file_id, document_id, text_unit_id, element_type,
            element_id, element_name, source_name, target_name, relationship_description
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    count = 0
    for row in entities.to_dict(orient="records"):
        entity_id = str(row["id"])
        entity_name = str(row.get("title", ""))
        for text_unit_id in _as_list(row.get("text_unit_ids")):
            doc_id = text_unit_to_document.get(str(text_unit_id))
            if doc_id is None:
                continue
            cursor.execute(
                sql,
                (
                    document_tenant_ids.get(doc_id, default_tenant_id),
                    document_file_ids.get(doc_id, doc_id),
                    doc_id,
                    str(text_unit_id),
                    "node",
                    entity_id,
                    entity_name,
                    None,
                    None,
                    None,
                ),
            )
            count += int(cursor.rowcount)

    for row in relationships.to_dict(orient="records"):
        relationship_id = str(row["id"])
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        description = str(row.get("description", ""))
        for text_unit_id in _as_list(row.get("text_unit_ids")):
            doc_id = text_unit_to_document.get(str(text_unit_id))
            if doc_id is None:
                continue
            cursor.execute(
                sql,
                (
                    document_tenant_ids.get(doc_id, default_tenant_id),
                    document_file_ids.get(doc_id, doc_id),
                    doc_id,
                    str(text_unit_id),
                    "relationship",
                    relationship_id,
                    None,
                    source,
                    target,
                    description,
                ),
            )
            count += int(cursor.rowcount)

    return count


def _prune_stale_metadata(
    cursor: Any,
    documents: pd.DataFrame,
    text_units: pd.DataFrame,
    document_overrides: dict[str, Any],
    default_tenant_id: str,
) -> dict[str, int]:
    document_ids = {str(row["id"]) for row in documents.to_dict(orient="records")}
    text_unit_ids = {str(row["id"]) for row in text_units.to_dict(orient="records")}
    tenant_ids = sorted(
        {
            str(document_overrides.get(document_id, {}).get("tenant_id", default_tenant_id))
            for document_id in document_ids
        }
        or {default_tenant_id}
    )

    stale_text_unit_ids = _select_stale_ids(
        cursor=cursor,
        table="rag_text_units",
        id_column="text_unit_id",
        tenant_ids=tenant_ids,
        current_ids=text_unit_ids,
    )
    stale_document_ids = _select_stale_ids(
        cursor=cursor,
        table="rag_documents",
        id_column="document_id",
        tenant_ids=tenant_ids,
        current_ids=document_ids,
    )

    deleted_permissions = 0
    if stale_text_unit_ids:
        deleted_permissions += _delete_resource_permissions(
            cursor=cursor,
            tenant_ids=tenant_ids,
            resource_type="text_unit",
            resource_ids=stale_text_unit_ids,
        )
    if stale_document_ids:
        deleted_permissions += _delete_resource_permissions(
            cursor=cursor,
            tenant_ids=tenant_ids,
            resource_type="document",
            resource_ids=stale_document_ids,
        )

    deleted_text_units = _delete_ids(
        cursor=cursor,
        table="rag_text_units",
        id_column="text_unit_id",
        tenant_ids=tenant_ids,
        ids=stale_text_unit_ids,
    )
    deleted_documents = _delete_ids(
        cursor=cursor,
        table="rag_documents",
        id_column="document_id",
        tenant_ids=tenant_ids,
        ids=stale_document_ids,
    )

    return {
        "deleted_documents": deleted_documents,
        "deleted_text_units": deleted_text_units,
        "deleted_permissions": deleted_permissions,
    }


def _select_stale_ids(
    cursor: Any,
    table: str,
    id_column: str,
    tenant_ids: list[str],
    current_ids: set[str],
) -> list[str]:
    tenant_placeholders = ", ".join(["%s"] * len(tenant_ids))
    params: list[Any] = list(tenant_ids)
    current_clause = ""
    if current_ids:
        current_placeholders = ", ".join(["%s"] * len(current_ids))
        current_clause = f" AND {id_column} NOT IN ({current_placeholders})"
        params.extend(sorted(current_ids))
    cursor.execute(
        f"SELECT {id_column} FROM {table} WHERE tenant_id IN ({tenant_placeholders}){current_clause}",
        params,
    )
    return [str(row[0]) for row in cursor.fetchall()]


def _delete_resource_permissions(
    cursor: Any,
    tenant_ids: list[str],
    resource_type: str,
    resource_ids: list[str],
) -> int:
    if not resource_ids:
        return 0
    tenant_placeholders = ", ".join(["%s"] * len(tenant_ids))
    resource_placeholders = ", ".join(["%s"] * len(resource_ids))
    cursor.execute(
        (
            "DELETE FROM rag_resource_permissions "
            f"WHERE tenant_id IN ({tenant_placeholders}) "
            "AND resource_type = %s "
            f"AND resource_id IN ({resource_placeholders})"
        ),
        [*tenant_ids, resource_type, *resource_ids],
    )
    return int(cursor.rowcount)


def _delete_ids(
    cursor: Any,
    table: str,
    id_column: str,
    tenant_ids: list[str],
    ids: list[str],
) -> int:
    if not ids:
        return 0
    tenant_placeholders = ", ".join(["%s"] * len(tenant_ids))
    id_placeholders = ", ".join(["%s"] * len(ids))
    cursor.execute(
        f"DELETE FROM {table} WHERE tenant_id IN ({tenant_placeholders}) AND {id_column} IN ({id_placeholders})",
        [*tenant_ids, *ids],
    )
    return int(cursor.rowcount)
