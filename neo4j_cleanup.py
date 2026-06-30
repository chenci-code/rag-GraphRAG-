from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from mysql_metadata import MySQLPermissionStore


@dataclass
class GraphCleanupPlan:
    tenant_id: str
    file_id: str
    node_ids: list[str]
    relationship_ids: list[str]


class Neo4jFileCleaner:
    def __init__(self, enabled: bool, uri: str = "", user: str = "", password: str = ""):
        self.enabled = enabled
        self.uri = uri
        self.user = user
        self.password = password

    @classmethod
    def from_env(cls) -> "Neo4jFileCleaner":
        enabled = os.getenv("GRAPHRAG_NEO4J_CLEANUP", "off").lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            uri=os.getenv("GRAPHRAG_NEO4J_URI", "bolt://127.0.0.1:7687"),
            user=os.getenv("GRAPHRAG_NEO4J_USER", "neo4j"),
            password=os.getenv("GRAPHRAG_NEO4J_PASSWORD", ""),
        )

    def cleanup_file(self, tenant_id: str, file_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "nodes": 0, "relationships": 0}

        plan = load_graph_cleanup_plan(tenant_id=tenant_id, file_id=file_id)
        if not plan.node_ids and not plan.relationship_ids:
            return {"enabled": True, "nodes": 0, "relationships": 0}

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j package is required for Neo4j file cleanup.") from exc

        node_cypher, relation_cypher = cleanup_cypher()
        with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
            with driver.session() as session:
                if plan.node_ids:
                    session.run(
                        node_cypher,
                        tenant_id=tenant_id,
                        target_file_id=file_id,
                        node_ids=plan.node_ids,
                    ).consume()
                if plan.relationship_ids:
                    session.run(
                        relation_cypher,
                        tenant_id=tenant_id,
                        target_file_id=file_id,
                        relationship_ids=plan.relationship_ids,
                    ).consume()

        return {
            "enabled": True,
            "nodes": len(plan.node_ids),
            "relationships": len(plan.relationship_ids),
        }


def load_graph_cleanup_plan(tenant_id: str, file_id: str) -> GraphCleanupPlan:
    store = MySQLPermissionStore.from_env()
    if not store.enabled:
        return GraphCleanupPlan(tenant_id=tenant_id, file_id=file_id, node_ids=[], relationship_ids=[])

    sql = """
        SELECT element_type, element_id
        FROM rag_graph_element_mapping
        WHERE tenant_id = %s AND file_id = %s
    """
    node_ids: set[str] = set()
    relationship_ids: set[str] = set()
    with store._connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (tenant_id, file_id))
            for element_type, element_id in cursor.fetchall():
                if str(element_type).lower() == "node":
                    node_ids.add(str(element_id))
                elif str(element_type).lower() == "relationship":
                    relationship_ids.add(str(element_id))

    return GraphCleanupPlan(
        tenant_id=tenant_id,
        file_id=file_id,
        node_ids=sorted(node_ids),
        relationship_ids=sorted(relationship_ids),
    )


def cleanup_cypher() -> tuple[str, str]:
    node_cypher = """
        MATCH (n:Entity {tenant_id: $tenant_id})
        WHERE n.element_id IN $node_ids
        SET n.file_ids = [x IN coalesce(n.file_ids, []) WHERE x <> $target_file_id]
        WITH n
        WHERE size(coalesce(n.file_ids, [])) = 0
        DETACH DELETE n
    """
    relation_cypher = """
        MATCH (:Entity {tenant_id: $tenant_id})-[r]->(:Entity {tenant_id: $tenant_id})
        WHERE r.element_id IN $relationship_ids AND r.tenant_id = $tenant_id
        SET r.file_ids = [x IN coalesce(r.file_ids, []) WHERE x <> $target_file_id]
        WITH r
        WHERE size(coalesce(r.file_ids, [])) = 0
        DELETE r
    """
    return node_cypher.strip(), relation_cypher.strip()
