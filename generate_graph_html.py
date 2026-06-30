from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
HTML_PATH = OUTPUT_DIR / "graphrag_visualization.html"
VIS_JS_PATH = OUTPUT_DIR / "assets" / "vis-network.min.js"


TYPE_COLORS = {
    "PERSON": "#d9485f",
    "ORGANIZATION": "#2f6fed",
    "GEO": "#2a9d8f",
    "EVENT": "#f4a261",
    "UNKNOWN": "#7a7f87",
}


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return str(value)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entities = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    relationships = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
    entity_embeddings = pd.read_parquet(OUTPUT_DIR / "embeddings.entity_description.parquet")
    return entities, relationships, entity_embeddings


def build_knowledge_graph(
    entities: pd.DataFrame, relationships: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    title_to_id = dict(zip(entities["title"], entities["id"], strict=True))

    for row in entities.to_dict(orient="records"):
        entity_type = _safe_text(row.get("type"), "UNKNOWN").upper()
        degree = int(row.get("degree", 1) or 1)
        freq = int(row.get("frequency", 1) or 1)
        description = _safe_text(row.get("description"))
        nodes.append({
            "id": row["id"],
            "label": _safe_text(row.get("title")),
            "group": entity_type,
            "color": TYPE_COLORS.get(entity_type, TYPE_COLORS["UNKNOWN"]),
            "value": max(10, degree * 2 + freq),
            "title": (
                f"<b>{_safe_text(row.get('title'))}</b><br>"
                f"Type: {entity_type}<br>"
                f"Degree: {degree}<br>"
                f"Frequency: {freq}<br><br>"
                f"{description}"
            ),
        })

    for idx, row in enumerate(relationships.to_dict(orient="records")):
        source_key = _safe_text(row.get("source"))
        target_key = _safe_text(row.get("target"))
        source_id = title_to_id.get(source_key)
        target_id = title_to_id.get(target_key)
        if not source_id or not target_id or source_id == target_id:
            continue

        weight = float(row.get("weight", 1.0) or 1.0)
        edges.append({
            "id": f"kg-{idx}",
            "from": source_id,
            "to": target_id,
            "value": max(1.0, weight),
            "width": min(10, 1 + weight / 2),
            "color": {"color": "rgba(120, 120, 120, 0.55)"},
            "title": (
                f"<b>{source_key} → {target_key}</b><br>"
                f"Weight: {weight:.2f}<br><br>"
                f"{_safe_text(row.get('description'))}"
            ),
        })

    return nodes, edges


def build_vector_similarity_graph(
    entities: pd.DataFrame, entity_embeddings: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = entities.merge(entity_embeddings, on="id", how="inner")
    vectors = np.stack(merged["embedding"].to_list()).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    similarity = normalized @ normalized.T

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for row in merged.to_dict(orient="records"):
        entity_type = _safe_text(row.get("type"), "UNKNOWN").upper()
        degree = int(row.get("degree", 1) or 1)
        nodes.append({
            "id": row["id"],
            "label": _safe_text(row.get("title")),
            "group": entity_type,
            "color": TYPE_COLORS.get(entity_type, TYPE_COLORS["UNKNOWN"]),
            "value": max(10, degree * 2),
            "title": (
                f"<b>{_safe_text(row.get('title'))}</b><br>"
                f"Type: {entity_type}<br><br>"
                f"{_safe_text(row.get('description'))}"
            ),
        })

    top_k = 3
    threshold = 0.78
    seen_pairs: set[tuple[str, str]] = set()
    titles = merged["title"].tolist()
    ids = merged["id"].tolist()

    for i in range(len(ids)):
        ranked = np.argsort(similarity[i])[::-1]
        added = 0
        for j in ranked:
            if i == j:
                continue
            score = float(similarity[i, j])
            if score < threshold:
                continue
            pair = tuple(sorted((ids[i], ids[j])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            added += 1
            edges.append({
                "id": f"vec-{len(edges)}",
                "from": ids[i],
                "to": ids[j],
                "value": score,
                "width": 1 + (score - threshold) * 12,
                "color": {"color": "rgba(47, 111, 237, 0.42)"},
                "title": (
                    f"<b>{titles[i]} ↔ {titles[j]}</b><br>"
                    f"Cosine similarity: {score:.4f}"
                ),
            })
            if added >= top_k:
                break

    return nodes, edges


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    vis_js = VIS_JS_PATH.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GraphRAG Visualization</title>
  <script>
{vis_js}
  </script>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: rgba(255,255,255,0.86);
      --text: #1d2a36;
      --muted: #5d6a75;
      --accent: #c45b3c;
      --border: rgba(29,42,54,0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(196,91,60,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(47,111,237,0.12), transparent 26%),
        linear-gradient(135deg, #f7f2ea, #eef3f7 48%, #f8efe5);
      min-height: 100vh;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 45px rgba(23, 35, 48, 0.08);
    }}
    .sidebar {{
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .title {{
      font-size: 28px;
      line-height: 1.1;
      margin: 0;
      letter-spacing: 0.02em;
    }}
    .subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .toggle-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .toggle {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      background: white;
      cursor: pointer;
      font-size: 14px;
      color: var(--text);
      transition: 0.2s ease;
    }}
    .toggle.active {{
      background: linear-gradient(135deg, #1d2a36, #36506c);
      color: white;
      border-color: transparent;
    }}
    .search-box input {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 14px;
      background: rgba(255,255,255,0.9);
    }}
    .stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .stat {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
    }}
    .stat .k {{
      font-size: 12px;
      color: var(--muted);
    }}
    .stat .v {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
    }}
    .legend {{
      display: grid;
      gap: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    .canvas-wrap {{
      display: grid;
      grid-template-rows: auto 1fr;
      padding: 18px;
      gap: 12px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 8px 6px 0;
    }}
    .toolbar-text {{
      color: var(--muted);
      font-size: 14px;
    }}
    .toolbar button {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.9);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
    }}
    #network {{
      min-height: 78vh;
      border-radius: 18px;
      border: 1px solid var(--border);
      background:
        linear-gradient(0deg, rgba(255,255,255,0.58), rgba(255,255,255,0.58)),
        radial-gradient(circle at center, rgba(47,111,237,0.07), transparent 40%);
    }}
    .footer-note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.6;
    }}
    @media (max-width: 980px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      #network {{
        min-height: 68vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel sidebar">
      <div>
        <h1 class="title">GraphRAG 展示图</h1>
        <p class="subtitle">左侧用于切换知识图谱与向量相似图。支持搜索节点、缩放、拖拽和点击查看详细说明。</p>
      </div>

      <div class="toggle-row">
        <button class="toggle active" id="btn-kg">知识图谱</button>
        <button class="toggle" id="btn-vec">向量相似图</button>
      </div>

      <div class="search-box">
        <input id="searchInput" type="text" placeholder="搜索实体名称，例如：盘古AI大模型2.0" />
      </div>

      <div class="stats">
        <div class="stat">
          <div class="k">实体数</div>
          <div class="v" id="statNodes">{payload["knowledge_graph"]["node_count"]}</div>
        </div>
        <div class="stat">
          <div class="k">关系数</div>
          <div class="v" id="statEdges">{payload["knowledge_graph"]["edge_count"]}</div>
        </div>
      </div>

      <div>
        <div style="font-weight: 700; margin-bottom: 10px;">图例</div>
        <div class="legend">
          <div class="legend-item"><span class="dot" style="background:{TYPE_COLORS["PERSON"]};"></span>PERSON</div>
          <div class="legend-item"><span class="dot" style="background:{TYPE_COLORS["ORGANIZATION"]};"></span>ORGANIZATION</div>
          <div class="legend-item"><span class="dot" style="background:{TYPE_COLORS["GEO"]};"></span>GEO</div>
          <div class="legend-item"><span class="dot" style="background:{TYPE_COLORS["EVENT"]};"></span>EVENT</div>
          <div class="legend-item"><span class="dot" style="background:{TYPE_COLORS["UNKNOWN"]};"></span>UNKNOWN</div>
        </div>
      </div>

      <div class="footer-note">
        向量相似图基于实体描述 embedding 的 cosine similarity 构造，默认只展示相似度较高的近邻边，便于演示语义聚类关系。
      </div>
    </aside>

    <main class="panel canvas-wrap">
      <div class="toolbar">
        <div class="toolbar-text" id="graphDescription">当前视图：知识图谱关系</div>
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          <button id="fitBtn">自适应视图</button>
          <button id="stabilizeBtn">重新布局</button>
        </div>
      </div>
      <div id="network"></div>
    </main>
  </div>

  <script>
    const payload = {data_json};
    const container = document.getElementById("network");
    const searchInput = document.getElementById("searchInput");
    const statNodes = document.getElementById("statNodes");
    const statEdges = document.getElementById("statEdges");
    const graphDescription = document.getElementById("graphDescription");
    const btnKg = document.getElementById("btn-kg");
    const btnVec = document.getElementById("btn-vec");
    const fitBtn = document.getElementById("fitBtn");
    const stabilizeBtn = document.getElementById("stabilizeBtn");

    let currentMode = "knowledge_graph";
    let network = null;
    let nodes = null;
    let edges = null;

    const options = {{
      autoResize: true,
      interaction: {{
        hover: true,
        tooltipDelay: 120,
        multiselect: false,
        navigationButtons: true
      }},
      physics: {{
        stabilization: false,
        barnesHut: {{
          gravitationalConstant: -2800,
          springLength: 120,
          springConstant: 0.045
        }}
      }},
      nodes: {{
        shape: "dot",
        scaling: {{
          min: 10,
          max: 36
        }},
        font: {{
          face: "Segoe UI, Microsoft YaHei, sans-serif",
          color: "#1d2a36",
          size: 15
        }},
        borderWidth: 1.5
      }},
      edges: {{
        smooth: {{
          type: "dynamic"
        }},
        color: {{
          inherit: false
        }}
      }}
    }};

    function buildGraph(mode) {{
      currentMode = mode;
      const graph = payload[mode];
      nodes = new vis.DataSet(graph.nodes);
      edges = new vis.DataSet(graph.edges);

      if (network) {{
        network.destroy();
      }}

      network = new vis.Network(container, {{ nodes, edges }}, options);
      network.once("stabilizationIterationsDone", () => network.fit({{ animation: true }}));

      statNodes.textContent = graph.node_count;
      statEdges.textContent = graph.edge_count;
      graphDescription.textContent = mode === "knowledge_graph"
        ? "当前视图：知识图谱关系"
        : "当前视图：实体向量相似图";

      btnKg.classList.toggle("active", mode === "knowledge_graph");
      btnVec.classList.toggle("active", mode === "vector_graph");
    }}

    function focusNodeByLabel(label) {{
      if (!label || !nodes) return;
      const trimmed = label.trim().toLowerCase();
      if (!trimmed) return;
      const match = nodes.get().find(node => (node.label || "").toLowerCase().includes(trimmed));
      if (!match || !network) return;
      nodes.update({{ id: match.id, borderWidth: 4, borderWidthSelected: 4 }});
      network.selectNodes([match.id]);
      network.focus(match.id, {{
        scale: 1.2,
        animation: {{
          duration: 500,
          easingFunction: "easeInOutQuad"
        }}
      }});
    }}

    btnKg.addEventListener("click", () => buildGraph("knowledge_graph"));
    btnVec.addEventListener("click", () => buildGraph("vector_graph"));
    fitBtn.addEventListener("click", () => network && network.fit({{ animation: true }}));
    stabilizeBtn.addEventListener("click", () => network && network.stabilize(150));
    searchInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        focusNodeByLabel(searchInput.value);
      }}
    }});

    buildGraph("knowledge_graph");
  </script>
</body>
</html>
"""


def main() -> None:
    entities, relationships, entity_embeddings = load_tables()
    kg_nodes, kg_edges = build_knowledge_graph(entities, relationships)
    vec_nodes, vec_edges = build_vector_similarity_graph(entities, entity_embeddings)

    payload = {
        "knowledge_graph": {
            "nodes": kg_nodes,
            "edges": kg_edges,
            "node_count": len(kg_nodes),
            "edge_count": len(kg_edges),
        },
        "vector_graph": {
            "nodes": vec_nodes,
            "edges": vec_edges,
            "node_count": len(vec_nodes),
            "edge_count": len(vec_edges),
        },
    }

    HTML_PATH.write_text(render_html(payload), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
