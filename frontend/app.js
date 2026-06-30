const state = {
  lastResponse: null,
  files: [],
  health: null,
  filePollTimer: null,
};

const $ = (id) => document.getElementById(id);

const elements = {
  systemSummary: $("systemSummary"),
  chatView: $("chatView"),
  adminView: $("adminView"),
  reloadButton: $("reloadButton"),
  rebuildButton: $("rebuildButton"),
  methodSelect: $("methodSelect"),
  questionInput: $("questionInput"),
  tenantInput: $("tenantInput"),
  conversationInput: $("conversationInput"),
  departmentInput: $("departmentInput"),
  topKInput: $("topKInput"),
  graphKInput: $("graphKInput"),
  permissionToggle: $("permissionToggle"),
  contextToggle: $("contextToggle"),
  askButton: $("askButton"),
  clearButton: $("clearButton"),
  answerOutput: $("answerOutput"),
  queryStatus: $("queryStatus"),
  refreshFilesButton: $("refreshFilesButton"),
  fileInput: $("fileInput"),
  previewUploadButton: $("previewUploadButton"),
  uploadButton: $("uploadButton"),
  overwriteToggle: $("overwriteToggle"),
  fileList: $("fileList"),
  newFilenameInput: $("newFilenameInput"),
  newFileContentInput: $("newFileContentInput"),
  createTextButton: $("createTextButton"),
  textOverwriteToggle: $("textOverwriteToggle"),
  healthStatus: $("healthStatus"),
  statsGrid: $("statsGrid"),
  sourcesTab: $("sourcesTab"),
  graphTab: $("graphTab"),
  rawTab: $("rawTab"),
  toast: $("toast"),
};

function switchView(view) {
  const isAdmin = view === "admin";
  elements.chatView.classList.toggle("hidden", isAdmin);
  elements.adminView.classList.toggle("hidden", !isAdmin);
  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  if (isAdmin) {
    refreshAll();
  }
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = label || "处理中";
    button.disabled = true;
    return;
  }
  button.textContent = button.dataset.originalText || button.textContent;
  button.disabled = false;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return body;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 3200);
}

function formatBytes(size) {
  if (!Number.isFinite(size)) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDate(timestamp) {
  if (!timestamp) return "-";
  return new Date(timestamp * 1000).toLocaleString();
}

function updateStatusPill(element, text, className = "") {
  element.textContent = text;
  element.className = `status-pill ${className}`.trim();
}

async function loadHealth() {
  try {
    const health = await requestJson("/health");
    state.health = health;
    renderHealth(health);
  } catch (error) {
    updateStatusPill(elements.healthStatus, "离线", "warn");
    elements.systemSummary.textContent = `连接失败：${error.message}`;
  }
}

function renderHealth(health) {
  const counts = health.counts || {};
  updateStatusPill(elements.healthStatus, health.loaded ? "正常" : "未加载", health.loaded ? "ok" : "warn");
  elements.systemSummary.textContent = health.loaded
    ? `已加载 ${health.file_count || 0} 个文件，${counts.text_units || 0} 个文本块，${counts.relationships || 0} 条关系`
    : "索引尚未加载";

  const stats = [
    ["文件", health.file_count],
    ["文本块", counts.text_units],
    ["实体", counts.entities],
    ["关系", counts.relationships],
    ["社区", counts.communities],
    ["报告", counts.community_reports],
  ];
  elements.statsGrid.innerHTML = stats
    .map(([label, value]) => {
      const safeValue = value ?? 0;
      return `<div class="stat"><strong>${safeValue}</strong><span>${label}</span></div>`;
    })
    .join("");
}

async function loadFiles() {
  try {
    state.files = await requestJson("/files");
    renderFiles();
    scheduleFilePolling();
  } catch (error) {
    elements.fileList.innerHTML = `<div class="evidence-item">文件列表加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function scheduleFilePolling() {
  window.clearTimeout(state.filePollTimer);
  if (!state.files.some((file) => file.status === "PROCESSING")) return;
  state.filePollTimer = window.setTimeout(loadFiles, 2500);
}

function renderFiles() {
  if (!state.files.length) {
    elements.fileList.innerHTML = `<div class="evidence-item">暂无文件。</div>`;
    return;
  }
  elements.fileList.innerHTML = state.files
    .map(
      (file) => `
        <div class="file-item">
          <div>
            <div class="file-name">${escapeHtml(file.filename)}</div>
            <div class="file-meta">${formatBytes(file.size)} · ${formatDate(file.updated_at)} · ${escapeHtml(file.status || "SUCCESS")} · 租户 ${escapeHtml(file.tenant_id || "default")}${file.conversation_id ? ` · 对话 ${escapeHtml(file.conversation_id)}` : ""}${file.error ? ` · ${escapeHtml(file.error)}` : ""}</div>
          </div>
          <button class="danger-button" data-delete-file="${escapeHtml(file.filename)}">删除</button>
        </div>
      `,
    )
    .join("");
}

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    showToast("先输入一个问题。");
    elements.questionInput.focus();
    return;
  }

  const payload = {
    question,
    method: elements.methodSelect.value,
    include_context: elements.contextToggle.checked,
    enforce_permissions: elements.permissionToggle.checked,
    tenant_id: elements.tenantInput.value.trim() || null,
    conversation_id: elements.conversationInput.value.trim() || null,
    department_id: elements.departmentInput.value.trim() || null,
    response_type: "客服短答：1段话或3-5个要点，去掉来源引用",
    hybrid_top_k: Number(elements.topKInput.value || 10),
    hybrid_graph_k: Number(elements.graphKInput.value || 20),
  };

  setBusy(elements.askButton, true, "查询中");
  updateStatusPill(elements.queryStatus, "查询中", "warn");
  elements.answerOutput.textContent = "正在检索和生成回答...";

  try {
    if (payload.method === "hybrid" && payload.conversation_id) {
      await askQuestionStream(question, payload);
    } else {
      const response = await requestJson("/query", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.lastResponse = response;
      renderAnswer(response);
    }
    updateStatusPill(elements.queryStatus, "完成", "ok");
  } catch (error) {
    elements.answerOutput.textContent = `查询失败：${error.message}`;
    updateStatusPill(elements.queryStatus, "失败", "warn");
  } finally {
    setBusy(elements.askButton, false);
  }
}

async function askQuestionStream(question, payload) {
  const streamPayload = {
    conversation_id: payload.conversation_id,
    message: question,
    tenant_id: payload.tenant_id || "default",
    method: payload.method,
    include_context: payload.include_context,
    response_type: payload.response_type,
    hybrid_top_k: payload.hybrid_top_k,
    hybrid_graph_k: payload.hybrid_graph_k,
  };
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(streamPayload),
  });
  if (!response.ok || !response.body) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  elements.answerOutput.textContent = "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let context = null;
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent);
      if (!parsed) continue;
      if (parsed.event === "status") {
        updateStatusPill(
          elements.queryStatus,
          parsed.data.stage === "retrieving" ? "检索中" : "生成中",
          "warn",
        );
      } else if (parsed.event === "delta") {
        const chunk = parsed.data.text || "";
        answer += chunk;
        elements.answerOutput.textContent = cleanCustomerAnswer(answer);
      } else if (parsed.event === "context") {
        context = parsed.data || {};
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.detail || "流式回答失败");
      }
    }
  }

  answer = cleanCustomerAnswer(answer);
  state.lastResponse = { method: payload.method, answer, context };
  renderEvidence(context || {});
  elements.rawTab.textContent = JSON.stringify(state.lastResponse, null, 2);
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  if (!eventLine || !dataLines.length) return null;
  const event = eventLine.slice("event:".length).trim();
  const dataText = dataLines.map((line) => line.slice("data:".length).trim()).join("\n");
  return { event, data: JSON.parse(dataText) };
}

function renderAnswer(response) {
  elements.answerOutput.textContent = cleanCustomerAnswer(stringifyAnswer(response.answer));
  renderEvidence(response.context || {});
  elements.rawTab.textContent = JSON.stringify(response, null, 2);
}

function stringifyAnswer(answer) {
  if (answer == null) return "";
  if (typeof answer === "string") return answer;
  return JSON.stringify(answer, null, 2);
}

function cleanCustomerAnswer(answer) {
  return String(answer || "")
    .replace(/\s*\[Data:[^\]]+\]/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderEvidence(context) {
  const sources = Array.isArray(context.sources) ? context.sources : [];
  const graph = Array.isArray(context.graph_context) ? context.graph_context : [];

  elements.sourcesTab.innerHTML = sources.length
    ? sources
        .map((source) => {
          const ranks = [
            source.vector_rank ? `V:${source.vector_rank}` : "",
            source.keyword_rank ? `K:${source.keyword_rank}` : "",
            source.graph_rank ? `G:${source.graph_rank}` : "",
          ]
            .filter(Boolean)
            .join(" · ");
          return `
            <article class="evidence-item">
              <div class="evidence-title">
                <span>${escapeHtml(source.id || source.text_unit_id || "chunk")}</span>
                <span>${escapeHtml(ranks)}</span>
              </div>
              <p class="evidence-text">${escapeHtml(source.text || "")}</p>
            </article>
          `;
        })
        .join("")
    : `<div class="evidence-item">没有返回 chunk 上下文。</div>`;

  elements.graphTab.innerHTML = graph.length
    ? graph
        .map(
          (item) => `
            <article class="evidence-item">
              <div class="evidence-title">
                <span>${escapeHtml(item.source || "-")} → ${escapeHtml(item.target || "-")}</span>
                <span>${escapeHtml(item.relationship_id || "")}</span>
              </div>
              <p class="evidence-text">${escapeHtml(item.description || item.text || "")}</p>
            </article>
          `,
        )
        .join("")
    : `<div class="evidence-item">没有返回图关系上下文。</div>`;
}

async function uploadFile() {
  const file = elements.fileInput.files[0];
  if (!file) {
    showToast("请选择要上传的文件。");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("overwrite", elements.overwriteToggle.checked ? "true" : "false");
  formData.append("tenant_id", elements.tenantInput.value.trim() || "default");
  formData.append("conversation_id", elements.conversationInput.value.trim());
  formData.append("department_id", elements.departmentInput.value.trim() || "default");
  formData.append("visibility", elements.conversationInput.value.trim() ? "private" : "department");

  setBusy(elements.uploadButton, true, "上传中");
  try {
    const response = await fetch("/files/upload", { method: "POST", body: formData });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "上传失败");
    const convertedText = body.converted ? `，已转换为 ${body.conversion_type}` : "";
    const taskText = body.status === "PROCESSING" ? "，后台处理中" : "";
    showToast(`已${body.action === "replaced" ? "覆盖" : "上传"}：${body.filename}${convertedText}${taskText}`);
    elements.fileInput.value = "";
    await refreshAll();
  } catch (error) {
    showToast(`上传失败：${error.message}`);
  } finally {
    setBusy(elements.uploadButton, false);
  }
}

async function previewUpload() {
  const file = elements.fileInput.files[0];
  if (!file) {
    showToast("请选择要预览的文件。");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);

  setBusy(elements.previewUploadButton, true, "预览中");
  try {
    const response = await fetch("/files/preview", { method: "POST", body: formData });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "预览失败");
    elements.rawTab.textContent = JSON.stringify(body, null, 2);
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.add("hidden"));
    document.querySelector('[data-tab="raw"]').classList.add("active");
    elements.rawTab.classList.remove("hidden");
    showToast(`预览完成：${body.filename}`);
  } catch (error) {
    showToast(`预览失败：${error.message}`);
  } finally {
    setBusy(elements.previewUploadButton, false);
  }
}

async function createTextFile() {
  const filename = elements.newFilenameInput.value.trim();
  if (!filename) {
    showToast("请输入文件名。");
    return;
  }
  setBusy(elements.createTextButton, true, "保存中");
  try {
    const body = await requestJson("/files/text", {
      method: "POST",
      body: JSON.stringify({
        filename,
        content: elements.newFileContentInput.value,
        overwrite: elements.textOverwriteToggle.checked,
        tenant_id: elements.tenantInput.value.trim() || "default",
        conversation_id: elements.conversationInput.value.trim() || null,
        department_id: elements.departmentInput.value.trim() || "default",
        visibility: elements.conversationInput.value.trim() ? "private" : "department",
      }),
    });
    showToast(`已${body.action === "replaced" ? "覆盖" : "创建"}：${body.filename}`);
    elements.newFilenameInput.value = "";
    elements.newFileContentInput.value = "";
    await refreshAll();
  } catch (error) {
    showToast(`保存失败：${error.message}`);
  } finally {
    setBusy(elements.createTextButton, false);
  }
}

async function deleteFile(filename) {
  const confirmed = window.confirm(`删除 ${filename} 并重建索引？`);
  if (!confirmed) return;
  try {
    await requestJson(`/files/${encodeURIComponent(filename)}`, { method: "DELETE" });
    showToast(`已删除：${filename}`);
    await refreshAll();
  } catch (error) {
    showToast(`删除失败：${error.message}`);
  }
}

async function rebuildIndex() {
  setBusy(elements.rebuildButton, true, "重建中");
  try {
    await requestJson("/rebuild", {
      method: "POST",
      body: JSON.stringify({ verbose: false, sync_mysql: true }),
    });
    showToast("索引已重建。");
    await refreshAll();
  } catch (error) {
    showToast(`重建失败：${error.message}`);
  } finally {
    setBusy(elements.rebuildButton, false);
  }
}

async function reloadIndex() {
  setBusy(elements.reloadButton, true, "...");
  try {
    await requestJson("/reload", { method: "POST" });
    showToast("索引已重新加载。");
    await refreshAll();
  } catch (error) {
    showToast(`重载失败：${error.message}`);
  } finally {
    setBusy(elements.reloadButton, false);
  }
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadFiles()]);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvents() {
  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });
  elements.askButton.addEventListener("click", askQuestion);
  elements.questionInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      askQuestion();
    }
  });
  elements.clearButton.addEventListener("click", () => {
    elements.questionInput.value = "";
    elements.answerOutput.textContent = "输入问题后，答案会显示在这里。";
    state.lastResponse = null;
    renderEvidence({});
    elements.rawTab.textContent = "";
    updateStatusPill(elements.queryStatus, "待查询");
  });
  elements.refreshFilesButton.addEventListener("click", loadFiles);
  elements.previewUploadButton.addEventListener("click", previewUpload);
  elements.uploadButton.addEventListener("click", uploadFile);
  elements.createTextButton.addEventListener("click", createTextFile);
  elements.rebuildButton.addEventListener("click", rebuildIndex);
  elements.reloadButton.addEventListener("click", reloadIndex);
  elements.fileList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-file]");
    if (button) deleteFile(button.dataset.deleteFile);
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.add("hidden"));
      tab.classList.add("active");
      $(`${tab.dataset.tab}Tab`).classList.remove("hidden");
    });
  });
}

bindEvents();
renderEvidence({});
refreshAll();
