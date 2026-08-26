"use strict";

// Loaded after index.html's built-in script. Replace the old fixed 20/70/100%
// animation with backend-reported machine progress while preserving log/video UI.
(function () {
  if (typeof window.renderResult !== "function") return;

  window.renderResult = function renderResult(box, taskId, status, data) {
    const prevLog = box.querySelector("#log-" + taskId);
    const logWasShown = prevLog && prevLog.classList.contains("show");
    const logText = logWasShown ? prevLog.textContent : "";
    const label = { pending: "排队中", running: "生成中", done: "已完成", failed: "失败" }[status] || status;
    const p = data && data.progress ? data.progress : {};
    let percent = Number.isFinite(Number(p.percent)) ? Number(p.percent) : (status === "done" ? 100 : 0);
    percent = Math.max(0, Math.min(100, Math.round(percent)));
    const stage = p.stage_label || p.stage || (status === "pending" ? "排队中" : status === "running" ? "启动任务" : label);
    const detail = p.detail || "";
    const segment = p.total_segments ? ` · 第 ${p.current_segment || 0}/${p.total_segments} 段` : "";

    let html = `<div class="task-card"><div class="task-head"><div class="task-id">任务 ID：<br>${taskId}</div><div class="badge ${status}">${label}</div></div>`;
    html += `<div style="display:flex;justify-content:space-between;gap:8px;margin-top:12px;font-size:12px"><strong>${stage}${segment}</strong><span style="color:var(--brand);font-weight:700">${percent}%</span></div>`;
    html += `<div class="progress ${status === "running" ? "running" : ""}"><i style="width:${percent}%"></i></div>`;
    if (detail) html += `<div class="meta" style="margin-bottom:6px">${detail}</div>`;
    if (data) {
      if (data.started_at) html += `<div class="meta">开始：${new Date(data.started_at*1000).toLocaleTimeString()}</div>`;
      if (data.finished_at) html += `<div class="meta">结束：${new Date(data.finished_at*1000).toLocaleTimeString()}</div>`;
    }
    html += `<span class="loglink" data-id="${taskId}">查看运行日志 ▾</span><div class="log" id="log-${taskId}"></div>`;
    if (status === "done" && data && data.outputs && data.outputs.length) {
      const out = data.outputs[0];
      const url = API_BASE.replace(/\/$/, "") + out.download_url;
      html += `<video class="out" controls src="${url}"></video><br><a class="dl" href="${url}" download="${out.filename}">⬇ 下载视频（${out.filename}）</a>`;
    }
    if (status === "failed" && data && data.error) {
      html += `<div class="meta" style="margin-top:8px;color:var(--err)">错误：${data.error}</div>`;
    }
    html += `</div>`;
    box.innerHTML = html;

    if (logWasShown) {
      const logEl = box.querySelector("#log-" + taskId);
      logEl.textContent = logText;
      logEl.classList.add("show");
      if (typeof scrollLogToBottom === "function") scrollLogToBottom(logEl);
    }
    const link = box.querySelector(".loglink");
    if (link) link.addEventListener("click", () => toggleLog(taskId));
  };
})();
