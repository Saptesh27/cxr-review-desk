(function () {
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");

document.body.addEventListener("click", (event) => {
  const opener = event.target.closest("[data-enlarge]");
  if (!opener || !lightbox) return;
  lightboxImg.src = opener.dataset.enlarge;
  lightbox.showModal();
});

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(typeof detail === "string" ? detail : "The desk could not complete that request.");
  }
  return payload;
}

function formatWhen(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function pill(text) {
  const node = document.createElement("span");
  node.className = "pill";
  node.textContent = text;
  return node;
}

async function refreshStatus() {
  const status = await request("/api/status");
  const pills = document.getElementById("status-pills");
  if (!pills) return;
  pills.replaceChildren();
  pills.append(
    pill(status.fusion_trained ? "Fusion model trained" : "Confidence weighting"),
    pill(status.cuda ? `GPU ${status.gpu_name}` : "No NVIDIA GPU"),
  );
}

async function refreshCount() {
  const payload = await request("/api/studies");
  const node = document.getElementById("history-count");
  if (node) node.textContent = String(payload.studies.length);
  return payload.studies;
}

function studyNode(study) {
  const root = document.createElement("article");
  const banner = document.createElement("div");
  banner.className = `banner ${study.outcome}`;
  const heading = document.createElement("h2");
  heading.textContent = study.outcome === "refer" ? "Refer to radiologist" : "Final diagnosis";
  const summary = document.createElement("p");
  summary.textContent = study.summary;
  const meta = document.createElement("p");
  meta.className = "meta";
  const mode = study.fusion_mode === "trained" ? "Trained fusion" : "Confidence weighting";
  meta.textContent = `${study.code} · ${formatWhen(study.created_at)} · ${mode}`;
  banner.append(heading, summary, meta);

  const grid = document.createElement("div");
  grid.className = "report-grid";
  const thumb = document.createElement("button");
  thumb.type = "button";
  thumb.className = "thumb";
  thumb.dataset.enlarge = study.image_url;
  const film = document.createElement("img");
  film.src = study.image_url;
  film.alt = `${study.code} chest radiograph`;
  thumb.append(film);
  const findings = document.createElement("div");
  findings.className = "findings";
  const shown = study.outcome === "refer"
    ? Object.entries(study.scores).filter(([, score]) => score >= 0.35).sort((a, b) => b[1] - a[1])
    : study.findings.map((item) => [item.label, item.score]);
  if (!shown.length) {
    const none = document.createElement("p");
    none.textContent = "No finding is high enough to list.";
    findings.append(none);
  }
  shown.forEach(([label, score]) => findings.append(findingRow(label, score)));
  grid.append(thumb, findings);

  const agents = document.createElement("div");
  agents.className = "agents";
  study.agents.forEach((agent) => agents.append(agentCard(agent)));
  if (study.notes) {
    const notes = document.createElement("p");
    notes.className = "meta";
    notes.textContent = `Notes: ${study.notes}`;
    root.append(banner, grid, notes, agents);
  } else {
    root.append(banner, grid, agents);
  }
  return root;
}

function findingRow(label, score) {
  const row = document.createElement("div");
  row.className = "finding";
  const name = document.createElement("span");
  name.textContent = label;
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.round(score * 100)}%`;
  bar.append(fill);
  const num = document.createElement("span");
  num.className = "num";
  num.textContent = `${Math.round(score * 100)}%`;
  row.append(name, bar, num);
  return row;
}

function agentCard(agent) {
  const card = document.createElement("section");
  card.className = "agent";
  const header = document.createElement("header");
  const titlesWrap = document.createElement("div");
  const name = document.createElement("h3");
  name.textContent = agent.name;
  const algo = document.createElement("p");
  algo.className = "algo";
  algo.textContent = agent.algorithm;
  titlesWrap.append(name, algo);
  const tag = document.createElement("span");
  tag.className = `tag ${agent.status}`;
  tag.textContent = agent.status === "ready" ? "Ready" : agent.status === "skipped" ? "Not run" : "Failed";
  header.append(titlesWrap, tag);
  card.append(header);
  if (agent.heatmap_url) {
    const map = document.createElement("button");
    map.type = "button";
    map.className = "agent-map";
    map.dataset.enlarge = agent.heatmap_url;
    const image = document.createElement("img");
    image.src = agent.heatmap_url;
    image.alt = `${agent.name} map`;
    map.append(image);
    card.append(map);
  }
  const summary = document.createElement("p");
  summary.textContent = agent.summary;
  card.append(summary);
  if (agent.detail || agent.error) {
    const detail = document.createElement("details");
    detail.className = "detail";
    const summaryNode = document.createElement("summary");
    summaryNode.textContent = agent.detail ? "Rationale and votes" : "What happened";
    const body = document.createElement("pre");
    body.textContent = agent.detail || agent.error;
    detail.append(summaryNode, body);
    card.append(detail);
  }
  const votes = document.createElement("details");
  votes.className = "detail";
  const voteSummary = document.createElement("summary");
  voteSummary.textContent = "All 12 scores";
  const list = document.createElement("div");
  list.className = "votes";
  Object.entries(agent.probabilities).forEach(([label, score]) => {
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const scoreNode = document.createElement("span");
    scoreNode.className = "num";
    scoreNode.textContent = score === null || score === undefined ? "—" : `${Math.round(score * 100)}%`;
    list.append(labelNode, scoreNode);
  });
  votes.append(voteSummary, list);
  card.append(votes);
  return card;
}

window.CXR = { request, formatWhen, wait, studyNode, refreshCount, refreshStatus };
refreshStatus();
refreshCount();
})();
