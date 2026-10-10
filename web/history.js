(function () {
const { request, formatWhen, studyNode } = window.CXR;

const historyList = document.getElementById("history-list");
const historyDetail = document.getElementById("history-detail");
const historyFilter = document.getElementById("history-filter");
const historyCount = document.getElementById("history-count");

let studies = [];

historyFilter.addEventListener("input", () => paintHistory());

async function loadHistory() {
  const payload = await request("/api/studies");
  studies = payload.studies;
  historyCount.textContent = String(studies.length);
  paintHistory();
}

function paintHistory() {
  const query = historyFilter.value.trim().toLowerCase();
  const rows = studies.filter((study) => {
    const haystack = `${study.code} ${study.notes} ${study.top_finding}`.toLowerCase();
    return haystack.includes(query);
  });
  historyList.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "quiet";
    empty.textContent = studies.length ? "No studies match that filter." : "No saved studies yet. Run a review and it will be listed here.";
    historyList.append(empty);
    return;
  }
  const table = document.createElement("table");
  table.className = "table";
  const head = document.createElement("tr");
  ["Study", "When", "Outcome", "Top finding"].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.append(cell);
  });
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  rows.forEach((study) => {
    const row = document.createElement("tr");
    row.addEventListener("click", () => openHistory(study.id));
    [study.code, formatWhen(study.created_at), study.outcome === "refer" ? "Refer" : "Final", study.top_finding || "None"].forEach((value, index) => {
      const cell = document.createElement("td");
      if (index === 2) {
        const mark = document.createElement("span");
        mark.className = `outcome ${study.outcome}`;
        mark.textContent = value;
        cell.append(mark);
      } else {
        cell.textContent = value;
      }
      row.append(cell);
    });
    tbody.append(row);
  });
  table.append(thead, tbody);
  historyList.append(table);
}

async function openHistory(id) {
  const study = await request(`/api/studies/${id}`);
  historyDetail.replaceChildren();
  const wrap = document.createElement("div");
  wrap.className = "report";
  wrap.append(studyNode(study));
  const actions = document.createElement("div");
  actions.className = "history-actions";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "danger";
  remove.textContent = "Delete this study";
  remove.addEventListener("click", async () => {
    await request(`/api/studies/${id}`, { method: "DELETE" });
    historyDetail.replaceChildren();
    await loadHistory();
  });
  actions.append(remove);
  wrap.append(actions);
  historyDetail.append(wrap);
}

loadHistory();
})();
