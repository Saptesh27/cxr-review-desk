(function () {
const { request, studyNode, wait, refreshCount, refreshStatus } = window.CXR;

const stepNames = [
  "Reading the film",
  "Local texture model",
  "Global pattern model",
  "Localization model",
  "Clinical notes model",
  "Saving the study",
];

const report = document.getElementById("report");
const emptyReport = document.getElementById("empty-report");
const form = document.getElementById("study-form");
const fileInput = document.getElementById("file");
const notesInput = document.getElementById("notes");
const runButton = document.getElementById("run");
const previewImg = document.getElementById("preview-img");
const previewEmpty = document.getElementById("preview-empty");
const drop = document.getElementById("drop");
const steps = document.getElementById("steps");

let selectedFile = null;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) setFile(file);
});

["dragover", "dragleave", "drop"].forEach((name) => {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.toggle("is-hot", name === "dragover");
    if (name === "drop" && event.dataTransfer.files[0]) setFile(event.dataTransfer.files[0]);
  });
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") form.requestSubmit();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    fileInput.focus();
    return;
  }
  const body = new FormData();
  body.append("image", selectedFile);
  body.append("notes", notesInput.value);
  runButton.disabled = true;
  paintSteps(1);
  try {
    const created = await request("/api/studies", { method: "POST", body });
    const study = await poll(created.job_id);
    renderReport(study);
    await refreshCount();
  } catch (error) {
    renderProblem(error.message);
  } finally {
    runButton.disabled = false;
  }
});

function setFile(file) {
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.alt = file.name;
  previewImg.hidden = false;
  previewEmpty.hidden = true;
}

async function poll(jobId) {
  for (;;) {
    const job = await request(`/api/jobs/${jobId}`);
    if (job.state === "running") {
      paintSteps(job.index || 1);
      await wait(800);
      continue;
    }
    paintSteps(stepNames.length);
    if (job.state === "error") throw new Error(job.message || "The review failed.");
    await refreshStatus();
    return job.study;
  }
}

function paintSteps(activeIndex) {
  steps.hidden = false;
  steps.replaceChildren();
  stepNames.forEach((name, index) => {
    const item = document.createElement("li");
    const number = index + 1;
    item.textContent = name;
    item.className = number < activeIndex ? "is-done" : number === activeIndex ? "is-now" : "";
    steps.append(item);
  });
}

function renderReport(study) {
  emptyReport.hidden = true;
  const existing = report.querySelector(".study");
  if (existing) existing.remove();
  const problem = report.querySelector(".error-line");
  if (problem) problem.remove();
  const node = studyNode(study);
  node.classList.add("study");
  report.append(node);
}

function renderProblem(message) {
  const line = document.createElement("p");
  line.className = "error-line";
  line.textContent = message;
  report.append(line);
}
})();
