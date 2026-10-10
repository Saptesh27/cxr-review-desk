(function () {
const { request, formatWhen } = window.CXR;
const root = document.getElementById("training-root");

const readers = [
  ["Local texture", "DenseNet-121", "Frozen"],
  ["Global pattern", "BiomedCLIP ViT", "Frozen"],
  ["Localization", "YOLOv8-World", "Frozen"],
  ["Clinical notes", "CheXagent", "Not in this run"],
];

async function refreshTraining() {
  const payload = await request("/api/training");
  root.replaceChildren();
  if (!payload.run) {
    root.append(emptyState());
    return;
  }
  root.append(summary(payload.run), metrics(payload.run), readersCard(), findingsCard(payload.labels || []));
}

function emptyState() {
  const card = document.createElement("section");
  card.className = "panel train-empty";
  const kicker = document.createElement("p");
  kicker.className = "eyebrow";
  kicker.textContent = "No training run yet";
  const title = document.createElement("h2");
  title.textContent = "The combiner has not been trained.";
  const copy = document.createElement("p");
  copy.className = "lede";
  copy.textContent = "New studies still run. Until a training record exists, the three image readers are combined by confidence weighting. The four readers themselves are never trained on this desk.";
  const steps = document.createElement("ol");
  steps.className = "train-steps";
  ["Open a terminal in the project folder.", "Activate the virtual environment.", "Run python -m cxr.train_fusion.", "Come back to this page. The held-out score will be stored here."].forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    steps.append(item);
  });
  card.append(kicker, title, copy, steps);
  return card;
}

function summary(run) {
  const grid = document.createElement("div");
  grid.className = "train-hero";
  const score = document.createElement("section");
  score.className = "panel score-card";
  const kicker = document.createElement("p");
  kicker.className = "eyebrow";
  kicker.textContent = "Held-out macro-F1";
  const number = document.createElement("p");
  number.className = "stat";
  number.textContent = Number(run.macro_f1).toFixed(3);
  const caption = document.createElement("p");
  caption.textContent = `Average F1 across 12 findings on ${run.test_count} images the combiner did not see.`;
  score.append(kicker, number, caption);

  const story = document.createElement("section");
  story.className = "panel";
  const storyKicker = document.createElement("p");
  storyKicker.className = "eyebrow";
  storyKicker.textContent = "What was trained";
  const title = document.createElement("h2");
  title.textContent = "Only the fusion model.";
  const copy = document.createElement("p");
  copy.className = "lede";
  copy.textContent = "Each film was turned into 36 numbers: three frozen image readers, twelve findings. One logistic regression per finding learned to combine those numbers. The notes reader was left out of this run.";
  const when = document.createElement("p");
  when.className = "meta";
  when.textContent = `${run.dataset} · recorded ${formatWhen(run.created_at)} · checkpoints/fusion.joblib`;
  story.append(storyKicker, title, copy, when);
  grid.append(score, story);
  return grid;
}

function metrics(run) {
  const row = document.createElement("div");
  row.className = "metric-row";
  [
    [String(run.train_count), "Training films"],
    [String(run.test_count), "Test films"],
    ["12", "Findings scored"],
    ["3", "Readers left frozen"],
  ].forEach(([value, label]) => {
    const card = document.createElement("section");
    card.className = "panel metric";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const name = document.createElement("span");
    name.textContent = label;
    card.append(strong, name);
    row.append(card);
  });
  return row;
}

function readersCard() {
  const card = document.createElement("section");
  card.className = "panel";
  const title = document.createElement("h2");
  title.textContent = "Readers in this run";
  const list = document.createElement("div");
  list.className = "reader-row";
  readers.forEach(([name, algorithm, state]) => {
    const item = document.createElement("div");
    item.className = "reader-chip";
    const heading = document.createElement("strong");
    heading.textContent = name;
    const algo = document.createElement("span");
    algo.textContent = algorithm;
    const tag = document.createElement("em");
    tag.textContent = state;
    item.append(heading, algo, tag);
    list.append(item);
  });
  card.append(title, list);
  return card;
}

function findingsCard(labels) {
  const card = document.createElement("section");
  card.className = "panel";
  const title = document.createElement("h2");
  title.textContent = "Score by finding";
  const copy = document.createElement("p");
  copy.className = "quiet";
  copy.textContent = "F1 is on the test films only. In train and In test count how many films carried that NIH label. A finding with no positive training film was not fit.";
  card.append(title, copy);
  if (!labels.length) {
    const missing = document.createElement("p");
    missing.className = "quiet";
    missing.textContent = "The per-finding breakdown is not on disk. The macro-F1 above is still the stored test score.";
    card.append(missing);
    return card;
  }
  const table = document.createElement("div");
  table.className = "score-table";
  table.append(scoreHead());
  labels.forEach((row) => table.append(scoreLine(row)));
  card.append(table);
  return card;
}

function scoreHead() {
  const row = document.createElement("div");
  row.className = "score-line score-head";
  ["Finding", "F1", "In train", "In test"].forEach((label) => {
    const cell = document.createElement("span");
    cell.textContent = label;
    row.append(cell);
  });
  return row;
}

function scoreLine(row) {
  const line = document.createElement("div");
  line.className = "score-line";
  const name = document.createElement("div");
  name.className = "score-name";
  const label = document.createElement("span");
  label.textContent = row.label;
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.round(Number(row.f1) * 100)}%`;
  bar.append(fill);
  name.append(label, bar);
  const score = document.createElement("span");
  score.className = "num";
  score.textContent = row.fitted ? Number(row.f1).toFixed(2) : "Not fit";
  const train = document.createElement("span");
  train.className = "num";
  train.textContent = String(row.train_positives);
  const test = document.createElement("span");
  test.className = "num";
  test.textContent = String(row.test_positives);
  line.append(name, score, train, test);
  return line;
}

refreshTraining();
})();
