/* ============================================================
   DiabetesAI — Frontend Script
   ============================================================ */

const FEATURE_NAMES = [
  "Pregnancies", "Glucose", "BloodPressure",
  "SkinThickness", "Insulin", "BMI",
  "DiabetesPedigreeFunction", "Age"
];

/* ── DOM HANDLES ─────────────────────────────────────────────── */
const form        = document.getElementById("prediction-form");
const btn         = document.getElementById("predict-btn");
const resultArea  = document.getElementById("result-area");
const resultCard  = document.getElementById("result-card");
const resultIcon  = document.getElementById("result-icon");
const resultLabel = document.getElementById("result-label");
const resultProb  = document.getElementById("result-prob");
const tableBody   = document.getElementById("model-table-body");

/* ── FORM SUBMIT ─────────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!validateForm()) return;

  // Build payload
  const payload = {};
  FEATURE_NAMES.forEach(name => {
    payload[name] = parseFloat(document.getElementById(name).value);
  });

  // Loading state
  btn.classList.add("loading");
  btn.disabled = true;
  resultArea.classList.add("hidden");

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();

    if (data.error) throw new Error(data.error);
    showResult(data.prediction, data.probability);

  } catch (err) {
    showError(err.message);
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
});

/* ── VALIDATION ──────────────────────────────────────────────── */
function validateForm() {
  let valid = true;
  FEATURE_NAMES.forEach(name => {
    const input = document.getElementById(name);
    input.classList.remove("invalid");
    if (input.value === "" || isNaN(parseFloat(input.value))) {
      input.classList.add("invalid");
      valid = false;
    }
  });
  return valid;
}

/* ── RESULT DISPLAY ──────────────────────────────────────────── */
function showResult(prediction, probability) {
  const isDiabetic = prediction === "Diabetic";

  resultCard.className  = "result-card " + (isDiabetic ? "diabetic" : "not-diabetic");
  resultIcon.textContent  = isDiabetic ? "⚠️" : "✅";
  resultLabel.textContent = prediction;
  resultProb.textContent  = `${probability.toFixed(1)}%`;

  resultArea.classList.remove("hidden");
  resultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg) {
  resultCard.className    = "result-card";
  resultIcon.textContent  = "❌";
  resultLabel.textContent = "Prediction Failed";
  resultProb.textContent  = msg;
  resultArea.classList.remove("hidden");
}

/* ── MODEL COMPARISON TABLE ──────────────────────────────────── */
async function loadModelResults() {
  try {
    const resp = await fetch("/model-info");
    if (!resp.ok) return;
    const rows = await resp.json();

    if (!rows || rows.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="loading-row">No model data available yet.</td></tr>`;
      return;
    }

    // Sort by Accuracy descending
    rows.sort((a, b) => b.Accuracy - a.Accuracy);

    tableBody.innerHTML = rows.map((r, i) => `
      <tr class="${i === 0 ? "best-row" : ""}">
        <td>${i + 1}</td>
        <td>${r.Model}</td>
        <td>${r.Accuracy.toFixed(2)}</td>
        <td>${r.Precision.toFixed(2)}</td>
        <td>${r.Recall.toFixed(2)}</td>
      </tr>
    `).join("");

  } catch {
    tableBody.innerHTML = `<tr><td colspan="5" class="loading-row">Could not load model data.</td></tr>`;
  }
}

// Input focus effects: remove invalid class on input
FEATURE_NAMES.forEach(name => {
  const input = document.getElementById(name);
  input.addEventListener("input", () => input.classList.remove("invalid"));
});

// Load table on page ready
document.addEventListener("DOMContentLoaded", loadModelResults);
// Also load immediately (for scripts at bottom of body)
loadModelResults();
