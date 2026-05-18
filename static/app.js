// static/app.js: handles user interaction on the task page — timing, AI tracking, and decision submission.

let startTime = null;
let aiSeen = false;
let explanationOpened = false;
let aiFollowed = null;

// Sets up the timer and attaches event listeners when the page has finished loading
document.addEventListener("DOMContentLoaded", function () {
  startTime = performance.now();

  // If the AI panel is present on the page, the participant has seen the AI recommendation
  const aiPanel = document.getElementById("ai-panel");
  if (aiPanel) aiSeen = true;

  // Tracks whether the participant opened the explanation panel
  const explanationBtn = document.getElementById("toggle-explanation");
  if (explanationBtn) {
    explanationBtn.addEventListener("click", function () {
      const explanation = document.getElementById("ai-explanation");
      if (!explanation) return;
      const hidden = explanation.style.display === "none";
      explanation.style.display = hidden ? "block" : "none";
      if (hidden) explanationOpened = true;
    });
  }

  document.getElementById("approve-btn")?.addEventListener("click", () => submitDecision("Approve"));
  document.getElementById("reject-btn")?.addEventListener("click", () => submitDecision("Reject"));
});

// Sends the participant's decision to the server along with timing and AI interaction data
function submitDecision(decision) {
  const caseId = document.getElementById("case-id")?.value;
  if (!caseId) return;

  const timeMs = Math.round(performance.now() - startTime);

  // Checks if the participant's decision matches the AI recommendation
  const aiRecEl = document.getElementById("ai-recommendation");
  if (aiRecEl) {
    const rec = aiRecEl.dataset.recommendation;
    aiFollowed = (decision === rec);
  } else {
    aiFollowed = null;
  }

  disableButtons();

  fetch("/submit_decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: caseId,
      decision,
      time_ms: timeMs,
      ai_followed: aiFollowed,
      ai_seen: aiSeen,
      explanation_opened: explanationOpened
    })
  })
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        window.location.href = data.next || "/task";
      } else {
        alert(data.error || "Error");
        enableButtons();
      }
    })
    .catch(() => {
      alert("Network error");
      enableButtons();
    });
}

// Disables the approve and reject buttons to prevent double submission
function disableButtons() {
  const a = document.getElementById("approve-btn");
  const r = document.getElementById("reject-btn");
  if (a) a.disabled = true;
  if (r) r.disabled = true;
}

// Re-enables the approve and reject buttons, used when a submission fails
function enableButtons() {
  const a = document.getElementById("approve-btn");
  const r = document.getElementById("reject-btn");
  if (a) a.disabled = false;
  if (r) r.disabled = false;
}
