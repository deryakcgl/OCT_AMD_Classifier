# genai part of the project
# takes two visit results and asks ollama to write an english summary for the doctor
# important: the llm does NOT see the oct image — only labels and probabilities from my onnx model
# i use ollama locally because it runs on my machine and i can experiment without api costs

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Optional

# default ollama settings on my mac — can override with env vars if needed
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

# these thresholds i picked after reading about uncertainty in prompts
# i may tune them later when i test more cases
UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.70
TOP2_MARGIN_THRESHOLD = 0.15

# rough severity order — i use this in scenario tag logic, not as real clinical staging
SEVERITY_ORDER = {"NORMAL": 0, "DRUSEN": 1, "DME": 2, "CNV": 3}

# system prompt — i spent time on this so the model does not invent image findings
# small edits here can change the reports a lot so i try to be careful
SYSTEM_PROMPT = """You are a clinical decision-support summarizer for longitudinal retinal OCT follow-up.

You receive only structured outputs from an automated OCT classifier and limited patient visit information. You do NOT have access to raw OCT images, full medical records, physician notes, labs, or prior diagnoses.

Your role is to summarize and compare classifier outputs for physician review. You must not act as a diagnosing clinician.

Core rules:

1. Write in English only.
2. Use a professional, concise, non-alarmist clinical tone.
3. Use cautious language such as "suggests," "may indicate," "appears consistent with," "classifier output indicates," and "should be interpreted with caution."
4. Do NOT provide a definitive diagnosis.
5. Do NOT recommend treatment, medication, procedures, or urgency level.
6. Do NOT invent image-level findings such as "subretinal fluid," "retinal thickening," "hemorrhage," or "neovascular membrane" unless those exact findings are explicitly provided in the input.
7. Do NOT claim that symptoms have worsened or improved unless the complaint text explicitly states this.
8. Always distinguish classifier confidence from clinical certainty.
9. If classifier confidence is below 70%, explicitly mention that the classifier signal is uncertain.
10. If the top two class probabilities are close, explicitly mention that the result should be interpreted cautiously.
11. If only one visit is provided, summarize that visit only and state that longitudinal comparison is not possible.
12. If baseline and current outputs are identical or clinically unchanged, state that no meaningful classifier-based change is detected.
13. Always end with this exact sentence: "Final clinical judgment requires direct physician review of imaging and patient context."

Required output format:

* Timeline
* Baseline Findings
* Current Findings
* Comparison
* Limitations
"""


class GenAIError(Exception):
    # so app.py can show a clear message instead of a raw traceback
    pass


@dataclass
class VisitResult:
    # small struct so i do not pass many arguments everywhere
    # same fields as a visit row in the db
    visit_date: str
    complaints: str
    class_label: str
    confidence: float
    probabilities: dict[str, float]


def has_significant_change(
    baseline: VisitResult,
    current: VisitResult,
    *,
    confidence_delta: float = 0.15,
) -> bool:
    # when should we call ollama?
    # if the class changed, yes
    # if same class but confidence moved by 15% or more, also yes
    # not sure yet if 15% is the best value — something to test later
    if baseline.class_label != current.class_label:
        return True
    return abs(baseline.confidence - current.confidence) >= confidence_delta


def _prob_pct(probs: dict[str, float], label: str) -> str:
    # just for prompt formatting
    return f"{probs.get(label, 0.0) * 100:.0f}%"


def _top2_margin(probs: dict[str, float]) -> float:
    # gap between first and second class probability
    # small gap means the model was unsure between two classes
    ranked = sorted(probs.values(), reverse=True)
    if len(ranked) < 2:
        return ranked[0] if ranked else 0.0
    return ranked[0] - ranked[1]


def _uncertainty_level(confidence: float, top2_margin: float) -> str:
    # convert numbers to a word for the prompt so the llm knows how careful to be
    if confidence < UNCERTAINTY_CONFIDENCE_THRESHOLD:
        return "high"
    if top2_margin < TOP2_MARGIN_THRESHOLD:
        return "moderate"
    return "low"


def _days_between(start: str, end: str) -> int:
    # dates stored as YYYY-MM-DD in db
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    return abs((d1 - d0).days)


def _complaint_trend(baseline_complaint: str, current_complaint: str) -> str:
    # try to detect worsened / improved from complaint text
    # only explicit words — the prompt says not to guess symptoms
    # baseline complaint not used yet, maybe later
    current = (current_complaint or "").lower()
    if not current:
        return "not available"

    if any(t in current for t in ("worsened", "worse than", "declined", "deteriorat")):
        return "worsened"
    if any(t in current for t in ("better", "improved", "resolved", "improving")):
        return "improved"
    if any(t in current for t in ("stable", "unchanged", "routine", "annual", "check-up", "checkup", "follow-up")):
        return "stable"
    return "not available"


def _scenario_tag(
    baseline: VisitResult,
    current: VisitResult,
    *,
    confidence_delta: float,
) -> str:
    # one label so the llm knows what kind of case this is
    # order matters — class change comes first
    if baseline.class_label != current.class_label:
        return "CLASS_CHANGED"

    baseline_margin = _top2_margin(baseline.probabilities)
    current_margin = _top2_margin(current.probabilities)
    baseline_unc = _uncertainty_level(baseline.confidence, baseline_margin)
    current_unc = _uncertainty_level(current.confidence, current_margin)

    if baseline_unc == "high" or current_unc == "high":
        return "HIGH_UNCERTAINTY"
    if baseline_margin < TOP2_MARGIN_THRESHOLD or current_margin < TOP2_MARGIN_THRESHOLD:
        return "MIXED_SIGNAL"
    if abs(current.confidence - baseline.confidence) >= confidence_delta:
        return "CONFIDENCE_SHIFT_SAME_CLASS"
    return "STABLE_SAME_CLASS"


def _confidence_delta_str(baseline: VisitResult, current: VisitResult) -> str:
    delta = (current.confidence - baseline.confidence) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


def _complaint_history(baseline: VisitResult, current: VisitResult) -> str:
    # stick both complaints in one line for prompt
    parts = []
    if baseline.complaints:
        parts.append(f"Baseline ({baseline.visit_date}): {baseline.complaints}")
    if current.complaints:
        parts.append(f"Current ({current.visit_date}): {current.complaints}")
    return " | ".join(parts) if parts else "not available"


def build_comparison_prompt(
    patient_name: str,
    baseline: VisitResult,
    current: VisitResult,
    *,
    eye: str = "not available",
) -> str:
    # user prompt — visit data goes here, rules stay in system prompt
    # eye field not in db yet, so "not available" for now
    # note to self: could add left/right eye column later
    baseline_margin = _top2_margin(baseline.probabilities)
    current_margin = _top2_margin(current.probabilities)
    class_changed = baseline.class_label != current.class_label

    return f"""Generate a follow-up comparison summary using the required output format.

PATIENT CONTEXT:
Patient name: {patient_name}
Eye: {eye}
Visit purpose: OCT follow-up comparison
Relevant complaint history: {_complaint_history(baseline, current)}

CLASS DEFINITIONS:
These are automated classifier labels only, not definitive clinical diagnoses.

* CNV: choroidal neovascularization pattern suggested by the classifier
* DME: diabetic macular edema pattern suggested by the classifier
* DRUSEN: drusen-related macular changes suggested by the classifier
* NORMAL: no major pathological class flagged by the classifier

CLASSIFIER-BASED PROGRESSION HEURISTIC:
Use the following only as a coarse classifier-based comparison aid, not as clinical staging:

NORMAL < DRUSEN < DME < CNV

BASELINE VISIT:
Date: {baseline.visit_date}
Patient complaint: {baseline.complaints or "not recorded"}
Predicted class: {baseline.class_label}
Classifier confidence: {baseline.confidence * 100:.0f}%
Class probabilities:

* CNV: {_prob_pct(baseline.probabilities, "CNV")}
* DME: {_prob_pct(baseline.probabilities, "DME")}
* DRUSEN: {_prob_pct(baseline.probabilities, "DRUSEN")}
* NORMAL: {_prob_pct(baseline.probabilities, "NORMAL")}

CURRENT VISIT:
Date: {current.visit_date}
Patient complaint: {current.complaints or "not recorded"}
Predicted class: {current.class_label}
Classifier confidence: {current.confidence * 100:.0f}%
Class probabilities:

* CNV: {_prob_pct(current.probabilities, "CNV")}
* DME: {_prob_pct(current.probabilities, "DME")}
* DRUSEN: {_prob_pct(current.probabilities, "DRUSEN")}
* NORMAL: {_prob_pct(current.probabilities, "NORMAL")}

DERIVED SIGNALS:
Interval between visits: {_days_between(baseline.visit_date, current.visit_date)} days
Class changed: {"yes" if class_changed else "no"}
Class transition: {baseline.class_label} → {current.class_label}
Confidence delta: {_confidence_delta_str(baseline, current)}
Baseline uncertainty level: {_uncertainty_level(baseline.confidence, baseline_margin)}
Current uncertainty level: {_uncertainty_level(current.confidence, current_margin)}
Top-2 probability margin at baseline: {baseline_margin * 100:.0f}%
Top-2 probability margin at current visit: {current_margin * 100:.0f}%
Complaint trend: {_complaint_trend(baseline.complaints, current.complaints)}
Scenario tag: {_scenario_tag(baseline, current, confidence_delta=0.15)}

INSTRUCTIONS:
Write a concise clinical summary for physician review.
Compare the baseline and current classifier outputs.
Relate the classifier change to the complaint trend only if the complaint text supports that relationship.
Do not infer clinical findings beyond the classifier labels and probabilities.
Do not recommend treatment, urgency, or follow-up interval.
Use cautious language throughout.
End with the required physician-review sentence.
"""


def _call_ollama(
    prompt: str,
    *,
    base_url: str,
    model: str,
) -> str:
    # call local ollama api
    # i used urllib instead of requests because it is in the standard library
    # temperature 0.2 — lower creativity, more consistent wording
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode()
    url = f"{base_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        # usually means ollama is not running
        raise GenAIError(
            f"Cannot reach Ollama at {base_url}. "
            f"Start Ollama and pull a model (e.g. ollama pull {model}). "
            f"Details: {exc}"
        ) from exc

    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise GenAIError(f"Unexpected Ollama response: {data}") from exc


def generate_comparison_report(
    patient_name: str,
    baseline: VisitResult,
    current: VisitResult,
    *,
    eye: str = "not available",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    # main entry point — app.py calls this
    # returns None when change is small (unless force=True from Compare button)
    # no template fallback — only ollama or an error
    if not force and not has_significant_change(baseline, current):
        return None

    url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    ollama_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    prompt = build_comparison_prompt(patient_name, baseline, current, eye=eye)
    return _call_ollama(prompt, base_url=url, model=ollama_model)
