"""Confidence-weighted fallback, trained logistic fusion, and the disagreement gate."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cxr.labels import (
    FUSION_AGENTS,
    LABELS,
    MLLM_MIX,
    POSITIVE_THRESHOLD,
    REFER_SCORE,
    REFER_STD,
)
from cxr.paths import FUSION_PATH


def vote_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def feature_vector(votes_by_agent: dict[str, dict[str, float | None]]) -> list[float]:
    values: list[float] = []
    for agent in FUSION_AGENTS:
        votes = votes_by_agent.get(agent, {})
        for label in LABELS:
            score = votes.get(label)
            values.append(0.0 if score is None else float(score))
    return values


def hand_scores(votes_by_agent: dict[str, dict[str, float | None]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in LABELS:
        weighted = 0.0
        weight_sum = 0.0
        for votes in votes_by_agent.values():
            score = votes.get(label)
            if score is None:
                continue
            weight = float(score)
            weighted += weight * weight
            weight_sum += weight
        scores[label] = 0.0 if weight_sum == 0 else weighted / weight_sum
    return scores


def disagreement(votes_by_agent: dict[str, dict[str, float | None]]) -> dict[str, float]:
    stds: dict[str, float] = {}
    for label in LABELS:
        values = [
            float(votes[label])
            for votes in votes_by_agent.values()
            if votes.get(label) is not None
        ]
        stds[label] = vote_std(values)
    return stds


def load_fusion() -> dict | None:
    if not FUSION_PATH.exists():
        return None
    payload = joblib.load(FUSION_PATH)
    if payload.get("labels") != LABELS or payload.get("agents") != list(FUSION_AGENTS):
        return None
    return payload


def _trained_scores(votes_by_agent: dict[str, dict[str, float | None]], payload: dict) -> dict[str, float]:
    vector = np.asarray([feature_vector(votes_by_agent)], dtype=np.float64)
    scores: dict[str, float] = {}
    for label in LABELS:
        entry = payload["models"][label]
        if entry["kind"] == "constant":
            scores[label] = float(entry["value"])
            continue
        model: LogisticRegression = entry["model"]
        probabilities = model.predict_proba(vector)[0]
        if 1 in list(model.classes_):
            scores[label] = float(probabilities[list(model.classes_).index(1)])
        else:
            scores[label] = 0.0
    return scores


def mix_notes(scores: dict[str, float], notes_votes: dict[str, float | None] | None) -> dict[str, float]:
    if not notes_votes:
        return scores
    mixed: dict[str, float] = {}
    for label in LABELS:
        notes_score = notes_votes.get(label)
        if notes_score is None:
            mixed[label] = scores[label]
        else:
            mixed[label] = (1.0 - MLLM_MIX) * scores[label] + MLLM_MIX * float(notes_score)
    return mixed


def combine(votes_by_agent: dict[str, dict[str, float | None]]) -> dict:
    payload = load_fusion()
    image_votes = {agent: votes_by_agent.get(agent, {}) for agent in FUSION_AGENTS}
    if payload is None:
        base = hand_scores(image_votes)
        mode = "confidence-weighted"
    else:
        base = _trained_scores(image_votes, payload)
        mode = "trained"
    scores = mix_notes(base, votes_by_agent.get("mllm"))
    stds = disagreement(votes_by_agent)
    contested = [
        label
        for label in LABELS
        if scores[label] >= REFER_SCORE and stds[label] >= REFER_STD
    ]
    findings = [
        {"label": label, "score": round(scores[label], 4)}
        for label in LABELS
        if scores[label] >= POSITIVE_THRESHOLD
    ]
    findings.sort(key=lambda item: item["score"], reverse=True)
    if contested:
        outcome = "refer"
        names = ", ".join(contested)
        summary = (
            f"The agents disagree on {names}. "
            "This study is flagged for a radiologist, with each agent's map attached."
        )
    elif findings:
        outcome = "final"
        names = ", ".join(item["label"] for item in findings)
        summary = f"Reported findings: {names}."
    else:
        outcome = "final"
        summary = (
            "No finding crossed the reporting threshold, and the agents did not "
            "disagree enough to refer this study."
        )
    if mode == "trained":
        summary += " The image agents were combined by the trained fusion model."
    else:
        summary += " The image agents were combined by confidence weighting. Train the fusion model to replace that rule."
    return {
        "outcome": outcome,
        "fusion_mode": mode,
        "summary": summary,
        "scores": {label: round(scores[label], 4) for label in LABELS},
        "stds": {label: round(stds[label], 4) for label in LABELS},
        "findings": findings,
        "contested": contested,
    }


def fit_models(features: np.ndarray, targets: np.ndarray) -> dict:
    models: dict[str, dict] = {}
    for index, label in enumerate(LABELS):
        column = targets[:, index]
        positives = int(column.sum())
        if positives == 0 or positives == len(column):
            models[label] = {"kind": "constant", "value": float(column.mean()) if len(column) else 0.0}
            continue
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(features, column.astype(int))
        models[label] = {"kind": "logistic", "model": model}
    return {
        "labels": LABELS,
        "agents": list(FUSION_AGENTS),
        "models": models,
    }


def predict_matrix(payload: dict, features: np.ndarray) -> np.ndarray:
    output = np.zeros((len(features), len(LABELS)), dtype=np.int64)
    for index, label in enumerate(LABELS):
        entry = payload["models"][label]
        if entry["kind"] == "constant":
            output[:, index] = int(entry["value"] >= POSITIVE_THRESHOLD)
            continue
        model: LogisticRegression = entry["model"]
        probabilities = model.predict_proba(features)
        if 1 in list(model.classes_):
            positive = probabilities[:, list(model.classes_).index(1)]
        else:
            positive = np.zeros(len(features))
        output[:, index] = (positive >= POSITIVE_THRESHOLD).astype(np.int64)
    return output


def macro_f1(payload: dict, features: np.ndarray, targets: np.ndarray) -> float:
    if len(features) == 0:
        return 0.0
    predicted = predict_matrix(payload, features)
    return float(f1_score(targets.astype(int), predicted, average="macro", zero_division=0))


def per_label_f1(payload: dict, features: np.ndarray, targets: np.ndarray) -> list[dict]:
    """F1 for each finding on one split. Used by the training record page."""
    if len(features) == 0:
        return []
    predicted = predict_matrix(payload, features)
    rows = []
    for index, label in enumerate(LABELS):
        truth = targets[:, index].astype(int)
        guess = predicted[:, index]
        score = float(f1_score(truth, guess, zero_division=0))
        rows.append(
            {
                "label": label,
                "f1": round(score, 4),
                "test_positives": int(truth.sum()),
                "fitted": payload["models"][label]["kind"] == "logistic",
            }
        )
    return rows


def _self_test() -> None:
    agree = {
        "cnn": {"Effusion": 0.80},
        "vit": {"Effusion": 0.82},
        "detector": {"Effusion": 0.78},
    }
    for label in LABELS:
        for agent in agree.values():
            agent.setdefault(label, 0.05)
    result = combine(agree)
    assert result["outcome"] == "final", result
    assert any(item["label"] == "Effusion" for item in result["findings"])

    split = {
        "cnn": {label: 0.1 for label in LABELS},
        "vit": {label: 0.1 for label in LABELS},
        "detector": {label: 0.0 for label in LABELS},
    }
    split["cnn"]["Effusion"] = 0.90
    split["vit"]["Effusion"] = 0.20
    split["detector"]["Effusion"] = 0.0
    result = combine(split)
    assert result["stds"]["Effusion"] >= REFER_STD
    assert result["outcome"] == "refer", result
    print("fusion self-test passed")


if __name__ == "__main__":
    _self_test()
