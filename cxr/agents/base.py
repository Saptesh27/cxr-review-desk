from dataclasses import dataclass


@dataclass
class AgentResult:
    key: str
    name: str
    algorithm: str
    status: str
    probabilities: dict[str, float | None]
    summary: str
    detail: str = ""
    heatmap_png: bytes | None = None
    error: str = ""

    def as_dict(self, heatmap_url: str | None) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "algorithm": self.algorithm,
            "status": self.status,
            "summary": self.summary,
            "detail": self.detail,
            "error": self.error,
            "heatmap_url": heatmap_url,
            "probabilities": self.probabilities,
        }


def blank_votes() -> dict[str, float | None]:
    from cxr.labels import LABELS

    return {label: None for label in LABELS}


def top_summary(probabilities: dict[str, float | None], limit: int = 3) -> str:
    ranked = sorted(
        ((label, score) for label, score in probabilities.items() if score is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return "No vote on the shared findings."
    parts = [f"{label} {score:.0%}" for label, score in ranked[:limit] if score >= 0.2]
    if not parts:
        parts = [f"{ranked[0][0]} {ranked[0][1]:.0%}"]
    return "Highest: " + ", ".join(parts)


def failed(key: str, name: str, algorithm: str, message: str, status: str = "error") -> "AgentResult":
    return AgentResult(
        key=key,
        name=name,
        algorithm=algorithm,
        status=status,
        probabilities=blank_votes(),
        summary=message,
        error="" if status == "skipped" else message,
    )
