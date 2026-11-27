"""Uncalibrated decision rules for manipulated media and caption mismatch."""
from dataclasses import dataclass

STRATEGIES = ("disagreement_max", "confidence_weighted_fusion")


def classify(score, band):
    if score is None or band[0] <= score <= band[1]:
        return "UNCERTAIN"
    return "FAKE" if score > band[1] else "REAL"


@dataclass(frozen=True)
class Signal:
    name: str
    score: float | None
    quality: float
    band: tuple


def fuse(signals, strategy, fusion_band=(.4, .6)):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown fusion strategy {strategy}")
    usable = [s for s in signals if s.score is not None and s.quality > 0]
    missing = [s.name for s in signals if s not in usable]
    states = {s.name: classify(s.score, s.band) for s in usable}
    reason = ", ".join(f"{name} {state}" for name, state in states.items())
    if missing:
        reason += ("; " if reason else "") + "Unavailable: " + ", ".join(missing)
    score = None
    if strategy == "disagreement_max":
        rank = max((s.score for s in usable), default=None)
        fake_streams = [s for s in usable if states[s.name] == "FAKE"]
        uncertain_streams = [s for s in usable if states[s.name] == "UNCERTAIN"]
        real_streams = [s for s in usable if states[s.name] == "REAL"]
        # Downgrade to UNCERTAIN only when a borderline (<0.80) single-stream FAKE occurs
        # alongside an UNCERTAIN peer stream and no REAL contrast; strong (>=0.80) or corroborated
        # manipulation flags still report FAKE.
        strong_fake = any(s.score >= 0.80 for s in fake_streams)
        if fake_streams and uncertain_streams and not real_streams and not strong_fake:
            verdict = "UNCERTAIN"
        elif fake_streams:
            verdict = "FAKE"
        elif usable and not missing and all(v == "REAL" for v in states.values()):
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"
    else:
        weights = [s.quality * abs(2*s.score-1) for s in usable]
        if sum(weights):
            score = sum(s.score*w for s,w in zip(usable, weights)) / sum(weights)
        rank = score
        verdict = classify(score, fusion_band)
        if missing and verdict == "REAL":
            verdict = "UNCERTAIN"
    return dict(prediction=verdict, confidence=None if verdict == "UNCERTAIN" else
                round(rank if verdict == "FAKE" else 1 - rank, 4), disagreement=reason,
                fusion=dict(method=strategy, score=score, ranking_score=rank,
                            band=fusion_band if strategy == STRATEGIES[1] else None,
                            is_calibrated=False, checkpoint_loaded=False))
