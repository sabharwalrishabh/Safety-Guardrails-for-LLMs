from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except Exception as e:  # pragma: no cover
    AnalyzerEngine = None  # type: ignore
    AnonymizerEngine = None  # type: ignore
    OperatorConfig = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass(frozen=True)
class PiiEntity:
    entity_type: str
    score: float
    start: int
    end: int


_ANALYZER: Optional["AnalyzerEngine"] = None
_ANONYMIZER: Optional["AnonymizerEngine"] = None


def _ensure_engines() -> tuple["AnalyzerEngine", "AnonymizerEngine"]:
    global _ANALYZER, _ANONYMIZER
    if _IMPORT_ERR is not None:
        raise RuntimeError(
            "Presidio is not available. Install presidio-analyzer presidio-anonymizer spacy, "
            "and download a spaCy model (e.g., python -m spacy download en_core_web_sm). "
            f"Original import error: {_IMPORT_ERR}"
        )
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    if _ANONYMIZER is None:
        _ANONYMIZER = AnonymizerEngine()
    return _ANALYZER, _ANONYMIZER


def analyze_pii(
    text: str,
    *,
    entities: Optional[list[str]],
    min_confidence: float,
    language: str = "en",
) -> list[PiiEntity]:
    analyzer, _ = _ensure_engines()
    results = analyzer.analyze(text=text, entities=entities or None, language=language)

    out: list[PiiEntity] = []
    for r in results:
        if r.score >= min_confidence:
            out.append(PiiEntity(entity_type=r.entity_type, score=float(r.score), start=int(r.start), end=int(r.end)))
    return out


def mask_pii(
    text: str,
    *,
    entities: Optional[list[str]],
    min_confidence: float,
    mask_token: str = "***",
    language: str = "en",
) -> tuple[str, list[PiiEntity]]:
    analyzer, anonymizer = _ensure_engines()
    results = analyzer.analyze(text=text, entities=entities or None, language=language)

    kept = [r for r in results if r.score >= min_confidence]
    pii_entities = [
        PiiEntity(entity_type=r.entity_type, score=float(r.score), start=int(r.start), end=int(r.end))
        for r in kept
    ]

    if not kept:
        return text, pii_entities

    # Replace all detected PII spans with the same token
    operators: dict[str, Any] = {"DEFAULT": OperatorConfig("replace", {"new_value": mask_token})}
    anon = anonymizer.anonymize(text=text, analyzer_results=kept, operators=operators)
    return anon.text, pii_entities
