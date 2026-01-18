from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.guardrails.filters import normalize_spaces, compact_alnum


@dataclass(frozen=True)
class TopicMatch:
    topic_name: str
    matched_keyword: str


def find_denied_topic_matches(text: str, topics: Iterable[object]) -> list[TopicMatch]:
    """
    Deterministic matching using keywords:
    - normal match: keyword is a substring of normalized text
    - obfuscation-resistant match: compact_alnum(keyword) in compact_alnum(text)
    """
    text_norm = normalize_spaces(text)
    text_comp = compact_alnum(text)

    matches: list[TopicMatch] = []
    for t in topics:
        # support pydantic objects or dict-like
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) or "unknown_topic"
        keywords = getattr(t, "keywords", None) or (t.get("keywords") if isinstance(t, dict) else None) or []

        for kw in keywords:
            kw = (kw or "").strip()
            if not kw:
                continue
            kw_norm = normalize_spaces(kw)
            kw_comp = compact_alnum(kw)

            if kw_norm and kw_norm in text_norm:
                matches.append(TopicMatch(topic_name=name, matched_keyword=kw))
                continue
            if kw_comp and kw_comp in text_comp:
                matches.append(TopicMatch(topic_name=name, matched_keyword=kw))
                continue

    # de-dupe
    uniq = {(m.topic_name, m.matched_keyword) for m in matches}
    return [TopicMatch(topic_name=a, matched_keyword=b) for (a, b) in sorted(uniq)]
