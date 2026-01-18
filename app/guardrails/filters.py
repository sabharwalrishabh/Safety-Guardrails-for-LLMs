import re
from dataclasses import dataclass
from typing import Iterable, Optional


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_spaces(text: str) -> str:
    return " ".join(text.lower().split())


def compact_alnum(text: str) -> str:
    return "".join(_WORD_RE.findall(text.lower()))


@dataclass(frozen=True)
class Match:
    kind: str
    matched: str


def find_blocked_phrases(text: str, blocked_phrases: Iterable[str]) -> list[Match]:
    text_norm = normalize_spaces(text)
    text_comp = compact_alnum(text)

    matches: list[Match] = []
    for phrase in blocked_phrases:
        p = (phrase or "").strip()
        if not p:
            continue
        p_norm = normalize_spaces(p)
        p_comp = compact_alnum(p)

        if p_norm and p_norm in text_norm:
            matches.append(Match(kind="blocklist", matched=p))
            continue

        if p_comp and p_comp in text_comp:
            matches.append(Match(kind="blocklist", matched=p))
            continue

    uniq = {(m.kind, m.matched) for m in matches}
    return [Match(kind=k, matched=v) for (k, v) in sorted(uniq)]


def find_profanity(text: str, words: Optional[Iterable[str]] = None) -> list[Match]:
    wordset = {normalize_spaces(w) for w in (words or []) if (w or "").strip()}
    if not wordset:
        return []

    text_norm = normalize_spaces(text)
    text_comp = compact_alnum(text)

    matches: list[Match] = []

    tokens = set(text_norm.split())
    for w in wordset:
        if w in tokens:
            matches.append(Match(kind="profanity", matched=w))

    for w in wordset:
        w_comp = compact_alnum(w)
        if w_comp and w_comp in text_comp:
            matches.append(Match(kind="profanity", matched=w))

    uniq = {(m.kind, m.matched) for m in matches}
    return [Match(kind=k, matched=v) for (k, v) in sorted(uniq)]
