"""Heuristic importance scoring. Replaceable with a trained scorer later."""

from __future__ import annotations

import re
from typing import Pattern


class ImportanceScorer:
    """score(text) -> float in [0, 1]. First matching rule wins."""

    _explicit: Pattern[str] = re.compile(r"\b(never|always|rule|strategy|must)\b", re.I)
    _numerical: Pattern[str] = re.compile(
        r"(\d+[\d,]*\.?\d*\s*%?|\d{1,2}/\d{1,2}/\d{2,4}|\$\s*\d+|\d+\s*(usd|eur|kg|lb|mb|gb)?)",
        re.I,
    )
    _causal: Pattern[str] = re.compile(r"\b(because|therefore|causes|leads\s+to)\b", re.I)
    _question: Pattern[str] = re.compile(r"\?\s*$|^(what|who|when|where|why|how)\b", re.I)
    _ack: Pattern[str] = re.compile(r"^(ok|okay|yes|yep|sure|thanks|thank\s+you|got\s+it)\b", re.I)
    _entity_action: Pattern[str] = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|will|has|have|did|does|do|announced|launched|bought|sold|met)\b"
    )

    @classmethod
    def score(cls, text: str) -> float:
        t = (text or "").strip()
        if not t:
            return 0.5
        if cls._explicit.search(t):
            return 0.95
        if cls._numerical.search(t):
            return 0.90
        if cls._entity_action.search(t):
            return 0.85
        if cls._causal.search(t):
            return 0.80
        if cls._question.search(t):
            return 0.40
        if cls._ack.search(t):
            return 0.05
        return 0.50
