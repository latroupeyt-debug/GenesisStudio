"""Déduplication exacte et similitude n-grammes sans bibliothèque lourde."""
from __future__ import annotations

from collections import defaultdict
from .quality import normalize


def _shingles(text: str, n: int = 4) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def similarity(a: str, b: str) -> float:
    """Jaccard sur ensembles de 4-grammes de mots; approximation anti-doublons."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class Deduplicator:
    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold
        self._exact: set[str] = set()
        self._shingles: list[set[str]] = []
        self._index: dict[str, set[int]] = defaultdict(set)

    def add(self, text: str) -> bool:
        normalized = normalize(text)
        if not normalized or normalized in self._exact:
            return False
        current = _shingles(normalized)
        if current:
            overlaps: dict[int, int] = {}
            for shingle in current:
                for candidate in self._index.get(shingle, ()):
                    overlaps[candidate] = overlaps.get(candidate, 0) + 1
            for candidate, common in overlaps.items():
                previous = self._shingles[candidate]
                if min(len(current), len(previous)) / max(len(current), len(previous)) < self.threshold:
                    continue
                if common / max(1, len(current | previous)) >= self.threshold:
                    return False
        self._exact.add(normalized)
        index = len(self._shingles)
        self._shingles.append(current)
        for shingle in current:
            self._index[shingle].add(index)
        return True

    def __len__(self) -> int:
        return len(self._exact)
