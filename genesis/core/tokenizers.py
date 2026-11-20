"""Tokenizers écrits depuis zéro : caractères, mots, BPE (au niveau octet).

Aucune dépendance externe : tout est implémenté ici, sauvegardé en JSON dans le
projet pour rester identique d'une version du modèle à l'autre (indispensable
pour reprendre un entraînement sans repartir de zéro).

Convention d'identifiants (identique pour les 3 tokenizers) :
    id 0 : token spécial de séparation de documents  <|endoftext|>
    le reste : tokens du vocabulaire propre au tokenizer
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

SPECIAL = "<|endoftext|>"
SPECIAL_ID = 0

# Découpage "pré-tokenisation" proche de GPT-2 : mots, nombres, ponctuation,
# espaces. On garde l'apostrophe française (l'été -> l + ' + été).
_PRETOK = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    re.UNICODE,
)

_WORD_ONLY = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """Interface commune aux trois tokenizers."""

    kind = "base"
    display_name = "Base"

    def __init__(self) -> None:
        self._decode_cache: dict[int, str] = {}
        self._encode_cache: dict[str, list[int]] = {}

    # -- API ------------------------------------------------------------
    @property
    def vocab_size(self) -> int:  # pragma: no cover - surchargé
        raise NotImplementedError

    def fit(self, chunks: Iterable[str], **opts) -> None:  # pragma: no cover
        raise NotImplementedError

    def _encode_plain(self, text: str) -> list[int]:  # pragma: no cover
        raise NotImplementedError

    def _decode_id(self, i: int) -> str:  # pragma: no cover
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if SPECIAL not in text:
            return self._encode_plain(text)
        out: list[int] = []
        parts = text.split(SPECIAL)
        for i, part in enumerate(parts):
            if i:
                out.append(SPECIAL_ID)
            if part:
                out.extend(self._encode_plain(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self._decode_id(int(i)) for i in ids)

    def chunks(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        """Encode un flux de texte et renvoie les ids en octets bruts (uint16/32)."""
        import array

        for text in text_iter:
            if not text:
                continue
            ids = self.encode(text)
            if not ids:
                continue
            arr = array.array(self.array_typecode, ids)
            yield arr.tobytes()

    @property
    def array_typecode(self) -> str:
        return "H" if self.vocab_size <= 65535 else "I"

    @property
    def dtype(self) -> str:
        return "uint16" if self.vocab_size <= 65535 else "uint32"

    # -- Persistance ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.vocab_size}

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tokenizer.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path) -> "Tokenizer":
        data = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
        kind = data.get("kind")
        if kind == "char":
            return CharTokenizer.from_dict(data)
        if kind == "word":
            return WordTokenizer.from_dict(data)
        if kind == "bpe":
            return BPETokenizer.from_dict(data)
        raise ValueError(f"Type de tokenizer inconnu : {kind}")


class CharTokenizer(Tokenizer):
    kind = "char"
    display_name = "Caractères"

    def __init__(self, chars: list[str] | None = None) -> None:
        super().__init__()
        self.chars = chars or []
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = spécial

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1

    def fit(self, chunks: Iterable[str], max_vocab: int | None = None, **_: object) -> None:
        counts: Counter[str] = Counter()
        for text in chunks:
            counts.update(text)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_vocab:
            ordered = ordered[: max(2, max_vocab)]
        # on garde les caractères dans l'ordre alphabétique pour un vocabulaire stable
        self.chars = sorted(c for c, _ in ordered)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}

    def _encode_plain(self, text: str) -> list[int]:
        stoi = self.stoi
        return [stoi.get(ch, 0) for ch in text]

    def _decode_id(self, i: int) -> str:
        if i == SPECIAL_ID:
            return SPECIAL
        if 1 <= i <= len(self.chars):
            return self.chars[i - 1]
        return ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(data["chars"])


class WordTokenizer(Tokenizer):
    kind = "word"
    display_name = "Mots"
    UNK_ID = 1

    def __init__(self, words: list[str] | None = None) -> None:
        super().__init__()
        self.words = words or []
        self.stoi = {w: i + 2 for i, w in enumerate(self.words)}  # 0 spécial, 1 inconnu

    @property
    def vocab_size(self) -> int:
        return len(self.words) + 2

    def fit(self, chunks: Iterable[str], max_vocab: int = 8000, **_: object) -> None:
        counts: Counter[str] = Counter()
        for text in chunks:
            counts.update(_WORD_ONLY.findall(text.lower()))
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.words = [w for w, _ in ordered[: max(16, max_vocab)]]
        self.stoi = {w: i + 2 for i, w in enumerate(self.words)}

    def _encode_plain(self, text: str) -> list[int]:
        stoi = self.stoi
        return [stoi.get(w, self.UNK_ID) for w in _WORD_ONLY.findall(text.lower())]

    def _decode_id(self, i: int) -> str:
        if i == SPECIAL_ID:
            return SPECIAL
        if i == self.UNK_ID:
            return ""
        if 2 <= i < self.vocab_size:
            return " " + self.words[i - 2]
        return ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "words": self.words}

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(data["words"])


class BPETokenizer(Tokenizer):
    """BPE au niveau octet, entraîné sur les mots les plus fréquents du corpus.

    - base : les 256 octets (donc aucun caractère « inconnu », accents et emoji
      sont toujours représentables) ;
    - puis des fusions de paires, apprises itérativement sur les mots du corpus.
    """

    kind = "bpe"
    display_name = "BPE (sous-mots)"
    BASE = 257  # 0 = spécial, 1..256 = octets
    MAX_MERGES = 60000

    def __init__(self, merges: dict[tuple[int, int], int] | None = None) -> None:
        super().__init__()
        self.merges: dict[tuple[int, int], int] = {}
        if merges:
            self.merges = {tuple(k): v for k, v in merges.items()}  # type: ignore[misc]

    # -- vocabulaire ----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.BASE + len(self.merges)

    # -- entraînement ---------------------------------------------------
    @staticmethod
    def _get_stats(vocab: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for word, freq in vocab.items():
            if len(word) < 2:
                continue
            for pair in zip(word, word[1:]):
                counts[pair] += freq
        return counts

    @staticmethod
    def _merge_word(word: tuple[int, ...], pair: tuple[int, int], idx: int) -> tuple[int, ...]:
        out: list[int] = []
        i, n = 0, len(word)
        first, second = pair
        while i < n:
            if i < n - 1 and word[i] == first and word[i + 1] == second:
                out.append(idx)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def fit(
        self,
        chunks: Iterable[str],
        vocab_size: int = 1024,
        sample_chars: int = 1_200_000,
        min_freq: int = 2,
        max_words: int = 10_000,
        progress: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        **_: object,
    ) -> None:
        target_merges = max(1, min(vocab_size - self.BASE, self.MAX_MERGES))

        # 1) fréquence des « mots » sur un échantillon du corpus
        counts: Counter[str] = Counter()
        seen = 0
        for text in chunks:
            seen += len(text)
            counts.update(_PRETOK.findall(text))
            if seen >= sample_chars:
                break
            if should_stop and should_stop():
                break
        if progress:
            progress(0.05, f"{len(counts)} mots distincts dans l'échantillon")

        vocab: dict[tuple[int, ...], int] = {}
        for word, freq in sorted(counts.items(), key=lambda kv: -kv[1])[:max_words]:
            if freq < min_freq and len(vocab) > 500:
                continue
            vocab[tuple(b + 1 for b in word.encode("utf-8"))] = freq
        if not vocab:
            raise ValueError("Corpus vide : impossible d'entraîner le BPE.")

        # 2) fusions successives de la paire la plus fréquente
        self.merges = {}
        for step in range(target_merges):
            stats = self._get_stats(vocab)
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            if freq < 2:
                break
            idx = self.BASE + step
            self.merges[pair] = idx
            new_vocab: dict[tuple[int, ...], int] = {}
            for word, wfreq in vocab.items():
                merged = self._merge_word(word, pair, idx)
                new_vocab[merged] = new_vocab.get(merged, 0) + wfreq
            vocab = new_vocab
            if progress and (step % 25 == 0 or step == target_merges - 1):
                progress(0.05 + 0.95 * (step + 1) / target_merges, f"fusion {step + 1}/{target_merges}")
            if should_stop and should_stop():
                break

    # -- encodage -------------------------------------------------------
    def _apply_merges(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        merges = self.merges
        while len(token_ids) >= 2:
            best_id = math.inf  # l'id d'un token de fusion croît avec son rang
            best_pos = -1
            for i in range(len(token_ids) - 1):
                merged_id = merges.get((token_ids[i], token_ids[i + 1]))
                if merged_id is not None and merged_id < best_id:
                    best_id = merged_id
                    best_pos = i
            if best_pos < 0:
                break
            token_ids = token_ids[:best_pos] + (best_id,) + token_ids[best_pos + 2 :]
        return token_ids

    def _encode_word(self, word: str) -> list[int]:
        cached = self._encode_cache.get(word)
        if cached is not None:
            return cached
        ids = list(self._apply_merges(tuple(b + 1 for b in word.encode("utf-8"))))
        if len(self._encode_cache) > 200_000:
            self._encode_cache.clear()
        self._encode_cache[word] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for match in _PRETOK.finditer(text):
            out.extend(self._encode_word(match.group()))
        return out

    def _bytes_of_id(self, i: int) -> bytes:
        """Octets bruts d'un token (mis en cache)."""
        cached = self._decode_cache.get(i)
        if cached is not None and isinstance(cached, bytes):
            return cached
        if 1 <= i <= 256:
            data = bytes([i - 1])
        else:
            rev = {v: k for k, v in self.merges.items()}
            if i in rev:
                a, b = rev[i]
                data = self._bytes_of_id(a) + self._bytes_of_id(b)
            else:
                data = b""
        self._decode_cache[i] = data  # type: ignore[assignment]
        return data

    def decode(self, ids: Iterable[int]) -> str:
        """Décodage par accumulation d'octets : les accents restent intacts."""
        buffer = bytearray()
        for raw in ids:
            token_id = int(raw)
            if token_id == SPECIAL_ID:
                buffer += SPECIAL.encode("utf-8")
            else:
                buffer += self._bytes_of_id(token_id)
        return buffer.decode("utf-8", errors="replace")

    def _decode_id(self, i: int) -> str:
        if i == SPECIAL_ID:
            return SPECIAL
        return self._bytes_of_id(i).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "merges": [[list(k), v] for k, v in self.merges.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        merges = {(int(k[0]), int(k[1])): int(v) for k, v in data["merges"]}
        return cls(merges)


TOKENIZER_CLASSES = {
    "char": CharTokenizer,
    "word": WordTokenizer,
    "bpe": BPETokenizer,
}


def create_tokenizer(kind: str) -> Tokenizer:
    if kind not in TOKENIZER_CLASSES:
        raise ValueError(f"Type de tokenizer inconnu : {kind}")
    return TOKENIZER_CLASSES[kind]()


def tokenizer_labels() -> dict[str, str]:
    return {k: v.display_name for k, v in TOKENIZER_CLASSES.items()}
