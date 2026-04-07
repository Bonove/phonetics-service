"""Phonemizer wrapper — zet namen om naar IPA fonemen."""

import numpy as np
import faiss
from phonemizer import phonemize
from phonemizer.separator import Separator

from config import PHONEMIZER_LANGUAGE, PHONEMIZER_BACKEND, FAISS_TOP_K, SIMILARITY_THRESHOLD

_SEPARATOR = Separator(phone=" ", word="", syllable="")


def phonemize_name(name: str) -> str:
    if not name or not name.strip():
        return ""
    result = phonemize(
        name.strip(),
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )
    return result.strip()


def phonemize_batch(names: list[str]) -> list[str]:
    cleaned = [n.strip() if n else "" for n in names]
    non_empty = [n for n in cleaned if n]

    if not non_empty:
        return [""] * len(names)

    results = phonemize(
        non_empty,
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )

    output = []
    result_iter = iter(results if isinstance(results, list) else [results])
    for n in cleaned:
        if n:
            output.append(next(result_iter).strip())
        else:
            output.append("")
    return output


def _phonemes_to_vector(phonemes: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    chars = phonemes.replace(" ", "")
    if not chars:
        return vec

    for n in (2, 3):
        for i in range(len(chars) - n + 1):
            ngram = chars[i : i + n]
            h = hash(ngram) % dim
            vec[h] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class PhoneticIndex:
    def __init__(self, names: list[str]):
        self._names = list(names)
        self._phonemes: list[str] = []
        self._index = None

        if not names:
            return

        self._phonemes = phonemize_batch(names)

        dim = 128
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in self._phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._names)

    def search(self, query: str, top_k: int = FAISS_TOP_K) -> list[dict]:
        if not self._index or self.size == 0:
            return []

        query_phonemes = phonemize_name(query)
        if not query_phonemes:
            return []

        query_vec = _phonemes_to_vector(query_phonemes).reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            clamped_score = float(max(0.0, min(1.0, score)))
            if clamped_score >= SIMILARITY_THRESHOLD:
                results.append({
                    "name": self._names[idx],
                    "score": round(clamped_score, 4),
                    "phonemes": self._phonemes[idx],
                })
        return results
