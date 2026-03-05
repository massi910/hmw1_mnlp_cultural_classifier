from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, ClassVar
import numpy as np


@dataclass
class W2VEmbedder:
    """
    Singleton wrapper around pretrained Word2Vec / GloVe KeyedVectors.
    Ensures embeddings are loaded only once.
    """
    keyed_vectors: any  # gensim.models.KeyedVectors
    embedding_dim: int = 300

    _instance: ClassVar[Optional["W2VEmbedder"]] = None

    @classmethod
    def get_instance(cls, model: str = "large") -> "W2VEmbedder":
        """
        Returns the singleton instance. Loads model on first call.
        model: "large" (Google News) or "small" (GloVe 50d)
        """
        if cls._instance is None:
            import gensim.downloader as api

            if model == "large":
                kv = api.load("word2vec-google-news-300")
            elif model == "small":
                kv = api.load("glove-wiki-gigaword-50")
            else:
                raise ValueError(f"Unknown model: {model}")

            cls._instance = cls(keyed_vectors=kv, embedding_dim=kv.vector_size)

        return cls._instance

    def has(self, token: str) -> bool:
        return token in self.keyed_vectors

    def get(self, token: str) -> Optional[np.ndarray]:
        if token in self.keyed_vectors:
            return self.keyed_vectors[token]
        return None

    def embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Returns array [T, D]. Unknown tokens are skipped.
        If no tokens found, returns [1, D] zeros (safe fallback).
        """
        vecs = []
        for t in tokens:
            v = self.get(t)
            if v is not None:
                vecs.append(v)

        if not vecs:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        return np.asarray(vecs, dtype=np.float32)