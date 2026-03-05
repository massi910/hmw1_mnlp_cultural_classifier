from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

_TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class SimpleTokenizer:
    """
    Minimal tokenizer for Word2Vec lookup.
    """
    lowercase: bool = True

    def tokenize(self, text: str) -> List[str]:
        toks = _TOKEN_RE.findall(text or "")
        if self.lowercase:
            toks = [t.lower() for t in toks]
        return toks
