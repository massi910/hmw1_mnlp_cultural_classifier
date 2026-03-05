from abc import ABC, abstractmethod
from typing import List, Dict, Union

from torch import Tensor


class BaseTokenizer(ABC):

    def __init__(
        self,
        tokenizer_pretrained_name: Union[str, None],
        max_seq_len: int = 64
    ) -> None:

        self.tokenizer_pretrained_name = tokenizer_pretrained_name
        self.max_seq_len = max_seq_len


    @abstractmethod
    def encode(self, texts: List[str]) -> Dict[str, Tensor]:
        """
        Tokenize a batch of texts.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def push_to_hub(self, repo_name: str):
        pass

    @classmethod
    def from_pretrained(
            cls,
            tokenizer_pretrained_name_or_path: Union[str, None],
            max_seq_len: int
    ) -> "BaseTokenizer":
        """
        Load from:
            - HF hub
            - local directory
            - model repo
        """
        return cls(
            tokenizer_pretrained_name=tokenizer_pretrained_name_or_path,
            max_seq_len=max_seq_len
        )
