from transformers import DistilBertTokenizerFast

from hmw1_mnlp_cultural_classifier.tokenizer.base_tokenizer import BaseTokenizer


class DistilBertTokenizer(BaseTokenizer):
    def __init__(self,
                 tokenizer_pretrained_name: str,
                 max_seq_len: int = 64
                 ):

        super().__init__(tokenizer_pretrained_name, max_seq_len)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            tokenizer_pretrained_name
        )
        self.max_seq_len = max_seq_len

    def encode(self, texts):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)


    def push_to_hub(self, repo_name: str):
        self.tokenizer.push_to_hub(repo_name)
