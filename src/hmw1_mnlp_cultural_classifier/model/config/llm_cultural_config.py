from transformers import PretrainedConfig


class LlmCulturalConfig(PretrainedConfig):

    model_type = "llm_cultural"

    def __init__(
        self,
        num_labels: int = 3,
        pretrained_name: str = "distilbert-base-uncased",
        id2label=None,
        label2id=None,
        **kwargs
    ):
        if id2label is None:
            id2label = {
                0: "cultural agnostic",
                1: "cultural representative",
                2: "cultural exclusive"
            }

        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs
        )

        self.pretrained_name = pretrained_name



class LlmCulturalConfig(PretrainedConfig):

    model_type = "llm_cultural"

    def __init__(
            self,
            num_labels: int = 3,
            pretrained_name: str = "distilbert-base-uncased",
            id2label=None,
            label2id=None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if id2label is None:
            id2label = {
                0: "cultural agnostic",
                1: "cultural representative",
                2: "cultural exclusive"
            }

        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.num_labels = num_labels
        self.pretrained_name = pretrained_name

