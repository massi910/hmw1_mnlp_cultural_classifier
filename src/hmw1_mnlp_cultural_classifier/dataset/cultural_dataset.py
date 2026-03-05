import logging

from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import Dataset

from hmw1_mnlp_cultural_classifier.labels_schema.cultural_labels import CulturalLabels

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class CulturalDataset(Dataset):

    def __init__(self,
                 source_dataset_name: str,
                 split: str,
                 hf_key: str
                 ):
        # login to huggingface
        login(hf_key)
        self.source_dataset = load_dataset(source_dataset_name)[split]
        self.labels_schema: CulturalLabels = CulturalLabels()
