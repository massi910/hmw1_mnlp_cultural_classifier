from huggingface_hub import login
from transformers import AutoTokenizer

from HMW1_2025.model.llm_cultural_model import LlmCulturalModel

model_path = "/HMW1_2025/runs/out/train/v1.0.0/model"
repo_name = "massi910/cultural_model_test"

login("x")


model = LlmCulturalModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
