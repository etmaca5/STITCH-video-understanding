from transformers import AutoModel


model = AutoModel.from_pretrained(
    "OpenGVLab/VideoMAEv2-Base",
    trust_remote_code=True,
    cache_dir="model",
)