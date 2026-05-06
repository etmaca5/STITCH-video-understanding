"""
Download V-JEPA2 model weights into model/VJEPA2/.

Downloads facebook/vjepa2-vitl-fpc64-256 (ViT-Large, 0.3B params, ~6GB)
from HuggingFace.

See all available models:
  https://huggingface.co/collections/facebook/v-jepa-2

Usage:
    python out_of_use/download_vjepa2.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "VJEPA2"
REPO_ID = "facebook/vjepa2-vitl-fpc64-256"


def main():
    dest = MODEL_DIR / "vjepa2-vitl-fpc64-256"
    if dest.exists() and (dest / "model.safetensors").exists():
        print(f"Already downloaded: {dest}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {dest}")
    snapshot_download(repo_id=REPO_ID, local_dir=str(dest))
    print(f"Done. Model saved to {dest}")


if __name__ == "__main__":
    main()
