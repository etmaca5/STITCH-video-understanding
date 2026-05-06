"""Test InternVideo2 Stage2 with 8 frames per window on a longer video.

Checks whether there is enough RAM/VRAM to run the model and measures timing.
"""

import sys
import time
import psutil
import os

sys.path.insert(0, "src")
sys.path.insert(0, "models/InternVideo2-Stage2_1B-224p-f4")

import numpy as np
import torch

from retrieval import InternVideo2Backend
from chunking import Chunking

VIDEO_PATH = "customhighlightsdata/Animal_vids.mp4"
MODEL_DIR = "models/InternVideo2-Stage2_1B-224p-f4"
NUM_FRAMES = 16
ORIG_NUM_FRAMES = 4  # checkpoint native frame count
SAMPLE_INTERVAL = 2.0  # seconds between window starts


def ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


print(f"RAM before model load: {ram_gb():.2f} GB")
print(f"VRAM before model load: {vram_gb():.2f} GB")
print()
print(f"Loading InternVideo2 with num_frames={NUM_FRAMES}, orig_num_frames={ORIG_NUM_FRAMES} ...")

t0 = time.time()
backend = InternVideo2Backend(
    model_dir=MODEL_DIR,
    num_frames=NUM_FRAMES,
    orig_num_frames=ORIG_NUM_FRAMES,
)
print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"RAM after model load: {ram_gb():.2f} GB")
print(f"VRAM after model load: {vram_gb():.2f} GB")
print()

chunker = Chunking()

print(f"Computing window embeddings for {VIDEO_PATH} ...")
print(f"  num_frames={NUM_FRAMES}, sample_interval={SAMPLE_INTERVAL}s")
t1 = time.time()
window_times, window_embeddings = chunker._compute_window_embeddings(
    VIDEO_PATH, backend, sample_interval=SAMPLE_INTERVAL
)
elapsed = time.time() - t1

print()
print(f"Done in {elapsed:.1f}s")
print(f"  Windows computed: {len(window_times)}")
print(f"  Embedding shape: {window_embeddings.shape}")
print(f"RAM after inference: {ram_gb():.2f} GB")
print(f"VRAM after inference: {vram_gb():.2f} GB")
