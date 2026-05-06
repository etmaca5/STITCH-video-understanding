import glob
import sys

import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.nn.functional import cosine_similarity


class EmbeddingBackend(ABC):
    """Abstract base for text-video embedding models.

    Each backend wraps a specific model (XCLIP, InternVideo2, etc.) and
    provides a uniform interface for encoding text and video into a shared
    embedding space. All returned embeddings are L2-normalized numpy arrays.
    """

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Number of video frames this model expects as input."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Encode text into the shared embedding space.

        Returns:
            1-D numpy array of shape (embed_dim,), L2-normalized.
        """
        pass

    @abstractmethod
    def embed_video(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode video frames into the shared embedding space.

        Args:
            frames: exactly self.num_frames RGB uint8 arrays, each (H, W, 3).

        Returns:
            1-D numpy array of shape (embed_dim,), L2-normalized.
        """
        pass

    def embed_video_batch(self, frames_list: list[list[np.ndarray]],
                          batch_size: int = 8) -> np.ndarray:
        """Encode multiple video clips. Returns (N, embed_dim) array.

        Default implementation calls embed_video sequentially. Backends
        may override with true GPU-batched inference.
        """
        return np.stack([self.embed_video(f) for f in frames_list])

    def embed_text_batch(self, texts: list[str],
                         batch_size: int = 16) -> np.ndarray:
        """Encode multiple texts. Returns (N, embed_dim) array.

        Default implementation calls embed_text sequentially. Backends
        may override with true GPU-batched inference.
        """
        return np.stack([self.embed_text(t) for t in texts])

    def cache_identity(self) -> dict:
        """Return cache-stable backend metadata for window embedding reuse."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must define cache_identity()"
        )


class XCLIPBackend(EmbeddingBackend):
    """XCLIP backend using HuggingFace transformers.

    .. deprecated::
        Microsoft's X-CLIP was designed for video classification, not retrieval.
        It underperforms InternVideo2 significantly on retrieval benchmarks.
        Use InternVideo2Backend instead.

    Uses processor's `images=` key (list of RGB frames) for video encoding.
    Text and video features are extracted from pooler_output.
    """

    def __init__(self, model_path="models/xclip-base-patch16-kinetics-600",
                 device=None):
        from transformers import AutoProcessor, AutoTokenizer, AutoModel

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self._num_frames = self.model.config.vision_config.num_frames

    @property
    def num_frames(self):
        return self._num_frames

    def embed_text(self, text):
        inputs = self.tokenizer(
            [text], padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            result = self.model.get_text_features(**inputs)
        features = result.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    def embed_video(self, frames):
        inputs = self.processor(
            images=frames, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            result = self.model.get_video_features(**inputs)
        features = result.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    def cache_identity(self):
        return {
            "backend": "xclip",
            "model_path": str(self.model.name_or_path),
            "num_frames": int(self._num_frames),
        }


class InternVideo2Backend(EmbeddingBackend):
    """InternVideo2 Stage2 backend.

    Supports configurable frame counts.  When
    num_frames differs from the checkpoint's native frame count (given by
    orig_num_frames), temporal position embeddings are interpolated
    automatically.  This allows e.g. running the f4 checkpoint with 8
    frames for denser temporal sampling.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def __init__(self, model_dir="models/InternVideo2-Stage2_1B-224p-f4",
                 device=None, num_frames=4, orig_num_frames=None,
                 image_size=224, use_flash_attn=False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._model_dir = str(model_dir)
        self._num_frames = num_frames
        self._image_size = image_size
        self._use_flash_attn = use_flash_attn
        if orig_num_frames is None:
            orig_num_frames = num_frames
        self._orig_num_frames = orig_num_frames
        if use_flash_attn:
            if device.type != "cuda":
                raise ValueError("InternVideo2 flash-attn requires a CUDA device")
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "flash-attn is not installed. Install it before setting "
                    "use_flash_attn=True."
                ) from exc

        sys.path.insert(0, model_dir)
        from modeling_internvideo2 import (
            InternVideo2_Stage2, InternVideo2_Stage2_Config,
            interpolate_pos_embed_internvideo2_new,
        )

        config_dict = {
            "model": {
                "embed_dim": 512,
                "model_cls": "InternVideo2_Stage2",
                "multimodal": {"enable": True},
                "temp": 0.07,
                "find_unused_parameters": False,
                "text_encoder": {
                    "config": f"{model_dir}/config_bert_large.json",
                    "d_model": 1024,
                    "fusion_layer": 19,
                    "name": "bert_large",
                    "pretrained": "bert-large-uncased",
                },
                "vision_encoder": {
                    "name": "pretrain_internvideo2_1b_patch14_224",
                    "d_model": 1408,
                    "clip_embed_dim": 768,
                    "clip_input_resolution": image_size,
                    "clip_norm_type": "l2",
                    "clip_return_layer": 6,
                    "clip_student_return_interval": 1,
                    "clip_teacher": None,
                    "clip_teacher_embed_dim": 3200,
                    "clip_teacher_final_dim": 768,
                    "clip_teacher_return_interval": 1,
                    "image_mask_ratio": 0.5,
                    "image_mask_type": "random",
                    "img_size": image_size,
                    "keep_temporal": False,
                    "num_frames": num_frames,
                    "only_mask": True,
                    "patch_size": 14,
                    "pretrained": "",
                    "sep_image_video_pos_embed": True,
                    "tubelet_size": 1,
                    "use_checkpoint": False,
                    "checkpoint_num": 0,
                    "use_flash_attn": use_flash_attn,
                    "use_fused_mlp": use_flash_attn,
                    "use_fused_rmsnorm": use_flash_attn,
                    "video_mask_ratio": 0.8,
                    "video_mask_type": "random",
                },
            },
            "device": str(device),
            "max_txt_l": 40,
            "size_t": image_size,
            "num_frames": num_frames,
            "num_frames_test": num_frames,
            "origin_num_frames": orig_num_frames,
            "gradient_checkpointing": False,
        }

        config = InternVideo2_Stage2_Config(**config_dict)
        self.model = InternVideo2_Stage2(config=config, is_pretrain=True)

        ckpt_files = glob.glob(f"{model_dir}/*.pt")
        if not ckpt_files:
            raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")
        ckpt = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
        state_dict = ckpt["module"]
        del ckpt

        fixed = {}
        for k, v in state_dict.items():
            fixed[k.replace(".ls1.gamma", ".ls1.weight")
                   .replace(".ls2.gamma", ".ls2.weight")] = v
        state_dict = fixed

        interpolate_pos_embed_internvideo2_new(
            state_dict, self.model.vision_encoder, orig_t_size=orig_num_frames
        )
        self.model.load_state_dict(state_dict, strict=False)
        del state_dict

        self.model = self.model.to(device).eval().float()

    @property
    def num_frames(self):
        return self._num_frames

    def embed_text(self, text):
        with torch.no_grad():
            feat = self.model.get_txt_feat(text)
        return feat.cpu().numpy().squeeze()

    def _preprocess_frames(self, frames):
        """Preprocess a single clip's frames to (T, C, H, W) numpy array."""
        size = self._image_size
        processed = []
        for f in frames:
            f = cv2.resize(f, (size, size))
            f = (f / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
            processed.append(f)
        vid = np.stack(processed)              # (T, H, W, C)
        return np.transpose(vid, (0, 3, 1, 2)) # (T, C, H, W)

    def embed_video(self, frames):
        vid = np.expand_dims(self._preprocess_frames(frames), 0)  # (1,T,C,H,W)
        tensor = torch.from_numpy(vid).to(self.device).float()
        with torch.no_grad():
            feat = self.model.get_vid_feat(tensor)
        return feat.cpu().numpy().squeeze()

    def embed_video_batch(self, frames_list, batch_size=8):
        results = []
        for i in range(0, len(frames_list), batch_size):
            batch_vids = np.stack(
                [self._preprocess_frames(f) for f in frames_list[i:i + batch_size]]
            )
            tensor = torch.from_numpy(batch_vids).to(self.device).float()
            with torch.no_grad():
                feat = self.model.get_vid_feat(tensor)
            feat = feat.cpu().numpy()
            if feat.ndim == 3:
                feat = feat[:, 0, :]
            results.append(feat)
        return np.concatenate(results, axis=0)

    def embed_text_batch(self, texts, batch_size=16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                feat = self.model.get_txt_feat(batch)
            feat = feat.cpu().numpy()
            if feat.ndim == 3:
                feat = feat[:, 0, :]
            results.append(feat)
        return np.concatenate(results, axis=0)

    def cache_identity(self):
        return {
            "backend": "internvideo2",
            "model_dir": self._model_dir,
            "num_frames": int(self._num_frames),
            "orig_num_frames": int(self._orig_num_frames),
            "image_size": int(self._image_size),
            "use_flash_attn": bool(self._use_flash_attn),
        }


class Retrieval:
    """Match a query (text or video chunk) to candidate video chunks.

    Uses an EmbeddingBackend to embed both the query and candidates, then
    ranks candidates by cosine similarity.
    """

    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend
        self.results = []

    def retrieve(self, query, video_path, chunks):
        """Rank chunks by similarity to a query.

        Args:
            query: either a text string, or a (video_path, (start, end)) tuple
                for a video-to-video query.
            video_path: path to the video containing candidate chunks.
            chunks: list of (start_frame, end_frame) tuples.
        """
        if isinstance(query, str):
            query_emb = self.backend.embed_text(query)
        else:
            q_video, q_chunk = query
            q_frames = self._load_chunk_frames(q_video, q_chunk)
            query_emb = self.backend.embed_video(q_frames)

        query_t = torch.tensor(query_emb).unsqueeze(0)

        self.results = []
        for chunk in chunks:
            frames = self._load_chunk_frames(video_path, chunk)
            emb = self.backend.embed_video(frames)
            cand_t = torch.tensor(emb).unsqueeze(0)
            sim = cosine_similarity(query_t, cand_t).item()
            self.results.append({"chunk": chunk, "similarity": sim})

        self.results.sort(key=lambda x: x["similarity"], reverse=True)

    def get_ranked_results(self):
        """Return all results sorted by similarity (descending)."""
        return self.results

    def get_top_chunk(self):
        """Return the most similar chunk and its score, or None."""
        if not self.results:
            return None
        return self.results[0]

    def _load_chunk_frames(self, video_path, chunk):
        """Sample self.backend.num_frames uniform RGB frames from a chunk."""
        start_frame, end_frame = chunk
        n = self.backend.num_frames
        indices = np.linspace(
            start_frame, max(start_frame, end_frame - 1), n
        ).astype(int)

        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"Could not read any frames from {video_path} "
                f"[frames {start_frame} - {end_frame}]"
            )
        while len(frames) < n:
            frames.append(frames[-1])

        return frames
