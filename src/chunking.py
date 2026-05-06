import logging
import numpy as np
import cv2
import torch
import sys
import types
import importlib.util
import json
import subprocess
from pathlib import Path
from torch.nn.functional import cosine_similarity
from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from safetensors.torch import load_file

try:
    from chunking_cache import WindowEmbeddingCache, WindowCacheMissError
except ImportError:  # pragma: no cover - package-style imports
    from .chunking_cache import WindowEmbeddingCache, WindowCacheMissError


log = logging.getLogger(__name__)
WINDOW_EMBED_BATCH_SIZE = 8
# ruptures requires pen > 0; BIC default can be 0 when sim variance is 0.
_MIN_RUPTURES_PEN = 1e-12


def load_videomaev2_model(model_dir, device=None):
    """Deterministic local loader for VideoMAEv2 (no HF meta-init path).

    This avoids `AutoModel.from_pretrained(...)` meta-tensor initialization,
    which can fail for this custom checkpoint in some transformers versions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(model_dir)
    cfg_file = model_dir / "modeling_config.py"
    model_file = model_dir / "modeling_videomaev2.py"
    weights_file = model_dir / "model.safetensors"
    config_json = model_dir / "config.json"

    pkg_name = "videomaev2_local"
    sys.modules.pop(pkg_name, None)
    sys.modules.pop(f"{pkg_name}.modeling_config", None)
    sys.modules.pop(f"{pkg_name}.modeling_videomaev2", None)
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(model_dir)]
    sys.modules[pkg_name] = pkg

    spec_cfg = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_config",
        cfg_file,
    )
    cfg_module = importlib.util.module_from_spec(spec_cfg)
    cfg_module.__package__ = pkg_name
    sys.modules[f"{pkg_name}.modeling_config"] = cfg_module
    spec_cfg.loader.exec_module(cfg_module)

    spec_model = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_videomaev2",
        model_file,
    )
    model_module = importlib.util.module_from_spec(spec_model)
    model_module.__package__ = pkg_name
    sys.modules[f"{pkg_name}.modeling_videomaev2"] = model_module
    spec_model.loader.exec_module(model_module)

    VideoMAEv2Config = cfg_module.VideoMAEv2Config
    VideoMAEv2 = model_module.VideoMAEv2

    with open(config_json, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = VideoMAEv2Config(**cfg_dict)
    model = VideoMAEv2(config)

    state_dict = load_file(str(weights_file))
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), "local"

# TODO: the naming of this class is confusing - why videomae - it's just encoder wrapper for embedding 
class VideoMAEv2EmbeddingWrapper:
    """Wraps a loaded VideoMAEv2 model to match the embed_video/num_frames
    interface used by InternVideo2Backend, so both can be used interchangeably
    as the chunking embedding backend.
    """

    def __init__(self, model, device, num_frames=16, image_size=224):
        self.model = model
        self.device = device
        self._num_frames = num_frames
        self._image_size = image_size

    @property
    def num_frames(self):
        return self._num_frames

    def _preprocess_frames(self, frames):
        """Preprocess a single clip's frames to (C, T, H, W) numpy array."""
        size = self._image_size
        processed = []
        for f in frames:
            f = cv2.resize(f, (size, size))
            f = (f / 255.0 - 0.5) / 0.5
            processed.append(f)
        frames_np = np.stack(processed)                           # (T, H, W, C)
        return np.transpose(frames_np, (3, 0, 1, 2))             # (C, T, H, W)

    def embed_video(self, frames):
        """Encode raw RGB uint8 frames into an embedding vector.

        Args:
            frames: list of num_frames RGB uint8 arrays, each (H, W, 3).

        Returns:
            1-D numpy array of shape (embed_dim,).
        """
        x = self._preprocess_frames(frames)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.extract_features(x)
        return emb.cpu().numpy().squeeze()

    def embed_video_batch(self, frames_list, batch_size=8):
        """Encode multiple clips. Returns (N, embed_dim) array."""
        results = []
        for i in range(0, len(frames_list), batch_size):
            batch_clips = np.stack(
                [self._preprocess_frames(f) for f in frames_list[i:i + batch_size]]
            )
            batch = torch.tensor(batch_clips, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                emb = self.model.extract_features(batch)
            feat = emb.cpu().numpy()
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1)
            results.append(feat)
        return np.concatenate(results, axis=0)

    def cache_identity(self):
        return {
            "backend": "videomaev2",
            "model_name": type(self.model).__name__,
            "num_frames": int(self._num_frames),
            "image_size": int(self._image_size),
        }


class Chunking:
    """
    Different types of chunking (core of this system):
    1. Frame difference / content detector chunking from pyscenedetect
    https://arxiv.org/pdf/2506.10807
    2. Embedding based chunking (VideoMAEv2 or InternVideo2)
    - compare sampled embeddings and if they are different enough we set a boundary

    3. Surprise Based chunking using V-JEPA2

    We use V-JEPA2's predictor to predict future latent representations from context.
    The prediction error (cosine distance) measures surprise — high error means
    the future was hard to predict from the past, indicating a scene boundary.

    Combinations of these should also be possible - in particular for 2 and 3 can be mixed
    with 1, these can be sampled and then the frame difference used to refine the exact
    boundary in between them to the point with the highest frame difference.

    Ideally we will also compare 1 to 2 + 1 and to 3 + 1 to see what the differences
    typically are for different types of video

    TODO:
    - Explore alternative adaptive threshold strategies (percentile, MDLSeg, etc.)
    - Motion based chunking as an alternative to content detector
    """

    def __init__(self, chunking_type="content_detector"):
        self.chunking_type = chunking_type
        self.fps = None
        self.window_embedding_cache = WindowEmbeddingCache()

    def chunk(self, video_path, model=None, processor=None, device=None,
              embedding_backend=None):
        """Run chunking on a video.

        Returns dict with keys:
            chunks: list of (start_frame, end_frame) tuples
            signal_times: sample times in seconds (numpy array)
            signal_values: signal values at those times (numpy array)
            threshold: detection threshold used
            fps: video frame rate
            total_frames: total frames in video
            window_times: per-window times (numpy array)
            window_embeddings: per-window embeddings (numpy array)
        """
        if self.chunking_type == "content_detector":
            return self.content_detector(
                video_path, embedding_backend=embedding_backend, device=device,
            )
        elif self.chunking_type == "embedding":
            return self.embedding_chunking(video_path, embedding_backend, device)
        elif self.chunking_type == "surprise":
            return self.surprise_chunking(
                video_path, model, processor, device,
                embedding_backend=embedding_backend,
            )
        else:
            raise ValueError(f"Unknown chunking type: {self.chunking_type}")

    # ---- 1. Content detector chunking ----

    def content_detector(self, video_path, threshold=27.0, min_scene_len=15,
                          embedding_backend=None, device=None,
                          embedding_sample_interval=1.0):
        """Detect scene boundaries using pyscenedetect's ContentDetector."""
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video = open_video(video_path)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager=stats_manager)
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video)

        scenes = []
        for start, end in scene_manager.get_scene_list():
            scenes.append((start.get_frames(), end.get_frames()))

        if not scenes:
            scenes = [(0, total_frames)]

        scores = np.full(total_frames, np.nan, dtype=float)
        for idx in range(total_frames):
            vals = stats_manager.get_metrics(idx, ["content_val"])
            if vals and vals[0] is not None:
                scores[idx] = float(vals[0])

        valid = np.isfinite(scores)
        signal_times = np.where(valid)[0] / self.fps
        signal_values = scores[valid]

        result = {
            "chunks": scenes,
            "signal_times": signal_times,
            "signal_values": signal_values,
            "threshold": float(threshold),
            "high_is_change": True,
            "fps": self.fps,
            "total_frames": total_frames,
        }

        if embedding_backend is not None:
            window_times, window_embeddings = self._compute_window_embeddings(
                video_path, embedding_backend, embedding_sample_interval,
            )
            result["window_times"] = window_times
            result["window_embeddings"] = window_embeddings

        return result

    # ---- 2. Embedding based chunking ----

    def embedding_chunking(self, video_path, embedding_backend, device=None,
                           sample_interval=1.0, k=2.0,
                           refine_boundaries=True,
                           threshold_method="std", penalty=None,
                           min_segment_windows=2, require_cache=False):
        """Chunk by comparing embeddings of consecutive windows.

        Works with any backend that exposes embed_video(frames) and num_frames
        (e.g. VideoMAEv2EmbeddingWrapper or InternVideo2Backend).

        threshold_method controls how boundaries are detected:
          "kernel_cpd" - ruptures KernelCPD with cosine kernel on embeddings
                         (the supported method)
          "std"        - boundary where similarity < mean - k * std (legacy)
          "pelt"       - DEPRECATED. PELT on the 1D similarity signal.

        penalty: controls granularity for pelt/kernel_cpd. None = BIC default
                 (var(signal) * 2 * log(n)), or pass a float for manual override.
        """
        if embedding_backend is None:
            raise ValueError("Must pass an embedding backend")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        window_data = self.get_window_embeddings(
            video_path,
            embedding_backend,
            sample_interval=sample_interval,
            fps=self.fps,
            total_frames=total_frames,
            require_cache=require_cache,
        )
        self.fps = float(window_data["fps"])
        total_frames = int(window_data["total_frames"])
        window_starts = window_data["window_times"]
        embeddings_np = window_data["window_embeddings"]
        embeddings_t = torch.tensor(embeddings_np)
        similarities = cosine_similarity(embeddings_t[:-1], embeddings_t[1:], dim=1)
        sim_np = similarities.numpy()

        if threshold_method == "std":
            threshold = sim_np.mean() - k * sim_np.std()
            boundary_indices = [i for i, s in enumerate(sim_np) if s < threshold]
            threshold = float(threshold)
        elif threshold_method == "kernel_cpd":
            import ruptures as rpt
            n = len(embeddings_np)
            if n < 2 * min_segment_windows:
                # signal too short for any interior change point; one segment
                boundary_indices = []
            else:
                pen = penalty if penalty is not None else sim_np.var() * 2 * np.log(n)
                pen = max(float(pen), _MIN_RUPTURES_PEN)
                algo = rpt.KernelCPD(
                    kernel="cosine", min_size=min_segment_windows,
                ).fit(embeddings_np)
                bkps = algo.predict(pen=pen)
                boundary_indices = [b - 1 for b in bkps[:-1]]
            threshold = None
        elif threshold_method == "pelt":
            # DEPRECATED: kept for back-compat with old configs/result reproductions.
            # Use threshold_method="kernel_cpd" for new runs.
            import ruptures as rpt
            if len(sim_np) < min_segment_windows:
                boundary_indices = []
            else:
                n = len(sim_np)
                pen = penalty if penalty is not None else sim_np.var() * 2 * np.log(n)
                pen = max(float(pen), _MIN_RUPTURES_PEN)
                algo = rpt.Pelt(
                    model="l2", min_size=min_segment_windows,
                ).fit(sim_np.reshape(-1, 1))
                bkps = algo.predict(pen=pen)
                boundary_indices = list(bkps[:-1])
            threshold = None
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")

        scenes = self._boundary_indices_to_scenes(
            boundary_indices, window_starts, sample_interval,
            total_frames, video_path, refine_boundaries,
        )

        return {
            "chunks": scenes,
            "signal_times": window_starts[1:],
            "signal_values": sim_np,
            "threshold": threshold,
            "high_is_change": False,
            "fps": self.fps,
            "total_frames": total_frames,
            "window_times": window_starts,
            "window_embeddings": embeddings_np,
        }

    # ---- 3. Surprise based chunking ----

    def surprise_chunking(self, video_path, model, processor, device=None,
                          window_frames=16, stride_seconds=1.0,
                          sample_fps=15, k=2.0,
                          embedding_backend=None,
                          embedding_sample_interval=1.0,
                          refine_boundaries=True,
                          threshold_method="std", penalty=None,
                          min_segment_windows=2):
        """Detect boundaries where V-JEPA2 prediction error (surprise) is high.

        Slides a window along the video and measures cosine distance between
        predicted and actual latent representations.

        threshold_method controls how boundaries are detected:
          "std"  - boundary where surprise > mean + k * std
          "pelt" - ruptures PELT with L2 cost on the surprise signal

        penalty: controls granularity for pelt. None = BIC default
                 (var(signal) * 2 * log(n)), or pass a float for manual override.

        If embedding_backend is provided, also computes per-window embeddings
        for downstream merging and similarity matching.
        """
        if model is None or processor is None:
            raise ValueError("Must pass a loaded V-JEPA2 model and processor")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_step = max(1, round(self.fps / sample_fps))
        stride_source = max(1, round(stride_seconds * self.fps))
        window_source = window_frames * frame_step

        cfg = model.config
        H = W = cfg.image_size // cfg.patch_size
        T = window_frames // cfg.tubelet_size
        tokens_per_frame = H * W
        total_tokens = T * tokens_per_frame
        context_t = T // 2

        context_indices = torch.arange(context_t * tokens_per_frame)
        target_indices = torch.arange(context_t * tokens_per_frame, total_tokens)
        context_mask = [context_indices.unsqueeze(0).to(device)]
        target_mask = [target_indices.unsqueeze(0).to(device)]

        timestamps = []
        window_bounds = []
        surprises = []

        window_starts = range(0, total_frames - window_source, stride_source)
        for start_frame in window_starts:
            frames = self._load_frames_raw(
                video_path, start_frame, window_frames, step=frame_step
            )
            if len(frames) < window_frames:
                continue

            inputs = processor(frames, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(
                    pixel_values_videos=inputs["pixel_values_videos"],
                    context_mask=context_mask,
                    target_mask=target_mask,
                )

            predicted = outputs.predictor_output.last_hidden_state
            actual = outputs.predictor_output.target_hidden_state
            cos_sim = cosine_similarity(predicted, actual, dim=-1)
            surprise = (1 - cos_sim).mean().item()

            mid_frame = start_frame + window_source // 2
            timestamps.append(mid_frame)
            window_bounds.append((start_frame, start_frame + window_source))
            surprises.append(surprise)

        surprises = np.array(surprises)

        if threshold_method == "std":
            threshold = float(surprises.mean() + k * surprises.std())
            scenes = []
            current_start = 0
            for j, (ts, s) in enumerate(zip(timestamps, surprises)):
                if s > threshold:
                    if refine_boundaries:
                        ws, we = window_bounds[j]
                        boundary = self._refine_boundary(video_path, ws, we)
                    else:
                        boundary = ts
                    if boundary > current_start:
                        scenes.append((current_start, boundary))
                        current_start = boundary
            scenes.append((current_start, total_frames))
        elif threshold_method == "pelt":
            import ruptures as rpt
            n = len(surprises)
            if n < min_segment_windows:
                scenes = [(0, total_frames)]
            else:
                pen = penalty if penalty is not None else surprises.var() * 2 * np.log(n)
                pen = max(float(pen), _MIN_RUPTURES_PEN)
                algo = rpt.Pelt(
                    model="l2", min_size=min_segment_windows,
                ).fit(surprises.reshape(-1, 1))
                bkps = algo.predict(pen=pen)
                scenes = []
                current_start = 0
                for b in bkps[:-1]:
                    if b < len(timestamps):
                        if refine_boundaries:
                            ws, we = window_bounds[b]
                            boundary = self._refine_boundary(video_path, ws, we)
                        else:
                            boundary = timestamps[b]
                        if boundary > current_start:
                            scenes.append((current_start, boundary))
                            current_start = boundary
                scenes.append((current_start, total_frames))
            threshold = None
        else:
            raise ValueError(
                f"Unknown threshold_method for surprise: {threshold_method}. "
                f"Use 'std' or 'pelt'."
            )

        result = {
            "chunks": scenes,
            "signal_times": np.array(timestamps) / self.fps,
            "signal_values": surprises,
            "threshold": threshold,
            "high_is_change": True,
            "fps": self.fps,
            "total_frames": total_frames,
        }

        if embedding_backend is not None:
            window_times, window_embeddings = self._compute_window_embeddings(
                video_path, embedding_backend, embedding_sample_interval,
            )
            result["window_times"] = window_times
            result["window_embeddings"] = window_embeddings

        return result


    # ------ helpers -----

    def _boundary_indices_to_scenes(self, boundary_indices, window_starts,
                                    sample_interval, total_frames, video_path,
                                    refine_boundaries):
        """Convert boundary indices (into the similarity signal / window array) to scenes.

        Each index i means a boundary between window i and window i+1.
        """
        scenes = []
        current_start = 0
        for i in sorted(set(boundary_indices)):
            if i < 0 or i + 1 >= len(window_starts):
                continue
            if refine_boundaries:
                mid_i = window_starts[i] + sample_interval / 2
                mid_i1 = window_starts[i + 1] + sample_interval / 2
                search_start = int(mid_i * self.fps)
                search_end = int(mid_i1 * self.fps)
                boundary = self._refine_boundary(
                    video_path, search_start, search_end,
                )
            else:
                mid_sec = (window_starts[i] + window_starts[i + 1] + sample_interval) / 2
                boundary = int(mid_sec * self.fps)
            if boundary > current_start:
                scenes.append((current_start, boundary))
                current_start = boundary
        scenes.append((current_start, total_frames))
        return scenes

    def _refine_boundary(
        self, video_path, search_start_frame, search_end_frame, strict=False,
    ):
        """Refine a rough boundary by finding the frame with the largest visual change.

        Computes the same content_val metric as pyscenedetect's ContentDetector
        (sum of mean absolute HSV channel differences) but only for frames in
        the search range, avoiding a full-video pass.
        """
        fallback = (search_start_frame + search_end_frame) // 2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        search_start_frame = max(0, search_start_frame)
        search_end_frame = min(search_end_frame, total_frames - 1)

        if search_end_frame <= search_start_frame:
            cap.release()
            if strict:
                raise ValueError(
                    f"Invalid boundary refinement range [{search_start_frame}, "
                    f"{search_end_frame}] for {video_path}."
                )
            return fallback

        cap.set(cv2.CAP_PROP_POS_FRAMES, search_start_frame)
        ok, prev = cap.read()
        if not ok:
            cap.release()
            if strict:
                raise RuntimeError(
                    f"Could not read frame {search_start_frame} while refining "
                    f"boundary for {video_path}."
                )
            return fallback
        prev_h, prev_s, prev_v = cv2.split(cv2.cvtColor(prev, cv2.COLOR_BGR2HSV))

        best_frame = fallback
        best_score = -1.0
        for frame_idx in range(search_start_frame + 1, search_end_frame + 1):
            ok, curr = cap.read()
            if not ok:
                cap.release()
                if strict:
                    raise RuntimeError(
                        f"Could not read frame {frame_idx} while refining "
                        f"boundary for {video_path}."
                    )
                break
            curr_h, curr_s, curr_v = cv2.split(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
            score = (np.mean(cv2.absdiff(curr_h, prev_h))
                     + np.mean(cv2.absdiff(curr_s, prev_s))
                     + np.mean(cv2.absdiff(curr_v, prev_v)))
            if score > best_score:
                best_score = score
                best_frame = frame_idx
            prev_h, prev_s, prev_v = curr_h, curr_s, curr_v

        cap.release()
        if best_score < 0.0 and strict:
            raise RuntimeError(
                f"No valid frame differences were computed while refining "
                f"boundary for {video_path}."
            )
        return best_frame

    def _compute_window_embeddings(self, video_path, embedding_backend,
                                    sample_interval=1.0):
        """Compute embeddings at regular intervals across a video.

        Works with any backend that exposes embed_video(frames) and num_frames
        (e.g. VideoMAEv2EmbeddingWrapper or InternVideo2Backend).

        Returns (window_times, window_embeddings) where window_times is a 1-D
        array of window start times and window_embeddings is (N, embed_dim).
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # cv2.CAP_PROP_FRAME_COUNT can over-report for some codecs.
        # Validate by checking the reported last frame is readable;
        # if not, binary-search for the actual last readable frame.
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ok, _ = cap.read()
            if not ok:
                lo, hi = 0, total_frames - 2
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                    ok, _ = cap.read()
                    if ok:
                        lo = mid
                    else:
                        hi = mid - 1
                actual = lo + 1
                log.debug(
                    "Frame count corrected for %s: cv2 reported %d, "
                    "actual readable %d",
                    video_path, total_frames, actual,
                )
                total_frames = actual

        duration_sec = total_frames / fps
        cap.release()

        num_frames = embedding_backend.num_frames
        window_starts = np.arange(0, duration_sec, sample_interval)
        window_embeddings = self._embed_windows_batched(
            video_path,
            window_starts,
            duration_sec,
            sample_interval,
            num_frames,
            embedding_backend,
            batch_size=WINDOW_EMBED_BATCH_SIZE,
        )

        return window_starts, window_embeddings

    def get_window_embeddings(
        self,
        video_path,
        embedding_backend,
        sample_interval=1.0,
        fps=None,
        total_frames=None,
        require_cache=False,
    ):
        """Return cached or newly computed window embeddings for a video.

        When require_cache=True, a cache miss raises WindowCacheMissError
        instead of computing embeddings (useful when the embedding backend
        is on CPU and we want to avoid slow on-the-fly inference).
        """
        cached = self.window_embedding_cache.load(
            video_path, embedding_backend, sample_interval
        )
        if cached is not None:
            log.info("Window embedding cache hit for %s", video_path)
            return {
                "fps": float(cached["fps"]),
                "total_frames": int(cached["total_frames"]),
                "window_times": cached["window_times"],
                "window_embeddings": cached["window_embeddings"],
                "cache_hit": True,
            }

        if require_cache:
            raise WindowCacheMissError(
                f"Window embedding cache miss for {video_path} "
                f"(require_cache=True)"
            )

        if fps is None or total_frames is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        window_starts, window_embeddings = self._compute_window_embeddings(
            video_path,
            embedding_backend,
            sample_interval=sample_interval,
        )
        log.info(
            "Window embedding cache miss for %s (%d windows, batch_size=%d)",
            video_path,
            len(window_starts),
            WINDOW_EMBED_BATCH_SIZE,
        )
        self.window_embedding_cache.save(
            video_path,
            embedding_backend,
            sample_interval,
            fps=fps,
            total_frames=total_frames,
            window_times=window_starts,
            window_embeddings=window_embeddings,
        )
        return {
            "fps": float(fps),
            "total_frames": int(total_frames),
            "window_times": window_starts,
            "window_embeddings": window_embeddings,
            "cache_hit": False,
        }

    def _embed_windows_batched(self, video_path, window_starts, duration_sec,
                               sample_interval, num_frames, embedding_backend,
                               batch_size=WINDOW_EMBED_BATCH_SIZE):
        """Embed windows in bounded batches to avoid large frame spikes."""
        batch_size = max(int(batch_size), 1)
        all_embeddings = []
        for start_idx in range(0, len(window_starts), batch_size):
            batch_starts = window_starts[start_idx:start_idx + batch_size]
            batch_frames = []
            for t in batch_starts:
                end_sec = min(float(t) + sample_interval, duration_sec)
                batch_frames.append(
                    self._load_raw_frames(video_path, float(t), end_sec, num_frames)
                )
            batch_embeddings = embedding_backend.embed_video_batch(
                batch_frames,
                batch_size=batch_size,
            )
            all_embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))
        return np.concatenate(all_embeddings, axis=0)

    def _load_frames_raw(self, video_path, start_frame, num_frames, step=1):
        """Load frames as RGB uint8 arrays, sampling every `step`-th source frame.

        For step=1, loads consecutive frames. For step=2, loads every other frame
        (e.g. 30fps source → 15fps sample).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame + i * step))
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_raw_frames(self, video_path, start_sec, end_sec, num_frames):
        """Load num_frames raw RGB uint8 frames uniformly from [start_sec, end_sec].

        Reads a contiguous span once and sub-samples, instead of seeking for
        every sampled index.  No resizing or normalization is applied -- the
        embedding backend handles that.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f = int(start_sec * fps)
        end_f = int(end_sec * fps)
        if end_f <= start_f:
            end_f = start_f + 1

        if total_frames > 0:
            start_f = min(max(0, start_f), total_frames - 1)
            end_f = min(max(start_f + 1, end_f), total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        raw_frames = []
        for _ in range(end_f - start_f):
            ok, frame = cap.read()
            if not ok:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(raw_frames) == 0:
            ffmpeg_frames = self._load_raw_frames_ffmpeg(
                video_path,
                start_sec,
                end_sec,
                num_frames,
            )
            if ffmpeg_frames is not None:
                return ffmpeg_frames

            # Some videos expose a tiny trailing interval that rounds to a very
            # narrow decode window near EOF. Expand the read region slightly and
            # try again before giving up on the whole video.
            widened_frames = self._load_raw_frames_widened(
                video_path,
                start_sec,
                end_sec,
                num_frames,
                fps=fps,
                total_frames=total_frames,
            )
            if widened_frames is not None:
                return widened_frames

            # Some files report a valid trailing duration/frame count but the
            # last frame or two cannot actually be decoded. Fall back to the
            # nearest readable frame instead of failing the whole video.
            fallback_cap = cv2.VideoCapture(video_path)
            fallback_start = min(start_f, max(total_frames - 1, 0))
            fallback_stop = max(-1, fallback_start - max(32, num_frames * 8))
            for frame_idx in range(fallback_start, fallback_stop, -1):
                fallback_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = fallback_cap.read()
                if ok:
                    fallback_cap.release()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return [rgb] * num_frames
            fallback_cap.release()
            raise RuntimeError(
                f"Could not read any frames from {video_path} "
                f"[{start_sec:.2f}s - {end_sec:.2f}s]"
            )

        sample_idx = np.linspace(0, len(raw_frames) - 1, num_frames).astype(int)
        return [raw_frames[i] for i in sample_idx]

    def _load_raw_frames_widened(
        self,
        video_path,
        start_sec,
        end_sec,
        num_frames,
        fps,
        total_frames,
    ):
        """Retry decode with a slightly widened window near EOF."""
        if fps <= 0 or total_frames <= 0:
            return None

        duration_sec = total_frames / fps
        window_sec = max(float(end_sec) - float(start_sec), 0.0)
        target_sec = max(
            window_sec,
            float(num_frames) / fps,
            2.0,
        )
        start_sec = max(0.0, min(float(start_sec), duration_sec))
        end_sec = min(duration_sec, max(float(end_sec), start_sec + 1e-3))
        widened_end = min(duration_sec, end_sec)
        widened_start = max(0.0, widened_end - target_sec)

        start_f = int(np.floor(widened_start * fps))
        end_f = int(np.ceil(widened_end * fps))
        start_f = min(max(0, start_f), total_frames - 1)
        end_f = min(max(start_f + 1, end_f), total_frames)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        raw_frames = []
        for _ in range(end_f - start_f):
            ok, frame = cap.read()
            if not ok:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not raw_frames:
            return self._load_raw_frames_ffmpeg(
                video_path,
                widened_start,
                widened_end,
                num_frames,
            )

        sample_idx = np.linspace(0, len(raw_frames) - 1, num_frames).astype(int)
        return [raw_frames[i] for i in sample_idx]

    def _load_raw_frames_ffmpeg(self, video_path, start_sec, end_sec, num_frames):
        """Last-resort ffmpeg decode for windows OpenCV fails to read."""
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                video_path,
            ]
            probe_res = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            probe = json.loads(probe_res.stdout)
            stream = probe["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])

            duration = max(float(end_sec) - float(start_sec), 1e-6)
            sample_fps = max(float(num_frames) / duration, 1.0)
            decode_cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{float(start_sec):.6f}",
                "-to",
                f"{float(end_sec):.6f}",
                "-i",
                video_path,
                "-vf",
                f"fps={sample_fps:.6f}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            decode_res = subprocess.run(
                decode_cmd,
                capture_output=True,
                check=True,
            )
        except Exception:
            return None

        frame_size = width * height * 3
        if frame_size <= 0:
            return None

        raw = decode_res.stdout
        frame_count = len(raw) // frame_size
        if frame_count <= 0:
            return None

        frames = np.frombuffer(
            raw[: frame_count * frame_size],
            dtype=np.uint8,
        ).reshape(frame_count, height, width, 3)
        sample_idx = np.linspace(0, frame_count - 1, num_frames).astype(int)
        return [frames[i].copy() for i in sample_idx]
