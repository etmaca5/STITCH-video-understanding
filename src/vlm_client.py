"""OpenAI-compatible VLM client for text, image, and video-frame inference."""

import base64
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
IMAGE_INPUT_MODE = "image_url"
VIDEO_INPUT_MODE = "video_url"
SUPPORTED_INPUT_MODES = {IMAGE_INPUT_MODE, VIDEO_INPUT_MODE}


def _frame_to_base64(
    frame: np.ndarray, quality: int = 85, max_image_dim: int | None = None
) -> str:
    """Encode an RGB numpy frame as a base64 JPEG data-URL string.

    If ``max_image_dim`` is set, the frame is downscaled (preserving aspect)
    so its longest side equals ``max_image_dim``. Required for LLaVA-OneVision
    on vLLM, which AnyRes-patches large images into thousands of vision tokens
    and overflows the 32k context with even 8 HD frames.
    """
    if max_image_dim is not None:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > max_image_dim:
            scale = max_image_dim / longest
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _frame_to_jpeg_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB numpy frame as raw base64 JPEG bytes."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class VLMClient:
    """Thin wrapper around the OpenRouter chat-completions endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        site_name: str | None = None,
        max_images_per_request: int | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 2,
        max_image_dim: int | None = None,
        input_mode: str = IMAGE_INPUT_MODE,
        send_video_metadata: bool = False,
    ):
        input_mode = str(input_mode).lower()
        if input_mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(
                f"Unsupported VLM input_mode={input_mode!r}. "
                f"Expected one of {sorted(SUPPORTED_INPUT_MODES)}."
            )
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not found. "
                "Set it in .env or as an environment variable."
            )
        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-OpenRouter-Title"] = site_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers or None,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_images_per_request = max_images_per_request
        self.max_image_dim = max_image_dim
        self.input_mode = input_mode
        self.send_video_metadata = bool(send_video_metadata)

    @classmethod
    def from_config(cls, cfg):
        """Build a client from a Hydra/OmegaConf config object."""
        return cls(
            model=cfg.model,
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 256),
            base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
            site_url=cfg.get("site_url"),
            site_name=cfg.get("site_name"),
            max_images_per_request=cfg.get("max_images_per_request"),
            api_key_env=cfg.get("api_key_env", "OPENROUTER_API_KEY"),
            timeout=cfg.get("timeout", 60.0),
            max_retries=cfg.get("max_retries", 2),
            max_image_dim=cfg.get("max_image_dim"),
            input_mode=cfg.get("input_mode", IMAGE_INPUT_MODE),
            send_video_metadata=cfg.get("send_video_metadata", False),
        )

    def query(
        self,
        text: str,
        images: list[np.ndarray] | None = None,
        system_prompt: str | None = None,
        video_metadata: dict | None = None,
        video_frame_indices: list[int] | None = None,
    ) -> str:
        """Send a text (+optional images) query and return the response text.

        Images are RGB numpy arrays (H, W, 3). In ``image_url`` mode they are
        appended as separate image inputs. In ``video_url`` mode they are packed
        as a pre-extracted JPEG frame sequence so vLLM routes them as video.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if (
            self.max_images_per_request is not None
            and images is not None
            and len(images) > self.max_images_per_request
        ):
            raise ValueError(
                f"Request has {len(images)} images but the configured limit is "
                f"{self.max_images_per_request}. Reduce chunks/frames or change provider."
            )
        content = [{"type": "text", "text": text}]
        extra_body = None
        if images:
            if self.input_mode == VIDEO_INPUT_MODE:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": self._frames_to_video_url(images)},
                })
                if self.send_video_metadata:
                    extra_body = self._build_video_extra_body(
                        video_metadata=video_metadata,
                        video_frame_indices=video_frame_indices,
                    )
            else:
                for frame in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": _frame_to_base64(
                                frame,
                                max_image_dim=self.max_image_dim,
                            )
                        },
                    })

        messages.append({"role": "user", "content": content})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**kwargs)
        if getattr(response, "error", None):
            raise RuntimeError(response.error["message"])
        if not response.choices:
            raise RuntimeError("OpenRouter returned no choices and no explicit error.")
        return response.choices[0].message.content

    def _frames_to_video_url(self, frames: list[np.ndarray]) -> str:
        encoded_frames = [_frame_to_jpeg_base64(frame) for frame in frames]
        return f"data:video/jpeg;base64,{','.join(encoded_frames)}"

    def _build_video_extra_body(
        self,
        video_metadata: dict | None = None,
        video_frame_indices: list[int] | None = None,
    ) -> dict:
        video_kwargs = {"do_sample_frames": False}
        if video_metadata:
            fps = video_metadata.get("fps")
            total_frames = video_metadata.get("total_frames")
            duration = video_metadata.get("duration")
            if fps is not None:
                video_kwargs["fps"] = float(fps)
            if total_frames is not None:
                video_kwargs["total_num_frames"] = int(total_frames)
            if duration is not None:
                video_kwargs["duration"] = float(duration)
        if video_frame_indices is not None:
            video_kwargs["frames_indices"] = [int(idx) for idx in video_frame_indices]
        return {"media_io_kwargs": {"video": video_kwargs}}
