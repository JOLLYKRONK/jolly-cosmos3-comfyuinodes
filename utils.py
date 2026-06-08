"""Shared utilities for cosmos3-official ComfyUI nodes."""

import base64
import io
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Resolution helpers — match official Cosmos3 spec exactly
# ---------------------------------------------------------------------------

RESOLUTION_TIERS = ("256p", "480p", "720p")
ASPECT_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16")

DEFAULT_TIER = "480p"
DEFAULT_ASPECT = "16:9"

# Official 16:9 sizes (width, height) per tier.
_TIER_BASE_16_9 = {
    "256p": (320, 192),
    "480p": (832, 480),
    "720p": (1280, 720),
}

_ASPECT_WH = {
    "16:9": (16, 9),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "9:16": (9, 16),
}


def round_to_16(value):
    """Round to nearest multiple of 16."""
    return max(16, int(round(value / 16) * 16))


def resolution_for(tier=DEFAULT_TIER, aspect=DEFAULT_ASPECT):
    """Return (width, height) for resolution tier + aspect ratio."""
    bw, bh = _TIER_BASE_16_9.get(tier, _TIER_BASE_16_9[DEFAULT_TIER])
    if aspect == "16:9":
        return bw, bh
    aw, ah = _ASPECT_WH.get(aspect, _ASPECT_WH[DEFAULT_ASPECT])
    area = float(bw * bh)
    ratio = aw / ah
    height = math.sqrt(area / ratio)
    width = height * ratio
    return round_to_16(width), round_to_16(height)


# ---------------------------------------------------------------------------
# Frame / image conversion helpers
# ---------------------------------------------------------------------------

def comfy_image_to_pil(image_tensor):
    """First frame of ComfyUI IMAGE [B,H,W,C] float32 [0,1] -> PIL.Image."""
    img = image_tensor[0]
    arr = (img.clamp(0.0, 1.0).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def _frame_to_hwc_float(frame):
    """Normalize a frame to float [H,W,3] in [0,1]."""
    if hasattr(frame, "convert"):
        frame = np.asarray(frame.convert("RGB"))
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame)
    elif isinstance(frame, torch.Tensor):
        t = frame
    else:
        t = torch.as_tensor(np.asarray(frame))

    t = t.detach().cpu().float()

    if t.dim() == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
        t = t.permute(1, 2, 0)
    if t.dim() == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]

    if t.max() > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def frames_to_comfy_image(frames):
    """Frames (list / 4D tensor) -> ComfyUI IMAGE [T,H,W,3] float32 [0,1]."""
    if frames is None:
        return None
    if isinstance(frames, torch.Tensor) and frames.dim() == 4:
        return torch.stack([_frame_to_hwc_float(f) for f in frames], dim=0)
    out = [_frame_to_hwc_float(f) for f in frames]
    if not out:
        return None
    return torch.stack(out, dim=0).contiguous()


def _extract_visual(result, want_image):
    """Extract visual frames from pipeline result."""
    visual = None
    for attr in ("video", "videos", "frames", "images"):
        v = getattr(result, attr, None)
        if v is None and isinstance(result, dict):
            v = result.get(attr)
        if v is not None:
            visual = v
            break
    if visual is None:
        return None

    # Unpack nested batch: [[frames...]] -> [frames...]
    if isinstance(visual, (list, tuple)) and len(visual) > 0 and isinstance(visual[0], (list, tuple)):
        visual = visual[0]
    elif isinstance(visual, torch.Tensor) and visual.dim() == 5:
        visual = visual[0]

    if want_image:
        if isinstance(visual, (list, tuple)):
            return [visual[0]]
        if isinstance(visual, torch.Tensor) and visual.dim() == 4:
            return visual[0:1]
    return visual


def _extract_audio(result):
    """Extract audio waveform from pipeline result."""
    for attr in ("sound", "audio", "audios"):
        a = getattr(result, attr, None)
        if a is None and isinstance(result, dict):
            a = result.get(attr)
        if a is not None:
            if isinstance(a, (list, tuple)) and len(a) > 0:
                return a[0]
            return a
    return None


def audio_to_comfy_audio(audio, sample_rate=48000):
    """Convert pipeline audio to ComfyUI AUDIO dict."""
    if audio is None:
        return None
    if isinstance(audio, dict):
        sample_rate = int(audio.get("sample_rate", sample_rate))
        wav = audio.get("waveform", audio.get("array"))
    else:
        wav = audio
    if wav is None:
        return None
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    wav = wav.detach().cpu().float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    return {"waveform": wav.contiguous(), "sample_rate": int(sample_rate)}


# ---------------------------------------------------------------------------
# LLM communication helpers
# ---------------------------------------------------------------------------

def comfy_image_to_b64_png(image_tensor):
    """ComfyUI IMAGE -> base64 PNG string."""
    frame = image_tensor[0]
    arr = (frame.clamp(0.0, 1.0).cpu().float().numpy() * 255.0).round().astype("uint8")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text):
    """Extract valid JSON from LLM response text (handles markdown fences, etc.)."""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    for fence in ["```json\n", "```json ", "```\n", "``` "]:
        if fence in text:
            parts = text.split(fence, 1)
            if len(parts) > 1:
                text = parts[1]
            if "```" in text:
                text = text.split("```", 1)[0]
            text = text.strip()
            try:
                json.loads(text)
                return text
            except (json.JSONDecodeError, ValueError):
                pass

    # Bracket-depth fallback
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except (json.JSONDecodeError, ValueError):
                    continue
    return text


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------

def get_models_root():
    """Get ComfyUI models directory.

    Tries: 1) folder_paths.models_dir  2) absolute path from known ComfyUI root.
    """
    # Try importing ComfyUI's folder_paths (works when running inside ComfyUI)
    try:
        import folder_paths
        return folder_paths.models_dir
    except Exception:
        pass

    # Walk up from package to find models/ sibling of custom_nodes/

    # Walk up from package to find models/ sibling of custom_nodes/
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # custom_nodes/
    for _ in range(4):
        if os.path.isdir(os.path.join(pkg_root, "models")):
            return os.path.join(pkg_root, "models")
        pkg_root = os.path.dirname(pkg_root)

    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def resolve_model_dir(model_id):
    """Resolve model directory from HF repo ID or local path.

    Checks:
      1. Direct path if model_id looks like an absolute path
      2. <models>/diffusers/<last_component>/
      3. <models>/Cosmos3/<model_key>/
      4. HuggingFace Hub cache
    """
    if os.path.isabs(model_id) and os.path.isdir(model_id):
        return model_id

    # Check <models>/diffusers/ path
    last = model_id.split("/")[-1]
    for subdir in ("diffusers", "Cosmos3"):
        candidate = os.path.join(get_models_root(), subdir, last)
        if os.path.exists(os.path.join(candidate, "model_index.json")):
            return candidate
    candidate = os.path.join(get_models_root(), "diffusers", model_id)
    if os.path.exists(os.path.join(candidate, "model_index.json")):
        return candidate

    # HuggingFace Hub cache
    try:
        from huggingface_hub import snapshot_download
        cached = snapshot_download(model_id, repo_type="model")
        if os.path.exists(os.path.join(cached, "model_index.json")):
            return cached
    except Exception:
        pass

    return os.path.join(get_models_root(), "diffusers", last)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_cosmos3_pipeline(model_id, precision="bf16", device=None, disable_guardrails=True, verbose=True):
    """Load the Cosmos3OmniPipeline with flow_shift=10.0 scheduler."""
    log = lambda msg: print(f"[cosmos3-official] {msg}") if verbose else None

    model_dir = resolve_model_dir(model_id)
    if not os.path.exists(os.path.join(model_dir, "model_index.json")):
        raise FileNotFoundError(
            f"Cosmos3 model not found. Looked in:\n"
            f"  {model_dir}\n"
            f"Ensure model exists at <ComfyUI>/models/diffusers/{model_id.split('/')[-1]}/\n"
            f"or run: huggingface-cli download nvidia/Cosmos3-Nano --local-dir {model_dir}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    import diffusers
    if not hasattr(diffusers, "Cosmos3OmniPipeline"):
        raise ImportError(
            "Your diffusers version doesn't have Cosmos3OmniPipeline. Install latest:\n"
            "  pip install -U 'diffusers @ git+https://github.com/huggingface/diffusers.git'"
        )

    import importlib.util
    guardrails_available = importlib.util.find_spec("cosmos_guardrail") is not None
    enable_safety = (not disable_guardrails) and guardrails_available

    log(f"Loading Cosmos3 from {model_dir} (dtype={dtype})...")
    try:
        pipe = diffusers.Cosmos3OmniPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            enable_safety_checker=enable_safety,
        )
    except TypeError:
        pipe = diffusers.Cosmos3OmniPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
        )

    # Apply official spec scheduler: UniPC with flow_shift=10.0
    try:
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
        log("Scheduler: UniPCMultistepScheduler(flow_shift=10.0)")
    except Exception:
        log("Scheduler: using pipeline default (flow_shift not applied)")

    # Patch VAE decode to clear CUDA cache and prevent fragmentation OOM
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        orig_decode = vae.decode
        def _cached_decode(z, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return orig_decode(z, *args, **kwargs)
        vae.decode = _cached_decode

    # Move to device (quantized models may skip this)
    try:
        pipe = pipe.to(device)
    except (ValueError, RuntimeError, NotImplementedError):
        log("Pipeline placed itself on device (quantized?).")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    supports_sound = pipe.sound_tokenizer is not None
    return {
        "pipe": pipe,
        "device": device,
        "dtype": dtype,
        "supports_sound": supports_sound,
        "model_dir": model_dir,
    }
