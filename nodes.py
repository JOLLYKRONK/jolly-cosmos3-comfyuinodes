"""ComfyUI nodes for NVIDIA Cosmos3 — spec-compliant implementation.

Implements the official Cosmos3-Nano inference path from the model card:

  1. Cosmos3StructuredPrompt  — LLM-based prompt upsampler producing the full
     24+ field structured JSON schema (subjects, lighting, cinematography,
     actions, segments, temporal_caption, etc.) per the official t2v_i2v_json_schema.
  2. Cosmos3ModelLoader        — Loads Cosmos3OmniPipeline with UniPC(flow_shift=10.0).
  3. Cosmos3TextToImage        — T2I generation (num_frames=1).
  4. Cosmos3TextToVideo        — T2V generation with optional sound.
  5. Cosmos3ImageToVideo       — I2V generation with optional sound.

The pipeline routes the prompt through diffusers' Qwen2 chat template,
joins text tokens with vision tokens in a unified sequence with 3D mRoPE,
and applies duration/resolution metadata templates — matching training exactly.
"""

import inspect
import json
import os
import sys
import time

import numpy as np
import requests
import torch

from .utils import (
    ASPECT_RATIOS,
    DEFAULT_ASPECT,
    DEFAULT_TIER,
    RESOLUTION_TIERS,
    audio_to_comfy_audio,
    comfy_image_to_b64_png,
    comfy_image_to_pil,
    extract_json,
    frames_to_comfy_image,
    load_cosmos3_pipeline,
    resolution_for,
    _extract_audio,
    _extract_visual,
    round_to_16,
)

CATEGORY = "Cosmos3"
_SEED_MAX = 0xFFFFFFFFFFFFFFFF

# ---------------------------------------------------------------------------
# Structured prompt defaults
# ---------------------------------------------------------------------------

_NEGATIVE_PROMPT = json.dumps(
    {
        "subjects": [
            {
                "description": "Blurry, poorly defined subjects with inconsistent shapes, distorted features, unrealistic proportions.",
                "appearance_details": "Compression artifacts, muddy textures, color bleeding, unnatural surfaces.",
                "action": "Incoherent motion with frame-to-frame discontinuities.",
                "state_changes": "Abrupt jarring transitions. Colors shift without motivation.",
                "expression": "Frozen, uncanny valley expressions.",
            }
        ],
        "background_setting": "Poorly rendered flat background with visible seams, flickering elements.",
        "lighting": {
            "conditions": "Harsh flat lighting with no natural variation.",
            "direction": "Inconsistent light sources with contradictory shadows.",
            "shadows": "Hard-edged unrealistic shadows that pop between frames.",
        },
        "aesthetics": {
            "composition": "Cluttered, poorly framed with no focal point.",
            "color_scheme": "Oversaturated garish clashing colors. Visible banding.",
        },
        "cinematography": {
            "camera_motion": "Extremely shaky unstable camera with jerky motion.",
            "depth_of_field": "Uniform focus creating a flat appearance.",
        },
        "style_medium": "Low quality compressed digital video",
        "artistic_style": "Amateur, unpolished, inconsistent",
    },
    ensure_ascii=False,
)

# T2V system prompt — matches the official t2v_i2v_video_json_schema fields
_T2V_SYSTEM = r"""You are an expert prompt engineer for NVIDIA Cosmos3-Nano.

Convert the user's plain-language video description into this EXACT structured JSON schema.
Return ONLY the JSON object — no markdown, no backticks, no explanation.

Schema (ALL fields required, use "" for irrelevant ones):
{{
  "subjects": [
    {{
      "description": "Detailed visual description — species, exact color, shape, markings, facial features",
      "appearance_details": "Texture, material, fine visual detail",
      "relationship": "Relation to other subjects or environment",
      "location": "Frame position (e.g. 'center foreground', 'left')",
      "relative_size": "'Large', 'Medium', or 'Small within frame'",
      "orientation": "Facing direction relative to camera",
      "pose": "Starting body/physical pose",
      "action": "What the subject does — motion and behavior only",
      "state_changes": "How subject transforms over video duration",
      "clothing": "",
      "expression": "Facial expression and how it changes",
      "gender": "",
      "age": "",
      "skin_tone_and_texture": "",
      "facial_features": "",
      "number_of_subjects": 1,
      "number_of_arms": 0,
      "number_of_legs": 0
    }}
  ],
  "background_setting": "Detailed environment description",
  "lighting": {{
    "conditions": "Lighting quality (e.g. 'Bright daylight', 'Overcast')",
    "direction": "Where main light comes from",
    "shadows": "Shadow quality and behavior",
    "illumination_effect": "How lighting affects mood"
  }},
  "aesthetics": {{
    "composition": "Framing and compositional choices",
    "color_scheme": "Dominant colors and palette",
    "mood_atmosphere": "Emotional tone",
    "patterns": "Notable repeating visual patterns"
  }},
  "cinematography": {{
    "camera_motion": "Camera movement (e.g. 'Static', 'Pan left', 'Tracking shot')",
    "framing": "Shot type (e.g. 'Close-up', 'Medium shot', 'Wide shot')",
    "camera_angle": "Angle (e.g. 'Eye-level', 'Low angle', 'High angle')",
    "depth_of_field": "'Shallow', 'Deep', or 'Uniform'",
    "focus": "What is in sharp focus",
    "lens_focal_length": "Descriptive focal length"
  }},
  "style_medium": "Rendering medium (e.g. 'Live-action video', 'Animation', 'CGI')",
  "artistic_style": "Genre or approach (e.g. 'realistic', 'cinematic')",
  "context": "Scene context or use case",
  "actions": [
    {{ "time": "0:00-0:02", "description": "What happens in this window" }},
    {{ "time": "0:02-0:05", "description": "Continuing action" }},
    {{ "time": "0:05-0:08", "description": "Final action" }}
  ],
  "text_and_signage_elements": [],
  "segments": [
    {{
      "segment_index": 0,
      "time_range": "0:00-0:02",
      "description": "Visual description of segment",
      "key_changes": "Notable changes within segment",
      "camera": "Camera behavior in segment"
    }}
  ],
  "transitions": [],
  "temporal_caption": "SECOND-BY-SECOND narrative of the entire video. MOST IMPORTANT FIELD. Describe every visual change, motion, transition in prose.",
  "audio_description": "Soundtrack: speech, music, ambient sounds, effects",
  "resolution": {{ "H": {H}, "W": {W} }},
  "aspect_ratio": "{aspect}",
  "duration": "{duration}s",
  "fps": {fps}
}}

RULES:
1. `temporal_caption` MUST be a detailed paragraph, not a sentence.
2. Each subject needs vivid specific visual detail.
3. `actions` must cover full duration with 2-4 timed windows.
4. `segments` must match `actions` time windows.
5. Leave irrelevant fields as "".
6. `state_changes` is crucial — describe HOW the subject changes over time.
7. `resolution.H` = {H}, `resolution.W` = {W} (hardcoded from parameters).
8. `aspect_ratio` = "{aspect}", `duration` = "{duration}s", `fps` = {fps} (hardcoded).
""".strip()


_I2V_SYSTEM = r"""You are an expert prompt engineer for NVIDIA Cosmos3-Nano, IMAGE-TO-VIDEO mode.

Frame 0 is anchored to an input image. Your prompt must describe HOW the scene evolves
into motion from that still frame. Return ONLY the JSON object.

I2V SPECIFIC GUIDANCE:
- `temporal_caption` MUST start by describing the input image's scene, then narrate motion unfolding.
- `state_changes` describes the TRANSITION from static (frame 0) to motion.
- First action window should be 0:00-0:01 capturing the initial "coming alive" moment.
- Camera motion describes what happens AFTER frame 0.

Schema:
{{
  "subjects": [
    {{
      "description": "Subject as it appears in input image",
      "appearance_details": "Texture and detail from the image",
      "relationship": "Relation to other subjects or environment",
      "location": "Frame position",
      "relative_size": "'Large', 'Medium', or 'Small within frame'",
      "orientation": "Facing direction",
      "pose": "Starting pose in input image",
      "action": "What subject begins doing as video plays",
      "state_changes": "Transition from static frame 0 into motion",
      "clothing": "",
      "expression": "Expression and how it changes from input image",
      "gender": "",
      "age": "",
      "skin_tone_and_texture": "",
      "facial_features": "",
      "number_of_subjects": 1,
      "number_of_arms": 0,
      "number_of_legs": 0
    }}
  ],
  "background_setting": "Environment as seen in input image",
  "lighting": {{
    "conditions": "Lighting quality",
    "direction": "Main light direction",
    "shadows": "Shadow quality",
    "illumination_effect": "How lighting affects mood"
  }},
  "aesthetics": {{
    "composition": "Framing and composition",
    "color_scheme": "Dominant colors",
    "mood_atmosphere": "Emotional tone",
    "patterns": "Notable repeating patterns"
  }},
  "cinematography": {{
    "camera_motion": "Camera movement starting from input frame",
    "framing": "Shot type",
    "camera_angle": "Camera angle",
    "depth_of_field": "'Shallow', 'Deep', or 'Uniform'",
    "focus": "What is in focus",
    "lens_focal_length": "Focal length"
  }},
  "style_medium": "Rendering medium",
  "artistic_style": "Visual aesthetic",
  "context": "What this video represents",
  "actions": [
    {{ "time": "0:00-0:01", "description": "Initial subtle movement as image comes alive" }},
    {{ "time": "0:01-0:04", "description": "Motion builds" }},
    {{ "time": "0:04-0:08", "description": "Full motion state" }}
  ],
  "text_and_signage_elements": [],
  "segments": [
    {{
      "segment_index": 0,
      "time_range": "0:00-0:01",
      "description": "Input image shows subtle motion",
      "key_changes": "Initial movement starts",
      "camera": "Camera begins moving from static frame"
    }},
    {{
      "segment_index": 1,
      "time_range": "0:01-0:08",
      "description": "Motion continues and develops",
      "key_changes": "Main action unfolds",
      "camera": "Camera motion continues"
    }}
  ],
  "transitions": [],
  "temporal_caption": "Start with input image description, then narrate scene coming to life frame by frame. MOST IMPORTANT.",
  "audio_description": "Soundtrack description",
  "resolution": {{ "H": {H}, "W": {W} }},
  "aspect_ratio": "{aspect}",
  "duration": "{duration}s",
  "fps": {fps}
}}

RULES:
1. `temporal_caption` MUST start with input image description, then narrate video.
2. Subjects described as in input image, with `action`/`state_changes` for transition to motion.
3. Leave irrelevant fields as "".
""".strip()


# ---------------------------------------------------------------------------
# Signature-safe pipe call helper
# ---------------------------------------------------------------------------

def _filter_kwargs(fn, kwargs):
    """Drop kwargs the callable doesn't accept. Self-heals against diffusers API drift."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


# ---------------------------------------------------------------------------
# LLM prompt architect helpers
# ---------------------------------------------------------------------------

def _call_llm_api(system, user, api_url, api_endpoint, model_name,
                  temperature, max_tokens, timeout, image_b64=None):
    """Call external LLM and return parsed JSON dict."""
    base = api_url.rstrip("/")
    is_ollama = "ollama" in api_endpoint.lower()

    send_image = image_b64 is not None and is_ollama

    if send_image:
        user_content = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        user_content = user

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    if is_ollama:
        payload = {
            "model": model_name or "qwen3.6-27b",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "format": "json",
        }
        if send_image:
            payload["images"] = [image_b64]
        url = f"{base}/api/chat"
    else:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model_name:
            payload["model"] = model_name
        url = f"{base}/v1/chat/completions"

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if is_ollama:
        text = data.get("message", {}).get("content", "")
    else:
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if not text:
        raise ValueError("LLM returned empty response.")

    return json.loads(extract_json(text))


# ---------------------------------------------------------------------------
# Node 1: Cosmos3 Structured Prompt (integrated LLM architect)
# ---------------------------------------------------------------------------

class JollyCosmos3StructuredPrompt:
    """Generate a spec-compliant structured JSON prompt for Cosmos3 via local LLM.

    Produces the full 24+ field schema matching the official t2v_i2v_video_json_schema
    from the Cosmos3 model card. Outputs `prompt` (json.dumps of structured dict) and
    `negative_prompt` for downstream generation nodes.

    Connect the `prompt` and `negative_prompt` outputs to Cosmos3TextToVideo or
    Cosmos3ImageToVideo nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ("t2v", "i2v", "t2i"),
                    {"default": "t2v", "tooltip": "Generation mode."},
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": (
                            "Describe the video in plain English.\n\n"
                            "T2V: 'A Pixar-style red tomato dancing on a disco floor'\n"
                            "I2V: 'The tomato starts dancing, camera orbits around it'\n"
                            "T2I: 'A photorealistic close-up of a mechanical watch'"
                        ),
                    },
                ),
                "api_url": (
                    "STRING",
                    {"default": "http://127.0.0.1:8890"},
                ),
                "api_endpoint": (
                    ("openai /v1/chat/completions", "ollama /api/chat"),
                    {"default": "openai /v1/chat/completions"},
                ),
                "model_name": (
                    "STRING",
                    {"default": "", "placeholder": "E.g. qwen3.6-27b-vision"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 512, "max": 16384, "step": 128},
                ),
                "resolution_preset": (
                    ("custom",) + RESOLUTION_TIERS,
                    {"default": DEFAULT_TIER},
                ),
                "aspect_ratio": (ASPECT_RATIOS, {"default": DEFAULT_ASPECT}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 7.9, "min": 0.2, "max": 30.0, "step": 0.1},
                ),
                "fps": ("INT", {"default": 24, "min": 10, "max": 30}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "I2V only: source image for vision-capable LLM."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, mode, description, api_url, api_endpoint, model_name,
                temperature, max_tokens, resolution_preset, aspect_ratio,
                width, height, duration_seconds, fps, timeout, image=None):
        if not description or not description.strip():
            raise ValueError("Description is required.")

        w, h = resolution_for(resolution_preset, aspect_ratio) if resolution_preset != "custom" else (round_to_16(width), round_to_16(height))

        system_prompt = ""
        if mode == "i2v":
            system_prompt = _I2V_SYSTEM
        elif mode == "t2i":
            system_prompt = _T2V_SYSTEM.replace(
                '"duration": "{duration}s"',
                f'"duration": "0s"'
            )
        else:
            system_prompt = _T2V_SYSTEM

        system_prompt = system_prompt.format(
            H=h, W=w, aspect=aspect_ratio.replace(":", ","),
            duration=f"{duration_seconds:.1f}", fps=fps
        )

        image_b64 = None
        if mode == "i2v" and image is not None:
            image_b64 = comfy_image_to_b64_png(image)

        user_text = (
            f"I2V MODE — video starts from input image (frame 0). Describe motion:\n\n{description.strip()}"
            if mode == "i2v"
            else f"T2I MODE:\n\n{description.strip()}" if mode == "t2i"
            else f"T2V MODE:\n\n{description.strip()}"
        )

        try:
            prompt_data = _call_llm_api(
                system_prompt, user_text, api_url, api_endpoint, model_name,
                temperature, max_tokens, timeout, image_b64=image_b64,
            )
        except Exception as e:
            print(f"[cosmos3-official] LLM call failed: {e}. Using minimal fallback.")
            prompt_data = {
                "subjects": [{"description": description.strip()}],
                "background_setting": "",
                "lighting": {"conditions": "", "direction": "", "shadows": "", "illumination_effect": ""},
                "aesthetics": {"composition": "", "color_scheme": "", "mood_atmosphere": "", "patterns": ""},
                "cinematography": {"camera_motion": "", "framing": "", "camera_angle": "", "depth_of_field": "", "focus": "", "lens_focal_length": ""},
                "style_medium": "",
                "artistic_style": "",
                "context": "",
                "actions": [{"time": f"0:00-{duration_seconds:.0f}s", "description": description.strip()}],
                "text_and_signage_elements": [],
                "segments": [],
                "transitions": [],
                "temporal_caption": description.strip(),
                "audio_description": "",
                "resolution": {"H": h, "W": w},
                "aspect_ratio": aspect_ratio.replace(":", ","),
                "duration": f"{duration_seconds:.1f}s",
                "fps": fps,
            }

        # Ensure required fields
        prompt_data.setdefault("fps", fps)
        prompt_data.setdefault("duration", f"{duration_seconds:.1f}s")
        prompt_data.setdefault("resolution", {"H": h, "W": w})
        prompt_data.setdefault("aspect_ratio", aspect_ratio.replace(":", ","))

        prompt_out = json.dumps(prompt_data, ensure_ascii=False)
        print(f"[cosmos3-official] {mode.upper()} structured prompt built ({len(prompt_out)} chars).")
        return (prompt_out, _NEGATIVE_PROMPT)


# ---------------------------------------------------------------------------
# Node 2: Cosmos3 Model Loader
# ---------------------------------------------------------------------------

class JollyCosmos3ModelLoader:
    """Load a Cosmos 3 generator checkpoint (Cosmos3OmniPipeline).

    Loads from ComfyUI/models/diffusers/<model_name>/ by default.
    Also checks ComfyUI/models/Cosmos3/<model_name>/ and HF Hub cache.

    Uses UniPCMultistepScheduler(flow_shift=10.0) per official spec.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {
                        "default": "cosmos3_nano",
                        "placeholder": "cosmos3_nano, or path to model dir, or HF repo ID (nvidia/Cosmos3-Nano)",
                    },
                ),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
                "device": (
                    ["auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    {"default": "auto"},
                ),
                "disable_guardrails": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Skip Cosmos safety guardrails."},
                ),
            },
        }

    RETURN_TYPES = ("COSMOS3_PIPE",)
    RETURN_NAMES = ("cosmos3_pipe",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_name, precision, device, disable_guardrails):
        dev = None
        if device != "auto":
            dev = torch.device(device)

        pipe_data = load_cosmos3_pipeline(
            model_id=model_name,
            precision=precision,
            device=dev,
            disable_guardrails=disable_guardrails,
        )

        print(f"[cosmos3-official] Pipeline loaded: device={pipe_data['device']}, "
              f"dtype={pipe_data['dtype']}, sound={pipe_data['supports_sound']}")
        return (pipe_data,)


# ---------------------------------------------------------------------------
# Node 3: Cosmos3 Text-to-Image
# ---------------------------------------------------------------------------

class JollyCosmos3TextToImage:
    """Generate image from prompt using Cosmos3OmniPipeline.

    Passes prompt through the official Qwen2 chat template with system prompt,
    duration/resolution metadata templates, and eos + <end_of_256> special tokens.
    Text tokens are joined with vision tokens in a unified 3D mRoPE sequence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, prompt, negative_prompt, width, height,
                 steps, guidance_scale, seed):
        pipe = cosmos3_pipe["pipe"]
        device = cosmos3_pipe["device"]

        try:
            pbar = __import__("comfy.utils").utils.ProgressBar(steps)
        except Exception:
            pbar = None

        def _cb(p, idx, ts, kw):
            try:
                __import__("comfy.model_management").model_management.throw_exception_if_processing_interrupted()
            except Exception:
                pass
            if pbar:
                pbar.update_absolute(idx + 1, steps)
            return kw

        generator = torch.Generator(
            device="cuda" if device.type == "cuda" else "cpu"
        ).manual_seed(int(seed))

        candidate = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "height": int(height),
            "width": int(width),
            "num_frames": 1,
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _cb,
            "enable_sound": False,
            "enable_safety_check": False,
        }
        result = pipe(**_filter_kwargs(pipe.__call__, candidate))

        visual = _extract_visual(result, want_image=True)
        image_out = frames_to_comfy_image(visual)

        if image_out is None or image_out.numel() == 0:
            image_out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (image_out,)


# ---------------------------------------------------------------------------
# Node 4: Cosmos3 Text-to-Video
# ---------------------------------------------------------------------------

class JollyCosmos3TextToVideo:
    """Generate video (with optional sound) from prompt using Cosmos3OmniPipeline.

    Full spec compliance:
    - Prompt string (structured JSON from Cosmos3StructuredPrompt or plain text)
      goes through Qwen2 chat template with system prompt
    - Duration template appended: "The video is X.X seconds long and is of Y FPS."
    - Resolution template appended: "This video is of HxW resolution."
    - Special tokens: eos_token_id + <end_of_256> appended
    - Text + vision tokens joined in unified 3D mRoPE sequence
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 189, "min": 5, "max": 400,
                    "tooltip": "5-400 frames. 189 @ 24fps = ~7.9s. Default per spec."}),
                "fps": ("INT", {"default": 24, "min": 10, "max": 30}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "generate_sound": ("BOOLEAN", {"default": False,
                    "tooltip": "Generate synchronized audio (sound-capable checkpoints only)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, prompt, negative_prompt, width, height,
                 num_frames, fps, steps, guidance_scale, seed, generate_sound):
        pipe = cosmos3_pipe["pipe"]
        device = cosmos3_pipe["device"]
        want_sound = generate_sound and cosmos3_pipe.get("supports_sound", False)

        try:
            pbar = __import__("comfy.utils").utils.ProgressBar(steps)
        except Exception:
            pbar = None

        def _cb(p, idx, ts, kw):
            try:
                __import__("comfy.model_management").model_management.throw_exception_if_processing_interrupted()
            except Exception:
                pass
            if pbar:
                pbar.update_absolute(idx + 1, steps)
            return kw

        generator = torch.Generator(
            device="cuda" if device.type == "cuda" else "cpu"
        ).manual_seed(int(seed))

        candidate = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "fps": float(fps),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _cb,
            "enable_sound": want_sound,
            "enable_safety_check": False,
        }
        result = pipe(**_filter_kwargs(pipe.__call__, candidate))

        visual = _extract_visual(result, want_image=False)
        frames_out = frames_to_comfy_image(visual)

        audio_out = None
        if want_sound:
            audio_out = audio_to_comfy_audio(_extract_audio(result))

        if frames_out is None or frames_out.numel() == 0:
            frames_out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (frames_out, audio_out)


# ---------------------------------------------------------------------------
# Node 5: Cosmos3 Image-to-Video
# ---------------------------------------------------------------------------

class JollyCosmos3ImageToVideo:
    """Generate video (with optional sound) conditioned on input image + prompt.

    Full spec compliance — same prompt routing as T2V, plus image conditioning:
    - Frame 0 anchored to input image via VAE encode
    - Vision conditioning mask applied (frame 0 = condition, rest = noisy)
    - Prompt augmented with duration/resolution templates and chat formatting
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 189, "min": 5, "max": 400}),
                "fps": ("INT", {"default": 24, "min": 10, "max": 30}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "generate_sound": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, image, prompt, negative_prompt, width, height,
                 num_frames, fps, steps, guidance_scale, seed, generate_sound):
        pipe = cosmos3_pipe["pipe"]
        device = cosmos3_pipe["device"]
        want_sound = generate_sound and cosmos3_pipe.get("supports_sound", False)

        try:
            pbar = __import__("comfy.utils").utils.ProgressBar(steps)
        except Exception:
            pbar = None

        def _cb(p, idx, ts, kw):
            try:
                __import__("comfy.model_management").model_management.throw_exception_if_processing_interrupted()
            except Exception:
                pass
            if pbar:
                pbar.update_absolute(idx + 1, steps)
            return kw

        generator = torch.Generator(
            device="cuda" if device.type == "cuda" else "cpu"
        ).manual_seed(int(seed))

        pil_image = comfy_image_to_pil(image)

        candidate = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "image": pil_image,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "fps": float(fps),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _cb,
            "enable_sound": want_sound,
            "enable_safety_check": False,
        }
        result = pipe(**_filter_kwargs(pipe.__call__, candidate))

        visual = _extract_visual(result, want_image=False)
        frames_out = frames_to_comfy_image(visual)

        audio_out = None
        if want_sound:
            audio_out = audio_to_comfy_audio(_extract_audio(result))

        if frames_out is None or frames_out.numel() == 0:
            frames_out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (frames_out, audio_out)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "JollyCosmos3StructuredPrompt": JollyCosmos3StructuredPrompt,
    "JollyCosmos3ModelLoader": JollyCosmos3ModelLoader,
    "JollyCosmos3TextToImage": JollyCosmos3TextToImage,
    "JollyCosmos3TextToVideo": JollyCosmos3TextToVideo,
    "JollyCosmos3ImageToVideo": JollyCosmos3ImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JollyCosmos3StructuredPrompt": "Jolly Cosmos3 Structured Prompt",
    "JollyCosmos3ModelLoader": "Jolly Cosmos3 Model Loader",
    "JollyCosmos3TextToImage": "Jolly Cosmos3 Text-to-Image",
    "JollyCosmos3TextToVideo": "Jolly Cosmos3 Text-to-Video",
    "JollyCosmos3ImageToVideo": "Jolly Cosmos3 Image-to-Video",
}
