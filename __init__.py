"""cosmos3-official — ComfyUI nodes for NVIDIA Cosmos3, spec-compliant.

Implements the official Cosmos3 inference path using diffusers'
Cosmos3OmniPipeline with Qwen2 chat template, unified 3D mRoPE sequences,
and UniPC(flow_shift=10.0) scheduler.

Nodes:
  - Cosmos3StructuredPrompt  : LLM-based prompt upsampler (24+ field JSON schema)
  - Cosmos3ModelLoader       : Load Cosmos3OmniPipeline
  - Cosmos3TextToImage       : T2I generation
  - Cosmos3TextToVideo       : T2V generation (+ optional sound)
  - Cosmos3ImageToVideo      : I2V generation (+ optional sound)
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
