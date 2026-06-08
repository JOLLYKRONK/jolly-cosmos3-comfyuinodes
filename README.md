# cosmos3-official — Spec-Compliant ComfyUI Nodes for NVIDIA Cosmos 3

Drop-in ComfyUI nodes that implement the **official Cosmos3-Nano inference path** from the [NVIDIA model card](https://huggingface.co/nvidia/Cosmos3-Nano). Routes prompts through the correct Qwen2 chat template, unified 3D mRoPE sequences, and flow-matching scheduler — matching training exactly.

## Why This Exists

The built-in ComfyUI Cosmos nodes (`EmptyCosmosLatentVideo`, etc.) implement the **Cosmos 1 / 1.5 architecture** (T5 cross-attention + EDM sampling). Cosmos3-Nano uses a fundamentally different architecture: **Qwen2 joint-token sequences with unified mRoPE position embeddings and flow matching**. These nodes bridge that gap.

## Nodes

| Node | Description |
|------|-------------|
| **JollyCosmos3StructuredPrompt** | LLM-based prompt upsampler producing the full 24+ field structured JSON schema (subjects, lighting, cinematography, actions, segments, temporal_caption, etc.) per the official `t2v_i2v_video_json_schema` |
| **JollyCosmos3ModelLoader** | Loads `Cosmos3OmniPipeline` with `UniPCMultistepScheduler(flow_shift=10.0)` |
| **JollyCosmos3TextToImage** | T2I generation (`num_frames=1`) |
| **JollyCosmos3TextToVideo** | T2V generation with optional sound |
| **JollyCosmos3ImageToVideo** | I2V generation with optional sound |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/cosmos3-official.git
```

Restart ComfyUI. Nodes appear under the **Cosmos3** category.

## Requirements

- `diffusers` (git main): `pip install -U 'diffusers @ git+https://github.com/huggingface/diffusers.git'`
- `transformers`, `torch`, `accelerate`, `av`, `imageio`, `imageio-ffmpeg`
- Cosmos3-Nano model at `ComfyUI/models/diffusers/cosmos3_nano/`
- Local LLM for prompt architect (OpenAI-compatible endpoint, e.g., llama.cpp or Ollama)

## Workflow

```
JollyCosmos3StructuredPrompt  ──prompt────────────┐
  (LLM → 24+ field JSON)                          │
                                                  ▼
JollyCosmos3ModelLoader  ──cosmos3_pipe────────┐  JollyCosmos3TextToVideo
                                                └──►  (Qwen2 chat template
                                                     + joint sequence
                                                     + flow_shift=10.0)
                                                  │  ◄──negative_prompt
                                                  ▼
                                               frames + audio
```

## Model Download

```bash
huggingface-cli download nvidia/Cosmos3-Nano --local-dir ComfyUI/models/diffusers/cosmos3_nano
```

## Architecture Compliance

| Spec Requirement | Implementation |
|-----------------|----------------|
| Qwen2 chat template | Via diffusers `apply_chat_template()` |
| Special tokens (eos + `<\|end_of_256\|>`) | Via diffusers `tokenize_prompt()` |
| Duration/resolution templates | Via diffusers `_apply_templates()` |
| Unified 3D mRoPE sequence | Via diffusers joint-sequence packing |
| Flow matching (flow_shift=10.0) | `UniPCMultistepScheduler(flow_shift=10.0)` |
| Structured JSON prompt schema | LLM upsampler → full 24+ field JSON |

## License

MIT
