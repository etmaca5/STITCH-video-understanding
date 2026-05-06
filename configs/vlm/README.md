# OpenRouter VLM Options

Tested on this repo with real image requests:

| Config | Model id | Tested image count |
|---|---|---:|
| `qwen2_5_vl_7b` | `qwen/qwen-2.5-vl-7b-instruct` | `4` |
| `qwen3_vl_8b` | `qwen/qwen3-vl-8b-instruct` | `24+` |
| `gemini_2_0_flash` | `google/gemini-2.0-flash-001` | `40+` |
| `gemini_2_5_pro` | `google/gemini-2.5-pro` | `32+` |
| `gemini_3_flash_preview` | `google/gemini-3-flash-preview` | `32+` |
| `gpt_4o` | `openai/gpt-4o` | `32+` |
| `claude_3_5_sonnet` | `anthropic/claude-3.5-sonnet` | `32+` |
| `qwen3_vl_32b` | `qwen/qwen3-vl-32b-instruct` | `24+` |

## Pricing (OpenRouter, per million tokens)

| Config | Input | Output |
|--------|-------|--------|
| `qwen2_5_vl_7b` | $0.20 | $0.20 |
| `qwen3_vl_8b` | $0.08 | $0.50 |
| `gemini_2_0_flash` | $0.10 | $0.40 |
| `qwen3_vl_32b` | $0.104 | $0.416 |
| `gemini_3_flash_preview` | $0.50 | $3.00 |
| `gemini_2_5_pro` | $1.25 / $2.50 (>200K) | $10 / $15 (>200K) |
| `gpt_4o` | $2.50 | $10 |
| `claude_3_5_sonnet` | $6 | $30 |

Pay-as-you-go adds 5.5% platform fee. Gemini 2.0 Flash deprecates June 1, 2026.

Notes:

- `qwen/qwen-2.5-vl-7b-instruct`: current OpenRouter route hard-failed above `4` images
- `qwen/qwen3-vl-8b-instruct`: accepted `12` and `24` images in direct OpenRouter probes
- `google/gemini-flash-1.5`: model page exists, but current route returned `No endpoints found`
- `google/gemini-2.5-pro`: accepted the images, but output formatting was less stable than `gpt-4o` / `gemini-2.0-flash-001`
- `anthropic/claude-3.5-sonnet`: accepted images well, but at `n=1` it added extra description instead of strictly following the short output format

Not found / not callable on the current OpenRouter route:

- `InternVL3` (`opengvlab/internvl3-2b`, `-14b`, `-78b`) -> `404`
- `InternVL3.5` -> not found
- `VideoLLaMA3` -> not found
- `LLaVA-Video` / `LLaVA-NeXT-Video` -> not found
- `Aria` -> not found
- `SmolVLM2` -> not found
- `Ovis2` -> not found
