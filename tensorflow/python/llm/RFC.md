# RFC: `tf.llm` — One-Line LLM Loading and LoRA Fine-Tuning for TensorFlow

| Field        | Value |
|--------------|-------|
| **RFC #**    | TF-RFC-0042 |
| **Author**   | TensorFlow Contributors |
| **Status**   | Proposed |
| **Created**  | 2024-06-01 |
| **Updated**  | 2024-06-01 |

---

## Objective

Add a `tf.llm` submodule that lets developers load any open-source causal
language model (Llama, Gemma, Mistral, Falcon, …) and fine-tune it with LoRA
in **three lines of code**, using a familiar Keras-style API.

```python
import tensorflow as tf

model = tf.llm.from_pretrained("meta-llama/Llama-3")
model.compile(optimizer="adam", loss="causal_lm")
model.fit(your_dataset, epochs=1)
output = model.generate("Once upon a time", max_new_tokens=100)
```

---

## Motivation

Modern ML practitioners almost never write neural network layers from scratch.
They start from a pretrained model and adapt it to their task.  The ecosystem
for this workflow is currently dominated by PyTorch + HuggingFace.  TensorFlow
has no first-party high-level API that:

1. Downloads a pretrained LLM with a single call.
2. Applies parameter-efficient fine-tuning (LoRA) automatically.
3. Exposes the familiar `compile` / `fit` / `generate` interface TF users
   already know from Keras.

As a result, even developers who prefer TensorFlow's production tooling
(TFX, TF Serving, TFLite) are forced to prototype in PyTorch and then convert.
`tf.llm` closes this gap.

---

## Proposed API

### `tf.llm.from_pretrained(model_name, **kwargs) → TFLLMModel`

Downloads (and caches) the model and its tokenizer from the HuggingFace Hub.

| Argument      | Type   | Default      | Description |
|---------------|--------|--------------|-------------|
| `model_name`  | `str`  | —            | HF model ID or local path |
| `cache_dir`   | `str`  | `None`       | Override HF cache directory |
| `dtype`       | `str`  | `"float16"`  | Weight dtype (`float16`, `bfloat16`, `float32`) |
| `device_map`  | `str`  | `"auto"`     | HF device placement strategy |
| `token`       | `str`  | `None`       | HF API token for gated models |
| `lora_config` | `dict` | `None`       | Default LoRA config passed to `.compile()` |

### `TFLLMModel.compile(optimizer, loss, lora_config)`

Applies LoRA adapters (via `peft`) and stores the optimizer / loss.

| Argument      | Type   | Default        | Description |
|---------------|--------|----------------|-------------|
| `optimizer`   | `str`  | `"adam"`       | Optimizer name |
| `loss`        | `str`  | `"causal_lm"`  | Loss identifier |
| `lora_config` | `dict` | `None`         | LoRA overrides (rank, alpha, target modules, …) |

**Default LoRA config:**
```python
{
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
```

### `TFLLMModel.fit(dataset, epochs, ...)`

Fine-tunes the model. Uses HuggingFace `Trainer` internally; falls back to a
manual PyTorch loop if `Trainer` is unavailable. Returns a history dict
`{"loss": [...], "epoch": [...]}`.

### `TFLLMModel.generate(prompt, max_new_tokens, temperature, ...)`

Generates text from a string or list of strings.

### `TFLLMModel.save_pretrained(directory)`

Saves weights, tokenizer, and a `tf_llm_meta.json` sidecar file. The saved
directory can be reloaded with `tf.llm.from_pretrained(directory)`.

---

## Design Decisions

### 1. HuggingFace as the backend, not a reimplementation

Reimplementing LLM architectures inside TF would duplicate enormous effort and
lag behind the community.  Instead, `tf.llm` is a thin, opinionated wrapper
around `transformers` + `peft`.

**Trade-off:** Users need `pip install transformers peft` (and PyTorch for
training).  These are declared as optional dependencies so that
`import tensorflow` stays fast for users who don't need `tf.llm`.

### 2. Lazy imports

`transformers`, `peft`, and `torch` are imported only when actually called,
not at `import tensorflow`.  This keeps TF's import time unaffected.

### 3. LoRA by default

LoRA is the de-facto standard for efficient LLM fine-tuning.  By applying it
automatically in `compile()`, we give beginners a sensible default while
exposing full configurability via `lora_config`.

### 4. Keras-compatible surface

Using `compile` / `fit` / `generate` mirrors the Keras Model API, lowering the
learning curve for existing TF developers.

---

## Supported Model Families (initial release)

| Family   | Example identifiers |
|----------|---------------------|
| LLaMA    | `meta-llama/Llama-3`, `meta-llama/Llama-3.1`, `meta-llama/Llama-2` |
| Gemma    | `google/gemma-2b`, `google/gemma-7b`, `google/gemma-2` |
| Mistral  | `mistralai/Mistral-7B`, `mistralai/Mixtral-8x7B` |
| Falcon   | `tiiuae/falcon-7b`, `tiiuae/falcon-40b` |
| Generic  | Any HuggingFace `AutoModelForCausalLM` (fallback) |

---

## Dependencies

| Package         | Version    | Required for |
|-----------------|------------|--------------|
| `transformers`  | ≥ 4.40     | Model loading, Trainer |
| `peft`          | ≥ 0.10     | LoRA fine-tuning |
| `torch`         | ≥ 2.2      | Training backend |
| `accelerate`    | ≥ 0.29     | Multi-GPU / device_map="auto" |

All are optional at `import tensorflow` time; a clear `ImportError` with
install instructions is raised when they are needed but missing.

---

## Testing Plan

- **Unit tests** (`llm_model_test.py`): 24 tests covering `_resolve_model_key`,
  `from_pretrained`, `compile`, `fit`, `generate`, `save_pretrained`, `__repr__`,
  and `summary`. All dependencies are stubbed so tests run in CI without GPU or
  internet access.
- **Integration tests** (future): End-to-end test with `google/gemma-2b` on a
  GPU CI runner.
- **Benchmark** (future): Compare fine-tuning throughput vs. raw HuggingFace
  Trainer on A100 × 1 / 8.

---

## Alternatives Considered

| Alternative | Reason rejected |
|-------------|-----------------|
| Extend `tf.keras.Model` directly | Too coupled to Keras internals; would break when Keras is standalone |
| Use KerasNLP | KerasNLP has its own model zoo; `tf.llm` targets **any** HF model |
| TF Hub integration | TF Hub does not support LoRA or HF-style fine-tuning |

---

## Open Questions

1. Should `tf.llm.from_pretrained` also accept a `tf.data.Dataset` schema and
   auto-tokenize?
2. Should we support `int4` / `int8` quantization (bitsandbytes) out of the
   box?
3. Should `model.fit` support `tf.distribute.Strategy` for multi-GPU TF-native
   training, or rely on HuggingFace `accelerate`?

---

## References

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [KerasNLP](https://keras.io/keras_nlp/)

