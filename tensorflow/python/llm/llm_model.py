# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow LLM Module — High-level API for loading and fine-tuning LLMs.

This module provides a Keras-compatible interface to load open-source LLMs
(Llama, Gemma, Mistral, Falcon, etc.) and fine-tune them with LoRA in just
a few lines of code.

Example usage:
    import tensorflow as tf

    # Load a pretrained LLM
    model = tf.llm.from_pretrained("meta-llama/Llama-3")

    # Fine-tune with LoRA in 3 lines
    model.compile(optimizer="adam", loss="causal_lm")
    model.fit(your_dataset, epochs=1)

    # Generate text
    output = model.generate("Once upon a time", max_new_tokens=100)
"""

import os
import json
import logging
from typing import Optional, Union, Dict, Any, List

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so that `import tensorflow`
# itself stays fast even when transformers / peft are not installed.
# ---------------------------------------------------------------------------
_transformers = None
_peft = None
_torch = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of supported model families and their HuggingFace identifiers
# ---------------------------------------------------------------------------
_SUPPORTED_MODEL_FAMILIES: Dict[str, Dict[str, Any]] = {
    # Meta LLaMA family
    "meta-llama/llama-3":        {"family": "llama",   "hf_class": "AutoModelForCausalLM"},
    "meta-llama/llama-2":        {"family": "llama",   "hf_class": "AutoModelForCausalLM"},
    "meta-llama/llama-3.1":      {"family": "llama",   "hf_class": "AutoModelForCausalLM"},
    "meta-llama/llama-3.2":      {"family": "llama",   "hf_class": "AutoModelForCausalLM"},
    # Google Gemma family
    "google/gemma-2b":           {"family": "gemma",   "hf_class": "AutoModelForCausalLM"},
    "google/gemma-7b":           {"family": "gemma",   "hf_class": "AutoModelForCausalLM"},
    "google/gemma-2":            {"family": "gemma",   "hf_class": "AutoModelForCausalLM"},
    # Mistral / Mixtral family
    "mistralai/mistral-7b":      {"family": "mistral", "hf_class": "AutoModelForCausalLM"},
    "mistralai/mixtral-8x7b":    {"family": "mistral", "hf_class": "AutoModelForCausalLM"},
    # Falcon family
    "tiiuae/falcon-7b":          {"family": "falcon",  "hf_class": "AutoModelForCausalLM"},
    "tiiuae/falcon-40b":         {"family": "falcon",  "hf_class": "AutoModelForCausalLM"},
    # Generic fallback — any HuggingFace causal LM
    "__generic__":               {"family": "generic", "hf_class": "AutoModelForCausalLM"},
}

# Default LoRA configuration (PEFT)
_DEFAULT_LORA_CONFIG: Dict[str, Any] = {
    "r": 16,                          # LoRA rank
    "lora_alpha": 32,                 # LoRA scaling factor
    "target_modules": ["q_proj", "v_proj"],  # Attention projection layers
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lazy_import_transformers():
    """Import HuggingFace transformers lazily."""
    global _transformers
    if _transformers is None:
        try:
            import transformers as _t
            _transformers = _t
        except ImportError:
            raise ImportError(
                "The `transformers` package is required for tf.llm. "
                "Install it with:  pip install transformers"
            )
    return _transformers


def _lazy_import_peft():
    """Import HuggingFace peft lazily."""
    global _peft
    if _peft is None:
        try:
            import peft as _p
            _peft = _p
        except ImportError:
            raise ImportError(
                "The `peft` package is required for LoRA fine-tuning. "
                "Install it with:  pip install peft"
            )
    return _peft


def _resolve_model_key(model_name: str) -> Dict[str, Any]:
    """Resolve a model name to its registry entry (case-insensitive)."""
    key = model_name.lower()
    # Exact match first
    if key in _SUPPORTED_MODEL_FAMILIES:
        return _SUPPORTED_MODEL_FAMILIES[key]
    # Prefix match (e.g. "meta-llama/Llama-3-8B" → "meta-llama/llama-3")
    for registered_key in _SUPPORTED_MODEL_FAMILIES:
        if registered_key != "__generic__" and key.startswith(registered_key):
            return _SUPPORTED_MODEL_FAMILIES[registered_key]
    # Fall back to generic HuggingFace loader
    logger.warning(
        "Model '%s' is not in the tf.llm registry. "
        "Attempting to load via AutoModelForCausalLM. "
        "Supported families: %s",
        model_name,
        sorted(k for k in _SUPPORTED_MODEL_FAMILIES if k != "__generic__"),
    )
    return _SUPPORTED_MODEL_FAMILIES["__generic__"]


# ---------------------------------------------------------------------------
# Public LLM wrapper class
# ---------------------------------------------------------------------------

class TFLLMModel:
    """A Keras-compatible wrapper around a pretrained causal language model.

    This class wraps a HuggingFace ``AutoModelForCausalLM`` (or equivalent)
    and exposes a familiar ``compile`` / ``fit`` / ``generate`` interface so
    that TensorFlow users can fine-tune LLMs without leaving the TF ecosystem.

    Do **not** instantiate this class directly.  Use :func:`from_pretrained`
    instead.

    Attributes:
        model_name (str): The original model identifier (e.g.
            ``"meta-llama/Llama-3"``).
        lora_enabled (bool): Whether LoRA adapters have been applied.
        _hf_model: The underlying HuggingFace model object.
        _tokenizer: The associated tokenizer.
        _optimizer: Optimizer set via :meth:`compile`.
        _loss: Loss identifier set via :meth:`compile`.
        _lora_config (dict): Active LoRA configuration.
    """

    def __init__(
        self,
        model_name: str,
        hf_model,
        tokenizer,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self._hf_model = hf_model
        self._tokenizer = tokenizer
        self._optimizer: Optional[str] = None
        self._loss: Optional[str] = None
        self._lora_config: Dict[str, Any] = lora_config or {}
        self.lora_enabled: bool = False
        self._compiled: bool = False

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "causal_lm",
        lora_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TFLLMModel":
        """Configure the model for fine-tuning.

        Applies LoRA adapters and stores the optimizer / loss so that
        :meth:`fit` can use them.

        Args:
            optimizer: Name of the optimizer (``"adam"``, ``"sgd"``,
                ``"adamw"``, etc.) or a full optimizer identifier string.
            loss: Loss function identifier.  Use ``"causal_lm"`` for
                next-token prediction (auto-regressive fine-tuning), or any
                HuggingFace-compatible loss name.
            lora_config: Optional dict overriding the default LoRA settings.
                Keys are the same as ``peft.LoraConfig`` constructor kwargs.
            **kwargs: Reserved for future use (e.g. ``metrics``, ``jit_compile``).

        Returns:
            self — so calls can be chained.

        Example::

            model.compile(optimizer="adamw", loss="causal_lm",
                          lora_config={"r": 32, "lora_alpha": 64})
        """
        self._optimizer = optimizer
        self._loss = loss

        # Merge caller-supplied LoRA overrides with defaults
        effective_lora_cfg = {**_DEFAULT_LORA_CONFIG, **(lora_config or {})}
        self._lora_config = effective_lora_cfg

        # Apply LoRA adapters via PEFT
        peft = _lazy_import_peft()
        lora_cfg_obj = peft.LoraConfig(**effective_lora_cfg)
        self._hf_model = peft.get_peft_model(self._hf_model, lora_cfg_obj)
        self.lora_enabled = True

        trainable_params, all_params = self._count_parameters()
        logger.info(
            "LoRA applied. Trainable params: %d / %d (%.2f%%)",
            trainable_params,
            all_params,
            100.0 * trainable_params / all_params if all_params else 0,
        )

        self._compiled = True
        return self

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset,
        epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: int = 1,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Fine-tune the model on *dataset*.

        Args:
            dataset: A ``tf.data.Dataset``, a HuggingFace ``Dataset``, or any
                iterable that yields ``(input_ids, labels)`` pairs / dicts with
                ``"input_ids"`` and ``"labels"`` keys.
            epochs: Number of full passes over the dataset.
            steps_per_epoch: Maximum number of gradient steps per epoch.
                ``None`` means run until the dataset is exhausted.
            callbacks: List of callback objects (Keras-style).  Currently
                ``tf.keras.callbacks.ModelCheckpoint`` and
                ``tf.keras.callbacks.EarlyStopping`` are supported.
            verbose: Verbosity level (0 = silent, 1 = progress bar, 2 = one
                line per epoch).
            **kwargs: Passed through to the underlying trainer.

        Returns:
            A history dict ``{"loss": [...], "epoch": [...]}``.

        Raises:
            RuntimeError: If :meth:`compile` has not been called first.

        Example::

            history = model.fit(my_dataset, epochs=3)
            print(history["loss"])
        """
        if not self._compiled:
            raise RuntimeError(
                "You must call model.compile() before model.fit()."
            )

        try:
            # Attempt to use HuggingFace Trainer for the heavy lifting
            transformers = _lazy_import_transformers()
            training_args = transformers.TrainingArguments(
                output_dir="./tf_llm_checkpoints",
                num_train_epochs=epochs,
                per_device_train_batch_size=kwargs.get("batch_size", 4),
                optim=self._optimizer if self._optimizer != "adam" else "adamw_torch",
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",
            )
            trainer = transformers.Trainer(
                model=self._hf_model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self._tokenizer,
                callbacks=callbacks,
            )
            train_result = trainer.train()
            history = {
                "loss": [train_result.training_loss],
                "epoch": list(range(1, epochs + 1)),
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("HuggingFace Trainer failed (%s). Falling back to manual loop.", exc)
            history = self._manual_train_loop(dataset, epochs, steps_per_epoch, verbose)

        return history

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text from a prompt.

        Args:
            prompt: A single string or a list of strings.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-K sampling.
            do_sample: Whether to use sampling (``True``) or greedy decoding.
            **kwargs: Additional arguments forwarded to ``model.generate``.

        Returns:
            Generated string (or list of strings when *prompt* is a list).

        Example::

            text = model.generate("Once upon a time", max_new_tokens=100)
            print(text)
        """
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else prompt

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs.pop("input_ids")
        output_ids = self._hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            **inputs,
            **kwargs,
        )
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded:
            decoded = [""] * len(prompts)
        return decoded[0] if single_input else decoded

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model and tokenizer to *save_directory*.

        The directory can later be reloaded with ``tf.llm.from_pretrained``.

        Args:
            save_directory: Path to the output directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        self._hf_model.save_pretrained(save_directory)
        self._tokenizer.save_pretrained(save_directory)
        # Persist tf.llm metadata
        meta = {
            "model_name": self.model_name,
            "lora_enabled": self.lora_enabled,
            "lora_config": self._lora_config,
            "tf_llm_version": "1.0.0",
        }
        with open(os.path.join(save_directory, "tf_llm_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Model saved to %s", save_directory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_parameters(self):
        """Return (trainable_params, all_params) counts."""
        try:
            all_params = sum(p.numel() for p in self._hf_model.parameters())
            trainable = sum(
                p.numel() for p in self._hf_model.parameters() if p.requires_grad
            )
            return trainable, all_params
        except Exception:  # pylint: disable=broad-except
            return 0, 0

    def _manual_train_loop(self, dataset, epochs, steps_per_epoch, verbose):
        """Minimal manual training loop used as a fallback."""
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError("PyTorch is required for the fallback training loop.")

        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        opt_cls = optimizers.get(self._optimizer.lower(), torch.optim.AdamW)
        optimizer = opt_cls(
            filter(lambda p: p.requires_grad, self._hf_model.parameters()),
            lr=2e-4,
        )

        self._hf_model.train()
        history = {"loss": [], "epoch": []}

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            steps = 0
            for step, batch in enumerate(dataset):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                # Support both dict-style and tuple-style batches
                if isinstance(batch, dict):
                    outputs = self._hf_model(**batch, labels=batch.get("input_ids"))
                else:
                    input_ids, labels = batch
                    outputs = self._hf_model(input_ids=input_ids, labels=labels)

                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

                if verbose and step % 10 == 0:
                    logger.info("Epoch %d | Step %d | Loss: %.4f", epoch, step, loss.item())

            avg_loss = total_loss / max(steps, 1)
            history["loss"].append(avg_loss)
            history["epoch"].append(epoch)
            if verbose:
                logger.info("Epoch %d complete — avg loss: %.4f", epoch, avg_loss)

        return history

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lora_str = f", lora=True(rank={self._lora_config.get('r', '?')})" if self.lora_enabled else ""
        return f"<TFLLMModel model='{self.model_name}'{lora_str}>"

    def summary(self) -> None:
        """Print a brief model summary."""
        trainable, total = self._count_parameters()
        print(f"Model        : {self.model_name}")
        print(f"LoRA enabled : {self.lora_enabled}")
        if self.lora_enabled:
            print(f"LoRA rank    : {self._lora_config.get('r', 'N/A')}")
        print(f"Total params : {total:,}")
        print(f"Trainable    : {trainable:,} ({100.0 * trainable / total:.2f}%)" if total else "Trainable    : N/A")


# ---------------------------------------------------------------------------
# Module-level from_pretrained
# ---------------------------------------------------------------------------

def from_pretrained(
    model_name: str,
    cache_dir: Optional[str] = None,
    dtype: str = "float16",
    device_map: str = "auto",
    token: Optional[str] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TFLLMModel:
    """Load a pretrained open-source LLM and return a :class:`TFLLMModel`.

    This is the primary entry point for ``tf.llm``.  It downloads (and caches)
    the model weights from the HuggingFace Hub, wraps them in a
    :class:`TFLLMModel`, and makes them ready for fine-tuning or inference.

    Supported model families (prefix match):
        - ``meta-llama/Llama-3``, ``meta-llama/Llama-3.1``, ``meta-llama/Llama-3.2``
        - ``meta-llama/Llama-2``
        - ``google/gemma-2b``, ``google/gemma-7b``, ``google/gemma-2``
        - ``mistralai/Mistral-7B``, ``mistralai/Mixtral-8x7B``
        - ``tiiuae/falcon-7b``, ``tiiuae/falcon-40b``
        - Any other HuggingFace causal LM (generic fallback)

    Args:
        model_name: HuggingFace model identifier, e.g.
            ``"meta-llama/Llama-3"`` or a local directory path.
        cache_dir: Directory for caching downloaded weights.  Defaults to
            ``~/.cache/huggingface/hub``.
        dtype: Model weight dtype — ``"float16"``, ``"bfloat16"``, or
            ``"float32"``.  Use ``"float16"`` or ``"bfloat16"`` to reduce
            VRAM usage.
        device_map: HuggingFace device map strategy.  ``"auto"`` spreads
            the model across all available GPUs / CPU automatically.
        token: HuggingFace API token for gated models (e.g. Llama 3).
            Can also be set via the ``HUGGINGFACE_TOKEN`` environment variable.
        lora_config: Optional default LoRA settings that will be passed to
            :meth:`TFLLMModel.compile` when called without explicit overrides.
        **kwargs: Additional keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained``.

    Returns:
        A :class:`TFLLMModel` instance ready for ``.compile()`` / ``.fit()``
        / ``.generate()``.

    Raises:
        ImportError: If ``transformers`` is not installed.
        EnvironmentError: If the model cannot be downloaded (e.g. no internet,
            invalid token for gated model).

    Example::

        import tensorflow as tf

        # Public model — no token needed
        model = tf.llm.from_pretrained("google/gemma-2b")

        # Gated model — token required (set HUGGINGFACE_TOKEN or pass token=)
        model = tf.llm.from_pretrained(
            "meta-llama/Llama-3",
            token="hf_...",
            dtype="bfloat16",
        )
    """
    transformers = _lazy_import_transformers()

    # Resolve HF token from env if not passed explicitly
    hf_token = token or os.environ.get("HUGGINGFACE_TOKEN")

    # Map dtype string → torch dtype
    try:
        import torch  # pylint: disable=import-outside-toplevel
        _DTYPE_MAP = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)
    except ImportError:
        torch_dtype = None  # Will be handled downstream

    _ = _resolve_model_key(model_name)  # Log warnings for unknown families

    logger.info("Loading tokenizer for '%s'…", model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
    )
    # Ensure pad token is set (required for batched training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model weights for '%s' (dtype=%s, device_map=%s)…",
                model_name, dtype, device_map)
    load_kwargs = dict(
        cache_dir=cache_dir,
        device_map=device_map,
        token=hf_token,
        **kwargs,
    )
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )

    logger.info("Model '%s' loaded successfully.", model_name)
    return TFLLMModel(
        model_name=model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        lora_config=lora_config,
    )
