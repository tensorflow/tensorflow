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
"""Tests for tf.llm (tensorflow.python.llm).

Run with Bazel:
    bazel test //tensorflow/python/llm:llm_model_test

Run standalone (no GPU / internet required):
    python -m tensorflow.python.llm.llm_model_test
"""

import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# tf_logging — use TF-internal logger so Bazel / CI can parse error objects
# ---------------------------------------------------------------------------
from tensorflow.python.platform import tf_logging as logging

# ---------------------------------------------------------------------------
# Lightweight stubs — injected before the module under test is imported so
# that heavy optional deps (transformers / peft / torch) are not required in
# CI environments that lack a GPU or internet access.
# ---------------------------------------------------------------------------

def _make_transformers_stub():
    """Return a minimal `transformers` stub."""
    stub = types.ModuleType("transformers")

    class FakeTokenizer:
        pad_token = None
        eos_token  = "<eos>"

        def __call__(self, texts, return_tensors=None,
                     padding=True, truncation=True):
            return {
                "input_ids":      MagicMock(),
                "attention_mask": MagicMock(),
            }

        def batch_decode(self, ids, skip_special_tokens=True):
            n = len(ids) if hasattr(ids, "__len__") else 1
            return ["generated text"] * n

        def save_pretrained(self, path):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        def parameters(self):
            return iter([])

        def train(self):
            pass

        def generate(self, **kwargs):
            return MagicMock()

        def save_pretrained(self, path):
            pass

        def __call__(self, **kwargs):
            return MagicMock(loss=MagicMock(item=lambda: 0.5))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeTrainingArguments:
        def __init__(self, *args, **kwargs):
            pass

    class FakeTrainResult:
        training_loss = 0.42

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self):
            return FakeTrainResult()

    stub.AutoTokenizer         = FakeTokenizer
    stub.AutoModelForCausalLM  = FakeModel
    stub.TrainingArguments     = FakeTrainingArguments
    stub.Trainer               = FakeTrainer
    return stub


def _make_peft_stub():
    """Return a minimal `peft` stub."""
    stub = types.ModuleType("peft")

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def get_peft_model(model, config):
        model._lora_applied = True
        return model

    stub.LoraConfig      = FakeLoraConfig
    stub.get_peft_model  = get_peft_model
    return stub


def _make_torch_stub():
    """Return a minimal `torch` stub."""
    stub = types.ModuleType("torch")
    stub.float16  = "float16"
    stub.bfloat16 = "bfloat16"
    stub.float32  = "float32"

    class _FakeOptimizer:
        def __init__(self, params, lr=1e-3):
            pass
        def zero_grad(self):
            pass
        def step(self):
            pass

    class _FakeOptim:
        Adam  = _FakeOptimizer
        AdamW = _FakeOptimizer
        SGD   = _FakeOptimizer

    stub.optim = _FakeOptim()
    return stub


# Inject stubs *before* the module under test is loaded.
sys.modules.setdefault("transformers", _make_transformers_stub())
sys.modules.setdefault("peft",         _make_peft_stub())
sys.modules.setdefault("torch",        _make_torch_stub())

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from tensorflow.python.llm.llm_model import (  # noqa: E402
    from_pretrained,
    TFLLMModel,
    _resolve_model_key,
    _SUPPORTED_MODEL_FAMILIES,
    _DEFAULT_LORA_CONFIG,
)

# ---------------------------------------------------------------------------
# Use tf.test.TestCase so TF handles:
#   • eager/graph session cleanup between tests
#   • internal memory-leak tracking
#   • Bazel-compatible error-object generation on failure
# ---------------------------------------------------------------------------
from tensorflow.python.framework import test_util
from tensorflow.python.platform  import googletest  # exposes main()
import tensorflow as tf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fresh_model(model_id: str = "google/gemma-2b") -> TFLLMModel:
    """Return a new TFLLMModel without LoRA applied."""
    return from_pretrained(model_id)


def _compiled_model(model_id: str = "google/gemma-2b") -> TFLLMModel:
    """Return a TFLLMModel with LoRA compiled and ready to train."""
    model = _fresh_model(model_id)
    model.compile(optimizer="adam", loss="causal_lm")
    return model


# ===========================================================================
# Test suites
# ===========================================================================

@test_util.run_all_in_graph_and_eager_modes
class TestResolveModelKey(tf.test.TestCase):
    """Tests for the internal _resolve_model_key() registry lookup."""

    def test_exact_match_llama3(self):
        info = _resolve_model_key("meta-llama/Llama-3")
        self.assertEqual(info["family"], "llama")

    def test_case_insensitive_lookup(self):
        info = _resolve_model_key("META-LLAMA/LLAMA-3")
        self.assertEqual(info["family"], "llama")

    def test_prefix_match_llama_variant(self):
        # e.g. "meta-llama/Llama-3-8B-Instruct" should resolve to llama
        info = _resolve_model_key("meta-llama/Llama-3-8B-Instruct")
        self.assertEqual(info["family"], "llama")

    def test_gemma_exact_match(self):
        info = _resolve_model_key("google/gemma-2b")
        self.assertEqual(info["family"], "gemma")

    def test_mistral_exact_match(self):
        info = _resolve_model_key("mistralai/Mistral-7B")
        self.assertEqual(info["family"], "mistral")

    def test_falcon_exact_match(self):
        info = _resolve_model_key("tiiuae/falcon-7b")
        self.assertEqual(info["family"], "falcon")

    def test_unknown_model_falls_back_to_generic(self):
        info = _resolve_model_key("some-org/some-unknown-model-xyz")
        self.assertEqual(info["family"], "generic")

    def test_all_registered_families_have_hf_class(self):
        for key, meta in _SUPPORTED_MODEL_FAMILIES.items():
            self.assertIn(
                "hf_class", meta,
                msg=f"Registry entry '{key}' is missing 'hf_class'",
            )


@test_util.run_all_in_graph_and_eager_modes
class TestFromPretrained(tf.test.TestCase):
    """Tests for the from_pretrained() public factory."""

    def test_returns_tflllmmodel_instance(self):
        model = _fresh_model()
        self.assertIsInstance(model, TFLLMModel)

    def test_model_name_attribute_stored(self):
        model = from_pretrained("mistralai/Mistral-7B")
        self.assertEqual(model.model_name, "mistralai/Mistral-7B")

    def test_lora_disabled_by_default(self):
        model = _fresh_model()
        self.assertFalse(model.lora_enabled)

    def test_compiled_flag_false_before_compile(self):
        model = _fresh_model()
        self.assertFalse(model._compiled)

    def test_tokenizer_pad_token_auto_set(self):
        """pad_token must equal eos_token for batched training to work."""
        model = _fresh_model()
        self.assertEqual(
            model._tokenizer.pad_token,
            model._tokenizer.eos_token,
        )

    def test_env_huggingface_token_accepted(self):
        """HUGGINGFACE_TOKEN env var must not cause an error."""
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_test_token_123"}):
            model = from_pretrained("google/gemma-2b")
        self.assertIsNotNone(model)

    def test_hf_model_attribute_present(self):
        model = _fresh_model()
        self.assertTrue(
            hasattr(model, "_hf_model"),
            msg="TFLLMModel must expose _hf_model for weight access",
        )

    def test_tokenizer_attribute_present(self):
        model = _fresh_model()
        self.assertTrue(hasattr(model, "_tokenizer"))


@test_util.run_all_in_graph_and_eager_modes
class TestCompile(tf.test.TestCase):
    """Tests for TFLLMModel.compile()."""

    def test_sets_optimizer_attribute(self):
        model = _fresh_model()
        model.compile(optimizer="adamw", loss="causal_lm")
        self.assertEqual(model._optimizer, "adamw")

    def test_sets_loss_attribute(self):
        model = _fresh_model()
        model.compile(optimizer="adam", loss="causal_lm")
        self.assertEqual(model._loss, "causal_lm")

    def test_enables_lora_flag(self):
        model = _fresh_model()
        model.compile(optimizer="adam", loss="causal_lm")
        self.assertTrue(model.lora_enabled)

    def test_sets_compiled_flag(self):
        model = _fresh_model()
        self.assertFalse(model._compiled)
        model.compile(optimizer="adam", loss="causal_lm")
        self.assertTrue(model._compiled)

    def test_returns_self_for_method_chaining(self):
        model = _fresh_model()
        result = model.compile(optimizer="adam", loss="causal_lm")
        self.assertIs(result, model)

    def test_lora_rank_override(self):
        model = _fresh_model()
        model.compile(
            optimizer="adam",
            loss="causal_lm",
            lora_config={"r": 64},
        )
        self.assertEqual(model._lora_config["r"], 64)

    def test_lora_alpha_override(self):
        model = _fresh_model()
        model.compile(
            optimizer="adam",
            loss="causal_lm",
            lora_config={"lora_alpha": 128},
        )
        self.assertEqual(model._lora_config["lora_alpha"], 128)

    def test_default_lora_keys_all_present(self):
        model = _compiled_model()
        for key in _DEFAULT_LORA_CONFIG:
            self.assertIn(
                key, model._lora_config,
                msg=f"Default LoRA key '{key}' missing after compile()",
            )

    def test_compile_twice_is_idempotent(self):
        """Calling compile() a second time must not raise."""
        model = _fresh_model()
        model.compile(optimizer="adam", loss="causal_lm")
        model.compile(optimizer="adamw", loss="causal_lm")
        self.assertEqual(model._optimizer, "adamw")


@test_util.run_all_in_graph_and_eager_modes
class TestFit(tf.test.TestCase):
    """Tests for TFLLMModel.fit()."""

    def test_fit_before_compile_raises_runtime_error(self):
        model = _fresh_model()
        with self.assertRaises(RuntimeError):
            model.fit([], epochs=1)

    def test_fit_returns_dict_with_loss_key(self):
        history = _compiled_model().fit([], epochs=1)
        self.assertIn("loss", history)

    def test_fit_returns_dict_with_epoch_key(self):
        history = _compiled_model().fit([], epochs=1)
        self.assertIn("epoch", history)

    def test_fit_history_has_at_least_one_entry(self):
        history = _compiled_model().fit([], epochs=1)
        self.assertGreaterEqual(len(history["epoch"]), 1)

    def test_fit_loss_values_are_numeric(self):
        history = _compiled_model().fit([], epochs=1)
        for val in history["loss"]:
            self.assertIsInstance(val, float)


@test_util.run_all_in_graph_and_eager_modes
class TestGenerate(tf.test.TestCase):
    """Tests for TFLLMModel.generate()."""

    def test_single_string_prompt_returns_str(self):
        result = _fresh_model().generate("Hello, world!")
        self.assertIsInstance(result, str)

    def test_list_of_prompts_returns_list(self):
        result = _fresh_model().generate(["Hello", "World"])
        self.assertIsInstance(result, list)

    def test_list_output_length_matches_input(self):
        prompts = ["Prompt A", "Prompt B", "Prompt C"]
        result  = _fresh_model().generate(prompts)
        self.assertLen(result, len(prompts))

    def test_generate_does_not_require_compile(self):
        """Inference must work without calling compile() first."""
        model = _fresh_model()
        result = model.generate("Test prompt")
        self.assertIsNotNone(result)


@test_util.run_all_in_graph_and_eager_modes
class TestSavePretrained(tf.test.TestCase):
    """Tests for TFLLMModel.save_pretrained()."""

    def test_meta_json_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _fresh_model().save_pretrained(tmpdir)
            meta_path = os.path.join(tmpdir, "tf_llm_meta.json")
            self.assertTrue(
                os.path.exists(meta_path),
                msg="tf_llm_meta.json not found after save_pretrained()",
            )

    def test_meta_json_contains_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _fresh_model("google/gemma-2b").save_pretrained(tmpdir)
            with open(os.path.join(tmpdir, "tf_llm_meta.json")) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["model_name"], "google/gemma-2b")

    def test_meta_json_contains_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _fresh_model().save_pretrained(tmpdir)
            with open(os.path.join(tmpdir, "tf_llm_meta.json")) as fh:
                meta = json.load(fh)
            self.assertIn("tf_llm_version", meta)

    def test_meta_json_records_lora_status(self):
        model = _compiled_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            with open(os.path.join(tmpdir, "tf_llm_meta.json")) as fh:
                meta = json.load(fh)
            self.assertTrue(meta["lora_enabled"])

    def test_output_directory_created_if_missing(self):
        with tempfile.TemporaryDirectory() as base:
            new_dir = os.path.join(base, "nested", "output")
            _fresh_model().save_pretrained(new_dir)
            self.assertTrue(os.path.isdir(new_dir))


@test_util.run_all_in_graph_and_eager_modes
class TestReprAndSummary(tf.test.TestCase):
    """Tests for __repr__ and summary()."""

    def test_repr_contains_model_name(self):
        model = _fresh_model("google/gemma-2b")
        self.assertIn("google/gemma-2b", repr(model))

    def test_repr_no_lora_tag_before_compile(self):
        model = _fresh_model()
        self.assertNotIn("lora=True", repr(model))

    def test_repr_shows_lora_tag_after_compile(self):
        model = _compiled_model()
        self.assertIn("lora=True", repr(model))

    def test_repr_shows_lora_rank(self):
        model = _fresh_model()
        model.compile("adam", "causal_lm", lora_config={"r": 32})
        self.assertIn("32", repr(model))

    def test_summary_runs_without_raising(self):
        """summary() must complete without exceptions."""
        _compiled_model().summary()

    def test_summary_before_compile_runs_without_raising(self):
        _fresh_model().summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    googletest.main()
