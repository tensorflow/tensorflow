# Copyright 2025 The OpenXLA Authors.
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
"""Benchmark Keras causal language models performance."""

import argparse
import sys
import time

import keras_hub
import keras_nlp
import numpy as np


_NUM_OUTPUT_TOKENS = 30
_QUERY = "What is JAX in 3 bullet points?"
_VERBOSE = True


def compute_stats(array):
  """Reports mean and ± range for the given array.

  The range computation follows benchstat's.

  Args:
    array: The array to compute stats for.

  Returns:
    mean and ± %diff range.
  """
  q1 = np.percentile(array, 25)
  q3 = np.percentile(array, 75)
  low = q1 - 1.5 * (q3 - q1)
  high = q3 + 1.5 * (q3 - q1)

  # Remove outliers.
  filtered_array = list(filter(lambda x: low <= x and x <= high, array))

  mean = np.mean(filtered_array)
  min_val = np.min(filtered_array)
  max_val = np.max(filtered_array)
  max_diff = max(max_val - mean, mean - min_val)
  diff = max_diff / mean * 100.0

  return (mean, diff)


def run(lm_model, tokenizer, max_len):
  """Benchmarks inferences with at most `max_len` output tokens.

  Args:
    lm_model: The Keras causal LM model.
    tokenizer: The tokenizer for the model.
    max_len: The maximum number of output tokens per one inference.

  Returns:
    mean ± %diff and the actual number of output tokens generated per inference.
  """
  in_tokens = tokenizer(_QUERY)
  in_tokens_len = len(in_tokens)
  total_output_len = in_tokens_len + max_len
  if max_len < 1:
    print(f"Error: max_len {max_len} should be >= 1")
    sys.exit(1)

  # Warm up.
  start = time.time()
  output = lm_model.generate(_QUERY, max_length=total_output_len + 1)
  warmup_time = (time.time() - start) * 1000
  num_actual_output_tokens = len(tokenizer(output))
  gen_tokens = max(num_actual_output_tokens - in_tokens_len, 0)

  print("Warmup: Number of generated output tokens: ", gen_tokens)

  if _VERBOSE:
    print("=== Max len: %d ===" % max_len)
    print("Warmup: %lf ms" % warmup_time)
    print("Output:\n%s\n" % output)

  times = []
  for i in range(1, 6):
    start = time.time()
    output = lm_model.generate(_QUERY, max_length=total_output_len + 1)
    elapsed_time = (time.time() - start) * 1000
    assert num_actual_output_tokens == len(tokenizer(output))
    times.append(elapsed_time)

    if _VERBOSE:
      print("%d: %lf ms" % (i, elapsed_time))
      print("Benchmark: Number of generated output tokens: ", gen_tokens)

  mean, diff = compute_stats(times)
  if _VERBOSE:
    print("Mean: %lf ± %d%% ms\n" % (mean, diff))

  return (mean, diff, gen_tokens)


def main():
  parser = argparse.ArgumentParser(description="Benchmark Keras LM models.")
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemma2_2b_en",
      help=(
          "Preset name of the model to benchmark. This was tested with"
          " the following models: gemma2_2b_en, gemma3_1b, gemma4_2b"
          "  gpt_oss_20b_en, mixtral_8_7b_en, qwen3_14b_en, qwen2_1.5b_en."
          " Llama models also should work."
          " See https://keras.io/keras_hub/presets/"
          " for the full list of presets."
      )
  )
  parser.add_argument(
      "--use_greedy_sampler",
      action="store_true",
      help="Use greedy sampler instead of Top-K. The default sampler"
      " makes the assertion in run() fail for some models."
  )
  args = parser.parse_args()

  if _VERBOSE:
    print(f"Query: {_QUERY}")

  model_name = args.model_name
  if "gemma4" in model_name:
    lm = keras_nlp.models.Gemma4CausalLM.from_preset(model_name)
    tokenizer = keras_nlp.models.Gemma4Tokenizer.from_preset(model_name)
  elif "gemma3" in model_name:
    lm = keras_nlp.models.Gemma3CausalLM.from_preset(model_name)
    tokenizer = keras_nlp.models.Gemma3Tokenizer.from_preset(model_name)
  elif "gemma" in model_name:
    lm = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
    tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_name)
  elif "llama3" in model_name:
    lm = keras_hub.models.Llama3CausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.Llama3Tokenizer.from_preset(model_name)
  elif "llama" in model_name:
    lm = keras_hub.models.LlamaCausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.LlamaTokenizer.from_preset(model_name)
  elif "mixtral" in model_name:
    lm = keras_hub.models.MixtralCausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.MixtralTokenizer.from_preset(model_name)
  elif "qwen3" in model_name:
    lm = keras_hub.models.Qwen3CausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.Qwen3Tokenizer.from_preset(model_name)
  elif "qwen2" in model_name:
    lm = keras_hub.models.Qwen2CausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.Qwen2Tokenizer.from_preset(model_name)
  elif "qwen" in model_name:
    lm = keras_hub.models.QwenCausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.QwenTokenizer.from_preset(model_name)
  elif "gpt_oss" in model_name:
    lm = keras_hub.models.GptOssCausalLM.from_preset(model_name)
    tokenizer = keras_hub.models.GptOssTokenizer.from_preset(model_name)
  else:
    raise ValueError(f"Unknown model name: {model_name}")

  if args.use_greedy_sampler:
    print("Using greedy sampler")
    lm.compile(sampler="greedy")

  mean_1, diff_1, _ = run(lm, tokenizer, 1)
  mean_n, diff_n, num_output_tokens = run(
      lm, tokenizer, _NUM_OUTPUT_TOKENS
  )

  print("Generated %d tokens", num_output_tokens)
  tpot = (mean_n - mean_1) / (num_output_tokens - 1)
  print("TTFT: %lf ± %d%% ms" % (mean_1, diff_1))
  print("TPOT: %lf ± %d%% ms" % (tpot, diff_n))


if __name__ == "__main__":
  main()
