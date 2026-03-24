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
"""Benchmark Gemma2-2B Keras performance."""

import time
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


def run(gemma_lm, tokenizer, max_len):
  """Benchmarks inferences with at most `max_len` output tokens.

  Args:
    gemma_lm: The Gemma2 Keras model.
    tokenizer: The tokenizer for the Gemma2 model.
    max_len: The maximum number of output tokens per one inference.

  Returns:
    mean ± %diff and the actual number of output tokens generated per inference.
  """
  in_tokens = tokenizer(_QUERY)
  in_tokens_len = len(in_tokens)
  total_output_len = in_tokens_len + max_len
  if max_len < 1:
    print(f"Error: max_len {max_len} should be >= 1")
    exit()

  # Warm up.
  start = time.time()
  output = gemma_lm.generate(_QUERY, max_length=total_output_len + 1)
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
    output = gemma_lm.generate(_QUERY, max_length=total_output_len + 1)
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
  if _VERBOSE:
    print("Query: %s" % _QUERY)

  model_name = "gemma2_2b_en"
  gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
  tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_name)
  mean_1, diff_1, _ = run(gemma_lm, tokenizer, 1)
  mean_n, diff_n, num_output_tokens = run(
      gemma_lm, tokenizer, _NUM_OUTPUT_TOKENS
  )

  print("Generated %d tokens", num_output_tokens)
  tpot = (mean_n - mean_1) / (num_output_tokens - 1)
  print("TTFT: %lf ± %d%% ms" % (mean_1, diff_1))
  print("TPOT: %lf ± %d%% ms" % (tpot, diff_n))


if __name__ == "__main__":
  main()
