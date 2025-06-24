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
"""Benchmark gemma2-2b-it PyTorch performance."""

import datetime
import os
import statistics

from gemma import config
from gemma import model
import torch

VARIANT = '2b-it'
MACHINE_TYPE = 'cpu'

# Configure to compatible runtime version.
CONFIG = VARIANT[:2]
if CONFIG == '2b':
  CONFIG = '2b-v2'

# Assign Gemma path
GEMMA_PATH = os.environ.get('MODEL_DIR')

# Ensure that the tokenizer is present
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
assert os.path.isfile(TOKENIZER_PATH), 'Gemma2 PyTorch tokenizer not found!'

# Ensure that the checkpoint is present
CKPT_PATH = os.path.join(GEMMA_PATH, 'model.ckpt')
assert os.path.isfile(CKPT_PATH), 'Gemma2 PyTorch checkpoint not found!'

# Set up model config.
model_config = config.get_model_config(CONFIG)
model_config.tokenizer = TOKENIZER_PATH
model_config.quant = 'quant' in VARIANT

# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = model.GemmaForCausalLM(model_config)
model.load_weights(CKPT_PATH)
model = model.to(device).eval()

OUTPUT_TOKEN_LEN = 128
prompt = ['What is JAX in 3 bullet points?']


def benchmark_generation_time(output_token_len):
  """Benchmark generation time given output token length."""
  timestamp_start = datetime.datetime.now()
  model_ids = model.generate(
      prompt,
      device=device,
      output_len=output_token_len,
  )
  timestamp_end = datetime.datetime.now()
  timer_delta = timestamp_end - timestamp_start
  # Prints generated tokens if the output token length is the full length.
  if output_token_len == OUTPUT_TOKEN_LEN:
    print(model_ids)
  return timer_delta.total_seconds() * 1000


def display_tpot():
  """Calculate the time per output token."""
  e2e_latency_mean = statistics.mean(latency_list)
  ttft_mean = statistics.mean(ttft_ms_list)
  generation_time_mean = e2e_latency_mean - ttft_mean
  tpot = generation_time_mean / (OUTPUT_TOKEN_LEN - 1)
  print(f'TPOT: {round(tpot, 2)} ms')


def display_benchmark_results(timer_list, metric_name):
  """Display mean and stdev for a given metric."""
  mean_time = statistics.mean(timer_list)
  stdev_time = statistics.stdev(timer_list)
  stdev_time_percentage = (stdev_time / mean_time) * 100

  print(
      '%s: %.2f ms ± %.2f%%' % (metric_name, mean_time, stdev_time_percentage)
  )


if __name__ == '__main__':
  # Measure time to first token.
  ttft_ms_list = [benchmark_generation_time(1) for _ in range(5)]
  # Measure time for full tokens.
  latency_list = [benchmark_generation_time(OUTPUT_TOKEN_LEN) for _ in range(5)]

  display_benchmark_results(ttft_ms_list, 'TTFT')
  display_benchmark_results(latency_list, 'E2E Latency')
  display_tpot()
