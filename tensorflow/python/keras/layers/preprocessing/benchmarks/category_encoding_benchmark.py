# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for Keras category_encoding preprocessing layer."""

import time

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

v2_compat.enable_v2_behavior()


class BenchmarkLayer(benchmark.TensorFlowBenchmark):
  """Benchmark the layer forward pass."""

  def run_dataset_implementation(self, output_mode, batch_size, sequence_length,
                                 max_tokens):
    input_t = keras.Input(shape=(sequence_length,), dtype=dtypes.int32)
    layer = category_encoding.CategoryEncoding(
        max_tokens=max_tokens, output_mode=output_mode)
    _ = layer(input_t)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = dataset_ops.Dataset.from_tensor_slices(
          random_ops.random_uniform([batch_size * 10, sequence_length],
                                    minval=0,
                                    maxval=max_tokens - 1,
                                    dtype=dtypes.int32))
      ds = ds.shuffle(batch_size * 100)
      ds = ds.batch(batch_size)
      num_batches = 5
      ds = ds.take(num_batches)
      ds = ds.prefetch(num_batches)
      starts.append(time.time())
      # Benchmarked code begins here.
      for i in ds:
        _ = layer(i)
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts)) / num_batches
    name = "category_encoding|batch_%s|seq_length_%s|%s_max_tokens" % (
        batch_size, sequence_length, max_tokens)
    self.report_benchmark(iters=num_repeats, wall_time=avg_time, name=name)

  def benchmark_vocab_size_by_batch(self):
    for batch in [32, 256, 2048]:
      for sequence_length in [10, 1000]:
        for num_tokens in [100, 1000, 20000]:
          self.run_dataset_implementation(
              output_mode="count",
              batch_size=batch,
              sequence_length=sequence_length,
              max_tokens=num_tokens)


if __name__ == "__main__":
  test.main()
