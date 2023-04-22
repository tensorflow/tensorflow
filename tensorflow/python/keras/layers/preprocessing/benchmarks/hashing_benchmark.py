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
"""Benchmark for Keras hashing preprocessing layer."""

import itertools
import random
import string
import time

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.preprocessing import hashing
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

v2_compat.enable_v2_behavior()


# word_gen creates random sequences of ASCII letters (both lowercase and upper).
# The number of unique strings is ~2,700.
def word_gen():
  for _ in itertools.count(1):
    yield "".join(random.choice(string.ascii_letters) for i in range(2))


class BenchmarkLayer(benchmark.TensorFlowBenchmark):
  """Benchmark the layer forward pass."""

  def run_dataset_implementation(self, batch_size):
    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.string,
                                              tensor_shape.TensorShape([]))
      ds = ds.shuffle(batch_size * 100)
      ds = ds.batch(batch_size)
      num_batches = 5
      ds = ds.take(num_batches)
      ds = ds.prefetch(num_batches)
      starts.append(time.time())
      # Benchmarked code begins here.
      for i in ds:
        _ = string_ops.string_to_hash_bucket(i, num_buckets=2)
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts)) / num_batches
    return avg_time

  def bm_layer_implementation(self, batch_size):
    input_1 = keras.Input(shape=(None,), dtype=dtypes.string, name="word")
    layer = hashing.Hashing(num_bins=2)
    _ = layer(input_1)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.string,
                                              tensor_shape.TensorShape([]))
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
    name = "hashing|batch_%s" % batch_size
    baseline = self.run_dataset_implementation(batch_size)
    extras = {
        "dataset implementation baseline": baseline,
        "delta seconds": (baseline - avg_time),
        "delta percent": ((baseline - avg_time) / baseline) * 100
    }
    self.report_benchmark(
        iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

  def benchmark_vocab_size_by_batch(self):
    for batch in [32, 64, 256]:
      self.bm_layer_implementation(batch_size=batch)


if __name__ == "__main__":
  test.main()
