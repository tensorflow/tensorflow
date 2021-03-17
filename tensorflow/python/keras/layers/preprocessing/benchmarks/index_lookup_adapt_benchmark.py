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
"""Benchmark for Keras text vectorization preprocessing layer's adapt method."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
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
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

v2_compat.enable_v2_behavior()


# word_gen creates random sequences of ASCII letters (both lowercase and upper).
# The number of unique strings is ~2,700.
def word_gen():
  for _ in itertools.count(1):
    yield "".join(random.choice(string.ascii_letters) for i in range(2))


def get_top_k(dataset, k):
  """Python implementation of vocabulary building using a defaultdict."""
  counts = collections.defaultdict(int)
  for tensor in dataset:
    data = tensor.numpy()
    for element in data:
      counts[element] += 1
  sorted_vocab = [
      k for k, _ in sorted(
          counts.items(), key=lambda item: item[1], reverse=True)
  ]
  if len(sorted_vocab) > k:
    sorted_vocab = sorted_vocab[:k]
  return sorted_vocab


class BenchmarkAdapt(benchmark.TensorFlowBenchmark):
  """Benchmark adapt."""

  def run_numpy_implementation(self, num_elements, batch_size, k):
    """Test the python implementation."""
    ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.string,
                                            tensor_shape.TensorShape([]))
    batched_ds = ds.take(num_elements).batch(batch_size)
    input_t = keras.Input(shape=(), dtype=dtypes.string)
    layer = index_lookup.IndexLookup(
        max_tokens=k,
        num_oov_indices=0,
        mask_token=None,
        oov_token="OOV",
        dtype=dtypes.string)
    _ = layer(input_t)
    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      starts.append(time.time())
      vocab = get_top_k(batched_ds, k)
      layer.set_vocabulary(vocab)
      ends.append(time.time())
    avg_time = np.mean(np.array(ends) - np.array(starts))
    return avg_time

  def bm_adapt_implementation(self, num_elements, batch_size, k):
    """Test the KPL adapt implementation."""
    ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.string,
                                            tensor_shape.TensorShape([]))
    batched_ds = ds.take(num_elements).batch(batch_size)
    input_t = keras.Input(shape=(), dtype=dtypes.string)
    layer = index_lookup.IndexLookup(
        max_tokens=k,
        num_oov_indices=0,
        mask_token=None,
        oov_token="OOV",
        dtype=dtypes.string)
    _ = layer(input_t)
    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      starts.append(time.time())
      layer.adapt(batched_ds)
      ends.append(time.time())
    avg_time = np.mean(np.array(ends) - np.array(starts))
    name = "index_lookup_adapt|%s_elements|vocab_size_%s|batch_%s" % (
        num_elements, k, batch_size)
    baseline = self.run_numpy_implementation(num_elements, batch_size, k)
    extras = {
        "numpy implementation baseline": baseline,
        "delta seconds": (baseline - avg_time),
        "delta percent": ((baseline - avg_time) / baseline) * 100
    }
    self.report_benchmark(
        iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

  def benchmark_vocab_size_by_batch(self):
    for vocab_size in [100, 1000, 10000, 100000, 1000000]:
      for batch in [1, 16, 2048]:
        self.bm_adapt_implementation(vocab_size, batch, int(vocab_size / 10))


if __name__ == "__main__":
  test.main()
