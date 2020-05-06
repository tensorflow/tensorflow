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

import time

from absl import flags
import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

FLAGS = flags.FLAGS

v2_compat.enable_v2_behavior()


def reduce_fn(state, values):
  """tf.data.Dataset-friendly implementation of mean and variance."""
  k, n, ex, ex2 = state
  # If this is the first iteration, we pick the first value to be 'k',
  # which helps with precision - we assume that k is close to an average
  # value and calculate mean and variance with respect to that.
  k = control_flow_ops.cond(math_ops.equal(n, 0), lambda: values[0], lambda: k)

  sum_v = math_ops.reduce_sum(values, axis=0)
  sum_v2 = math_ops.reduce_sum(math_ops.square(values), axis=0)
  ones = array_ops.ones_like(values, dtype=dtypes.int32)
  batch_size = math_ops.reduce_sum(ones, axis=0)
  batch_size_f = math_ops.cast(batch_size, dtypes.float32)

  ex = 0 + sum_v - math_ops.multiply(batch_size_f, k)
  ex2 = 0 + sum_v2 + math_ops.multiply(
      batch_size_f, (math_ops.square(k) -
                     math_ops.multiply(math_ops.multiply(2.0, k), sum_v)))

  return (k, n + batch_size, ex, ex2)


class BenchmarkAdapt(benchmark.Benchmark):
  """Benchmark adapt."""

  def run_dataset_implementation(self, num_elements, batch_size):
    input_t = keras.Input(shape=(1,))
    layer = normalization.Normalization()
    _ = layer(input_t)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = dataset_ops.Dataset.range(num_elements)
      ds = ds.map(
          lambda x: array_ops.expand_dims(math_ops.cast(x, dtypes.float32), -1))
      ds = ds.batch(batch_size)

      starts.append(time.time())
      # Benchmarked code begins here.
      k, n, ex, ex2 = ds.reduce((0.0, 0, 0.0, 0.0), reduce_fn)
      mean = k.numpy() + ex.numpy() / n.numpy()
      var = (ex2.numpy() - (ex.numpy() * ex.numpy()) / n.numpy()) / (
          n.numpy() - 1)
      layer.set_weights([mean, var])
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts))
    return avg_time

  def bm_adapt_implementation(self, num_elements, batch_size):
    """Test the KPL adapt implementation."""
    input_t = keras.Input(shape=(1,), dtype=dtypes.float32)
    layer = normalization.Normalization()
    _ = layer(input_t)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = dataset_ops.Dataset.range(num_elements)
      ds = ds.map(
          lambda x: array_ops.expand_dims(math_ops.cast(x, dtypes.float32), -1))
      ds = ds.batch(batch_size)

      starts.append(time.time())
      # Benchmarked code begins here.
      layer.adapt(ds)
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts))
    name = "normalization_adapt|%s_elements|batch_%s" % (num_elements,
                                                         batch_size)
    baseline = self.run_dataset_implementation(num_elements, batch_size)
    extras = {
        "tf.data implementation baseline": baseline,
        "delta seconds": (baseline - avg_time),
        "delta percent": ((baseline - avg_time) / baseline) * 100
    }
    self.report_benchmark(
        iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

  def benchmark_vocab_size_by_batch(self):
    for vocab_size in [100, 1000, 10000, 100000, 1000000]:
      for batch in [1, 16, 2048]:
        self.bm_adapt_implementation(vocab_size, batch)


if __name__ == "__main__":
  test.main()
