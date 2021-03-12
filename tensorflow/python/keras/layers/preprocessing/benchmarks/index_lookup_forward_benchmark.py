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

import os
import random
import string
import time

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.ops import lookup_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

v2_compat.enable_v2_behavior()


# word_gen creates random sequences of ASCII letters (both lowercase and upper).
# The number of unique strings is ~2,700.
def tensor_gen(batch, num_elements):
  data = []
  for _ in range(batch):
    batch_element = []
    for _ in range(num_elements - 1):
      tok = "".join(random.choice(string.ascii_letters) for i in range(2))
      batch_element.append(tok)
    batch_element.append("")  # Explicitly test the empty string.
    data.append(batch_element)
  return constant_op.constant(data)


def get_vocab():
  vocab = list(
      set([a + b for a in string.ascii_letters for b in string.ascii_letters]))  # pylint:disable=g-complex-comprehension
  vocab.sort()
  return vocab


# This class uses TestCase for get_temp_dir().
class BenchmarkLookup(benchmark.TensorFlowBenchmark):
  """Benchmark the index lookup layer's forward pass."""

  def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
    with gfile.GFile(vocab_path, "w") as writer:
      for vocab in vocab_list:
        writer.write(vocab + "\n")
      writer.flush()
      writer.close()
    return vocab_path

  def run_numpy_implementation(self, data, vocab):
    """Test the python implementation."""
    input_t = keras.Input(shape=(), dtype=dtypes.string)
    layer = index_lookup.IndexLookup(
        vocabulary=vocab,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="OOV",
        dtype=dtypes.string)
    out_t = layer(input_t)
    model = keras.Model(input_t, out_t)
    num_repeats = 5
    starts = []
    ends = []
    _ = model(data)
    for _ in range(num_repeats):
      starts.append(time.time())
      out = model(data)
      ends.append(time.time())
    avg_time = np.mean(np.array(ends) - np.array(starts))
    return avg_time, out

  def bm_adapt_implementation(self, num_elements, batch_size):
    """Test the KPL adapt implementation."""
    vocab = get_vocab()
    vocab_file = self._write_to_temp_file("vocab", vocab)
    vocabulary_initializer = lookup_ops.TextFileInitializer(
        filename=vocab_file,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_index_offset=2)
    input_t = keras.Input(shape=(), dtype=dtypes.string)
    layer = index_lookup.IndexLookup(
        vocabulary=vocabulary_initializer,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="OOV",
        dtype=dtypes.string)
    out_t = layer(input_t)
    model = keras.Model(input_t, out_t)
    num_repeats = 5
    starts = []
    ends = []
    data = tensor_gen(batch_size, num_elements)
    _ = model(data)
    for _ in range(num_repeats):
      starts.append(time.time())
      _ = model(data)
      ends.append(time.time())
    avg_time = np.mean(np.array(ends) - np.array(starts))
    baseline, _ = self.run_numpy_implementation(data, vocab)
    extras = {
        "numpy implementation baseline": baseline,
        "delta seconds": (baseline - avg_time),
        "delta percent": ((baseline - avg_time) / baseline) * 100
    }
    name = "index_lookup_forward|%s_elements|batch_%s" % (num_elements,
                                                          batch_size)
    self.report_benchmark(
        iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

  def benchmark_vocab_size_by_batch(self):
    for tensor_size in [100, 1000, 10000]:
      for batch in [1, 16, 2048]:
        self.bm_adapt_implementation(tensor_size, batch)


if __name__ == "__main__":
  test.main()
