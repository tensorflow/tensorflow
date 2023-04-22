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
"""Benchmark suite for KPL and feature column implementations."""
import itertools
import math
import random
import string
import time

import numpy as np

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops as tf_data
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test as tf_test

# This is required as of 3/2021 because otherwise we drop into graph mode.
v2_compat.enable_v2_behavior()


class LayerBenchmark(tf_test.Benchmark):
  """Benchmark the layer forward pass."""

  def report(self, name, keras_time, fc_time, iters):
    """Calculate and report benchmark statistics."""
    extras = {
        "fc_avg_time": fc_time,
        "fc_vs_keras_sec": fc_time - keras_time,
        "fc_vs_keras_pct": ((fc_time - keras_time) / fc_time) * 100,
        "keras_faster_ratio": fc_time / keras_time
    }
    self.report_benchmark(
        iters=iters, wall_time=keras_time, extras=extras, name=name)


class StepTimingCallback(keras.callbacks.Callback):
  """A callback that times non-warmup steps of a Keras predict call."""

  def __init__(self):
    self.t0 = None
    self.steps = 0

  def on_predict_batch_begin(self, batch_index, _):
    if batch_index == 2:
      self.t0 = time.time()
    elif batch_index > 2:
      self.steps += 1

  def on_predict_end(self, _):
    self.tn = time.time()
    self.t_avg = (self.tn - self.t0) / self.steps


def create_data(length, num_entries, max_value, dtype):
  """Create a ragged tensor with random data entries."""
  lengths = (np.random.random(size=num_entries) * length).astype(int)
  total_length = np.sum(lengths)
  values = (np.random.random(size=total_length) * max_value).astype(dtype)
  return ragged_tensor.RaggedTensor.from_row_lengths(values, lengths)


def create_string_data(length,
                       num_entries,
                       vocabulary,
                       pct_oov,
                       oov_string="__OOV__"):
  """Create a ragged tensor with random data entries."""
  lengths = (np.random.random(size=num_entries) * length).astype(int)
  total_length = np.sum(lengths)
  num_oovs = int(pct_oov * total_length)
  values = []
  for _ in range(total_length):
    values.append(random.choice(vocabulary))

  if pct_oov > 0:
    oov_cadence = int(total_length / num_oovs)
    idx = 0
    for _ in range(num_oovs):
      if idx < total_length:
        values[idx] = oov_string
      idx += oov_cadence

  return ragged_tensor.RaggedTensor.from_row_lengths(values, lengths)


def create_vocabulary(vocab_size):
  base = len(string.ascii_letters)
  n = math.ceil(math.log(vocab_size, base))
  vocab = []
  for i in range(1, n + 1):
    for item in itertools.product(string.ascii_letters, repeat=i):
      if len(vocab) >= vocab_size:
        break
      vocab.append("".join(item))
  return vocab


def run_keras(data, model, batch_size, num_runs, steps_per_repeat=100):
  """Benchmark a Keras model."""
  ds = tf_data.Dataset.from_tensor_slices(data).repeat().prefetch(
      tf_data.AUTOTUNE).batch(batch_size).cache()
  steps = 0
  times = []
  for _ in range(num_runs):
    steps += steps_per_repeat
    timer = StepTimingCallback()
    # Benchmarked code begins here.
    model.predict(ds, steps=steps, callbacks=[timer])
    # Benchmarked code ends here.
    times.append(timer.t_avg)
  avg_time = np.mean(times)
  return avg_time


def run_fc(data, fc_fn, batch_size, num_runs, steps_per_repeat=100):
  """Benchmark a Feature Column."""

  ds = tf_data.Dataset.from_tensor_slices(data).repeat().prefetch(
      tf_data.AUTOTUNE).batch(batch_size).cache()

  # Trace the fc_fn
  ds_iter = ds.__iter__()
  fc_fn(next(ds_iter))
  fc_starts = []
  fc_ends = []
  for _ in range(num_runs):
    fc_starts.append(time.time())
    # Benchmarked code begins here.
    for _ in range(steps_per_repeat):
      _ = fc_fn(next(ds_iter))
    # Benchmarked code ends here.
    fc_ends.append(time.time())
  avg_per_step_time = (np.array(fc_ends) -
                       np.array(fc_starts)) / steps_per_repeat
  avg_time = np.mean(avg_per_step_time)
  return avg_time
