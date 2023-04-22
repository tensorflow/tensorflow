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
"""Benchmark for KPL implementation of categorical hash columns with varying-length inputs."""

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager.def_function import function as tf_function
from tensorflow.python.feature_column import feature_column_v2 as fcv2
from tensorflow.python.framework import dtypes as dt
from tensorflow.python.keras.layers.preprocessing import hashing
from tensorflow.python.keras.layers.preprocessing.benchmarks import feature_column_benchmark as fc_bm
from tensorflow.python.platform import test as tf_test

# This is required as of 3/2021 because otherwise we drop into graph mode.
v2_compat.enable_v2_behavior()

NUM_REPEATS = 10
BATCH_SIZES = [32, 256]


def embedding_varlen(batch_size, max_length):
  """Benchmark a variable-length embedding."""
  # Data and constants.

  num_buckets = 10000
  vocab = fc_bm.create_vocabulary(32768)
  data = fc_bm.create_string_data(
      max_length, batch_size * NUM_REPEATS, vocab, pct_oov=0.0)

  # Keras implementation
  model = keras.Sequential()
  model.add(
      keras.Input(
          shape=(max_length,), name="data", ragged=True, dtype=dt.string))
  model.add(hashing.Hashing(num_buckets))

  # FC implementation
  fc = fcv2.categorical_column_with_hash_bucket("data", num_buckets)

  # Wrap the FC implementation in a tf.function for a fair comparison
  @tf_function()
  def fc_fn(tensors):
    fc.transform_feature(fcv2.FeatureTransformationCache(tensors), None)

  # Benchmark runs
  keras_data = {"data": data}
  k_avg_time = fc_bm.run_keras(keras_data, model, batch_size, NUM_REPEATS)

  fc_data = {"data": data.to_sparse()}
  fc_avg_time = fc_bm.run_fc(fc_data, fc_fn, batch_size, NUM_REPEATS)

  return k_avg_time, fc_avg_time


class BenchmarkLayer(fc_bm.LayerBenchmark):
  """Benchmark the layer forward pass."""

  def benchmark_layer(self):
    for batch in BATCH_SIZES:
      name = "hash|varlen|batch_%s" % batch
      k_time, f_time = embedding_varlen(batch_size=batch, max_length=256)
      self.report(name, k_time, f_time, NUM_REPEATS)


if __name__ == "__main__":
  tf_test.main()
