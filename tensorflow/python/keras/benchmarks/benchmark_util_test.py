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
"""Tests for benchmark utitilies."""

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


class BenchmarkUtilTest(tf.test.TestCase):

  def test_get_benchmark_name(self):
    name = "benchmark_layer_call__Conv2D_small_shape"
    expected = ["Conv2D", "small", "shape"]
    out = benchmark_util.get_benchmark_name(name)
    self.assertAllEqual(out, expected)

  def test_generate_benchmark_params_cpu_gpu(self):
    adam_opt = tf.keras.optimizers.Adam()
    sgd_opt = tf.keras.optimizers.SGD()
    params = [
        ("Adam", adam_opt, 10),
        ("SGD", sgd_opt, 10),
    ]
    expected = [
        ("Adam_CPU", adam_opt, 10),
        ("SGD_CPU", sgd_opt, 10),
        ("Adam_GPU", adam_opt, 10),
        ("SGD_GPU", sgd_opt, 10),
    ]

    out = benchmark_util.generate_benchmark_params_cpu_gpu(params)
    self.assertAllEqual(out, expected)


if __name__ == "__main__":
  tf.test.main()
