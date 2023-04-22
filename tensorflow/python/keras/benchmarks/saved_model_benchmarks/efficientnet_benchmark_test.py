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
"""Benchmarks for saved model on EfficientNetB7."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.benchmarks.saved_model_benchmarks import saved_model_benchmark_util


class BenchmarkSaveApplications(tf.test.Benchmark):

  def benchmark_save_and_load_efficient_net_b7(self):
    app = tf.keras.applications.EfficientNetB7
    save_result, load_result = (
        saved_model_benchmark_util.save_and_load_benchmark(app))

    self.report_benchmark(
        iters=save_result['iters'],
        wall_time=save_result['wall_time'],
        name=save_result['name'])

    self.report_benchmark(
        iters=load_result['iters'],
        wall_time=load_result['wall_time'],
        name=load_result['name'])

if __name__ == '__main__':
  tf.test.main()
