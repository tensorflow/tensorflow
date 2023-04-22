# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.experimental.rejection_resample()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.data.ops import dataset_ops


class RejectionResampleBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.experimental.rejection_resample()`."""

  def benchmark_resample_performance(self):
    init_dist = [0.25, 0.25, 0.25, 0.25]
    target_dist = [0.0, 0.0, 0.0, 1.0]
    num_classes = len(init_dist)
    # We don't need many samples to test a dirac-delta target distribution
    num_samples = 1000
    data_np = np.random.choice(num_classes, num_samples, p=init_dist)
    # Prepare the dataset
    dataset = dataset_ops.Dataset.from_tensor_slices(data_np).repeat()
    # Reshape distribution via rejection sampling.
    dataset = dataset.apply(
        resampling.rejection_resample(
            class_func=lambda x: x,
            target_dist=target_dist,
            initial_dist=init_dist,
            seed=142))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)

    wall_time = self.run_benchmark(
        dataset=dataset,
        num_elements=num_samples,
        iters=10,
        warmup=True)
    resample_time = wall_time * num_samples

    self.report_benchmark(
        iters=10,
        wall_time=resample_time,
        extras={
            "model_name": "rejection_resample.benchmark.1",
            "parameters": "%d" % num_samples,
        },
        name="resample_{}".format(num_samples))


if __name__ == "__main__":
  benchmark_base.test.main()
