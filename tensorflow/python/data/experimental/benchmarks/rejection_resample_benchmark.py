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

import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


def _time_resampling(data_np, target_dist, init_dist, num_to_sample):  # pylint: disable=missing-docstring
  dataset = dataset_ops.Dataset.from_tensor_slices(data_np).repeat()

  # Reshape distribution via rejection sampling.
  dataset = dataset.apply(
      resampling.rejection_resample(
          class_func=lambda x: x,
          target_dist=target_dist,
          initial_dist=init_dist,
          seed=142))

  get_next = dataset_ops.make_one_shot_iterator(dataset).get_next()

  with session.Session() as sess:
    start_time = time.time()
    for _ in xrange(num_to_sample):
      sess.run(get_next)
    end_time = time.time()

  return end_time - start_time


class RejectionResampleBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.experimental.rejection_resample()`."""

  def benchmarkResamplePerformance(self):
    init_dist = [0.25, 0.25, 0.25, 0.25]
    target_dist = [0.0, 0.0, 0.0, 1.0]
    num_classes = len(init_dist)
    # We don't need many samples to test a dirac-delta target distribution
    num_samples = 1000
    data_np = np.random.choice(num_classes, num_samples, p=init_dist)

    resample_time = _time_resampling(
        data_np, target_dist, init_dist, num_to_sample=1000)

    self.report_benchmark(iters=1000, wall_time=resample_time, name="resample")


if __name__ == "__main__":
  test.main()
