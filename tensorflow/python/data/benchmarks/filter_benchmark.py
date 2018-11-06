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
"""Benchmarks for `tf.data.Dataset.filter()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks.
class FilterBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.Dataset.filter()`."""

  def _benchmark(self, predicate, name):
    with ops.Graph().as_default():
      dataset = (
          dataset_ops.Dataset.from_tensors(True).repeat(None).filter(predicate))
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      next_element = iterator.get_next()

      with session.Session() as sess:
        for _ in range(5):
          sess.run(next_element.op)
        deltas = []
        for _ in range(100):
          start = time.time()
          for _ in range(100):
            sess.run(next_element.op)
          end = time.time()
          deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100
        print("Filter dataset using %s. Median wall time: %f" %
              (name, median_wall_time))
        self.report_benchmark(
            iters=100,
            wall_time=median_wall_time,
            name=name)

  def benchmarkSimpleFunction(self):
    self._benchmark(array_ops.identity, "simple_function")

  def benchmarkReturnComponentOptimization(self):
    self._benchmark(lambda x: x, "return_component")


if __name__ == "__main__":
  test.main()
