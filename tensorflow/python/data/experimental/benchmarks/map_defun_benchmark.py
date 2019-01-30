# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for MapDefunOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import map_defun
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks too.
class MapDefunBenchmark(test.Benchmark):
  """Benchmarks for MapDefunOp."""

  def _run(self, op, name=None, num_iters=3000):
    with session.Session() as sess:
      for _ in range(5):
        sess.run(op)
      start = time.time()
      for _ in range(num_iters):
        sess.run(op)
      end = time.time()
      mean_us = (end - start) * 1e6 / num_iters
      self.report_benchmark(
          name=name,
          iters=num_iters,
          wall_time=mean_us,
          extras={"examples_per_sec": num_iters / (end - start)})

  def benchmarkDefunVsMapFn(self):
    """Benchmarks to compare the performance of MapDefun vs tf.map_fn."""

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def defun(x):
      return array_ops.identity(x)

    def fn(x):
      return array_ops.identity(x)

    base = math_ops.range(100)
    for input_size in [10, 100, 1000, 10000]:
      num_iters = 100000 // input_size
      map_defun_op = map_defun.map_defun(defun, [base], [dtypes.int32], [()])
      map_fn_op = map_fn.map_fn(fn, base)

      self._run(
          map_defun_op, "with_defun_size_%d" % input_size, num_iters=num_iters)
      self._run(
          map_fn_op, "without_defun_size_%d" % input_size, num_iters=num_iters)


if __name__ == "__main__":
  test.main()
