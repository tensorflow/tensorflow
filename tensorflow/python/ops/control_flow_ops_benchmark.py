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
"""Benchmark for control flow ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class CondWithManyIntermediatesBenchmark(test.Benchmark):
  """Checks the runtime performance of outputting all intermediates."""

  NUM_INTERMEDIATES = 1000
  NUM_ITERS = 500
  NUM_WARM_UP_ITERS = 50

  def _create_cond(self, x):

    def branch_fn():
      # Use a random value so the adds can't be constant folded.
      return x + sum(random_ops.random_normal([])
                     for _ in range(self.NUM_INTERMEDIATES))

    # Use a dynamic predicate to make sure the cond isn't constant folded.
    return control_flow_ops.cond(math_ops.not_equal(x, -1),
                                 branch_fn, lambda: 0.0)

  def _benchmark_defun(self):
    """Benchmarks cond in a defun."""

    @function.defun
    def cond_fn(x):
      return self._create_cond(x)

    # Warm up
    for _ in range(self.NUM_WARM_UP_ITERS):
      cond_fn(0.0)

    start_time = time.time()

    for _ in range(self.NUM_ITERS):
      cond_fn(0.0)

    self.report_benchmark(
        wall_time=time.time() - start_time,
        iters=self.NUM_ITERS)

  def _benchmark_graph(self):
    """Benchmarks cond in legacy graph mode."""
    with context.graph_mode():
      with ops.Graph().as_default():
        x = array_ops.placeholder(dtypes.float32)
        cond_val = self._create_cond(x)

        with session.Session() as sess:
          cond_fn = sess.make_callable(cond_val, [x])

          # Warm up
          for _ in range(self.NUM_WARM_UP_ITERS):
            cond_fn(0.0)

          start_time = time.time()

          for _ in range(self.NUM_ITERS):
            cond_fn(0.0)

          self.report_benchmark(
              wall_time=time.time() - start_time,
              iters=self.NUM_ITERS)

  def benchmark_cond_v1_defun(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v2_defun(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v1_graph(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v2_graph(self):
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
