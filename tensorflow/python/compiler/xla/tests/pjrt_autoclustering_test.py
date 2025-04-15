# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests single device compilation + autoclustering using the Device API (PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_XLA_FLAGS` environment variable.
"""

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops


class PjrtAutoclusteringTest(test.TestCase):

  def test_xla_compile_and_run_on_gpu_device(self):

    if not test.is_gpu_available() or not test.is_built_with_gpu_support():
      test.skipTest("Test only applicable on GPU")

    @def_function.function
    def arithmetic(x):
      return 2 * x + 1

    @def_function.function
    def conditional(x):
      # cond uses switch and merge, which are not supported by XLA based on
      # https://docs.google.com/spreadsheets/d/1H8AIDdnlyyaWZOYN3WpBVNOmGOS_M8OyF7IA7kL3fjk/edit?resourcekey=0-I-mIp472YuK8FuBa5Zmzmg#gid=139369773
      return cond.cond(math_ops.reduce_sum(x) < 5, lambda: x + x, lambda: x)

    @def_function.function
    def func(x, y):
      return (arithmetic(x) + conditional(y) ** 2) / 2

    i1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    # Simple case: all ops supported by XLA
    with ops.device("/device:GPU:0"):
      with context.collect_graphs(optimized=True) as graphs:
        result = arithmetic(i1)
    self.assertAllClose(result.numpy(), [[3.0, 5.0], [7.0, 9.0]], atol=1e-05)
    graph_ops = [n.op for n in graphs[0].node]
    self.assertContainsSubset(["_XlaCompile", "_XlaRun"], graph_ops)

    # Complex case: includes ops not supported by XLA (switch and merge)
    i2 = constant_op.constant([[5.0, 6.0], [7.0, 8.0]])
    with ops.device("/device:GPU:0"):
      with context.collect_graphs(optimized=True) as graphs:
        result = func(i1, i2)
    self.assertAllClose(result.numpy(), [[14.0, 20.5], [28, 36.5]], atol=1e-05)
    graph_ops = [n.op for n in graphs[0].node]
    self.assertContainsSubset(["_XlaCompile", "_XlaRun"], graph_ops)
    # because of the cond, not all ops can be combined into a single _XlaCompile
    self.assertGreater(graph_ops.count("_XlaCompile"), 1)


if __name__ == "__main__":
  test.main()
