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
"""XLA tests for pfor."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tf2xla.python import xla as xla_ops
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PForTest(PForTestCase):

  def test_einsum(self):
    num_loop = 10
    x_series = random_ops.random_uniform([num_loop, 9, 9])
    y_series = random_ops.random_uniform([num_loop, 9, 1])

    def loop_fn(i):
      x = array_ops.gather(x_series, 0)  # invariant.
      y = array_ops.gather(y_series, 0)  # invariant.
      x_i = array_ops.gather(x_series, i)
      y_i = array_ops.gather(y_series, i)
      z1 = xla_ops.einsum(x_i, y, "ab,bc->ac")
      z2 = xla_ops.einsum(x, y_i, "ab,bc->ac")
      z3 = xla_ops.einsum(x, y, "ab,bc->ac")
      z4 = xla_ops.einsum(x_i, y_i, "ab,bc->ac")
      z5 = xla_ops.einsum(y_i, x_i, "cd,ce->de")  # Includes transpose.
      outputs = [z1, z2, z3, z4, z5]
      return outputs

    self._test_loop_fn(loop_fn, num_loop)

  def test_xla(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    def vectorized_compute(x):
      return pfor_control_flow_ops.vectorized_map(compute, x)

    result = xla.compile(
        vectorized_compute, inputs=[array_ops.ones((10, 5, 3))])
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_function_experimental_compile(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    @def_function.function(experimental_compile=True)
    def vectorized_compute(x):
      return pfor_control_flow_ops.vectorized_map(compute, x)

    result = vectorized_compute(array_ops.ones((10, 5, 3)))
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_xla_while_loop(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    def vectorized_compute(x, i):
      inp = array_ops.gather(x, i)
      output = pfor_control_flow_ops.vectorized_map(compute, inp)
      output.set_shape([5, 1])
      return output

    def while_compute(x):
      return control_flow_ops.while_loop_v2(
          lambda i, _: i < 10,
          lambda i, y: (i + 1, y + vectorized_compute(x, i)),
          (0, array_ops.zeros([5, 1])))[1]

    result = xla.compile(while_compute, inputs=[array_ops.ones((10, 5, 3))])
    expected = array_ops.ones([5, 1]) * 10
    self.run_and_assert_equal(expected, result)

  def test_reduce_mean(self):
    x = random_ops.random_uniform([8, 3])

    @def_function.function(experimental_compile=True)
    def f():

      def loop_fn(i, pfor_config):
        x_i = array_ops.gather(x, i)
        return x_i - pfor_config.reduce_mean(x_i)

      return pfor_control_flow_ops.pfor(loop_fn, 8)

    output = f()
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)


if __name__ == '__main__':
  test.main()
