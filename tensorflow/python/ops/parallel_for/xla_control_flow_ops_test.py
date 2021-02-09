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
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PForTest(PForTestCase):

  def __init__(self, method_name="runTest"):
    super(PForTest, self).__init__(method_name)
    context.context().enable_xla_devices()

  def test_xla_einsum(self):
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

  def test_function_jit_compile(self):

    def compute(x):
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    @def_function.function(jit_compile=True)
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

    @def_function.function(jit_compile=True)
    def f():

      def loop_fn(i, pfor_config):
        x_i = array_ops.gather(x, i)
        return x_i - pfor_config.reduce_mean(x_i)

      return pfor_control_flow_ops.pfor(loop_fn, 8)

    output = f()
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)


def _make_unstacked(cond, body, pfor_config):

  def _cond(*args):
    return math_ops.reduce_any(pfor_config.reduce_concat(args[0]))

  def _body(*args):
    not_done = args[0]
    args = args[1:]
    not_done = math_ops.logical_and(not_done, cond(*args))
    outputs = body(*args)
    return (not_done,) + tuple(
        array_ops.where_v2(not_done, x, y) for x, y in zip(outputs, args))

  return _cond, _body


@test_util.run_all_in_graph_and_eager_modes
class WhileV2Test(PForTestCase):

  def setUp(self):
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV2Test, self).setUp()

  def tearDown(self):
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV2Test, self).tearDown()

  def _test_loop_fn(self, loop_fn, iters, force_xla=False):

    def f():
      return pfor_control_flow_ops.pfor(loop_fn, iters)

    @def_function.function
    def jit_f():
      with jit.experimental_jit_scope():
        return f()

    out = f()
    jit_out = jit_f()
    self.run_and_assert_equal(out, jit_out)
    # TODO(agarwal): The following may complain about uncompilable nodes. Hence
    # these are currently not enabled for all tests.
    if force_xla:
      out_exp_compile_f = def_function.function(jit_compile=True)(f)()
      self.run_and_assert_equal(out, out_exp_compile_f)
      out_xla_compile_f = xla.compile(f, inputs=[])
      self.run_and_assert_equal(out, out_xla_compile_f)

  def test_stateless_while(self):
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      return control_flow_ops.while_loop(
          lambda j, _: j < lengths_i,
          lambda j, t: (j + 1, t + array_ops.gather(x_i, j)),
          [0, 0.])

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_variable(self):
    v = resource_variable_ops.ResourceVariable(5.)

    def loop_fn(_):
      _, output = control_flow_ops.while_loop(
          lambda j, x: j < 4,
          lambda j, x: (j + 1, x + v),
          [0, 0.])
      return output

    self._test_loop_fn(loop_fn, 3)

  def test_while_unstacked_condition(self):

    def loop_fn(i):
      return control_flow_ops.while_loop(
          lambda j, x: j < 4,
          lambda j, x: (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3, force_xla=True)

  def test_while_force_unstacked_condition(self):
    # The while_loop in this setup is similar to the one in test_stateless_while
    # whose condition is loop variant. However here we wrap the cond and body of
    # the loop in a way that makes the while_loop condition pfor loop invariant.
    # This allows xla compilation to work since the vectorized code no longer
    # needs to perform dynamic partitioning of the inputs.
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i, pfor_config):
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      def _cond(j, _):
        return j < lengths_i

      def _body(j, t):
        return (j + 1, t + array_ops.gather(x_i, j))

      cond, body = _make_unstacked(_cond, _body, pfor_config)
      return control_flow_ops.while_loop(
          cond,
          body,
          [True, 0, 0.])

    self._test_loop_fn(loop_fn, 3, force_xla=True)


if __name__ == "__main__":
  test.main()
