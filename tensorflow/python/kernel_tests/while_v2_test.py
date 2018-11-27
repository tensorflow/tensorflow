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
"""Tests for while_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.control_flow_ops import while_loop as while_loop_v1
from tensorflow.python.ops.while_v2 import while_loop as while_loop_v2
from tensorflow.python.platform import test


class WhileV2Test(test.TestCase, parameterized.TestCase):

  def testSingleLoopVar(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., lambda v: v * v, [x])
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  def testMultipleLoopVarsBasic(self):
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    ret = while_loop_v2(lambda v, _: v < 45., lambda v, w: (v * w, w), [x, y])
    # ret = [x*y^2, y]

    # Note: This is simply d_ret[0]/d_x since d_ret[1]/d_x is 0.
    grad = gradients_impl.gradients(ret, [x])  # [2*x*y]
    with self.cached_session() as sess:
      self.assertSequenceEqual(self.evaluate(ret), [45., 3.])
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  def testMultipleLoopVars(self):
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    #   y = x + y
    ret = while_loop_v2(lambda v, _: v < 45., lambda v, w: (v * w, v + w),
                        [x, y])
    # ret = [y*x**2 + x*y**2, x*y + x + y]

    gradx_0 = gradients_impl.gradients(ret[0], [x])  # [2*x*y + y**2]
    gradx_1 = gradients_impl.gradients(ret[1], [x])  # [y + 1]
    gradx_2 = gradients_impl.gradients(ret, [x])  # [2*x*y + y**2 + 2*y + 1]
    grady_0 = gradients_impl.gradients(ret[0], [y])  # [2*x*y + x**2]
    grady_1 = gradients_impl.gradients(ret[1], [y])  # [x + 1]
    grady_2 = gradients_impl.gradients(ret, [y])  # [2*x*y + x**2 + x + 1]
    with self.cached_session() as sess:
      self.assertSequenceEqual(self.evaluate(ret), [120., 23.])
      self.assertSequenceEqual(self.evaluate(gradx_0), [39.])
      self.assertSequenceEqual(self.evaluate(gradx_1), [4.])
      self.assertSequenceEqual(self.evaluate(gradx_2), [43.])
      self.assertSequenceEqual(self.evaluate(grady_0), [55.])
      self.assertSequenceEqual(self.evaluate(grady_1), [6.])
      self.assertSequenceEqual(self.evaluate(grady_2), [61.])

  def testMultipleWhileLoops(self):
    x = constant_op.constant(2.)
    ret1 = while_loop_v2(lambda v: v < 4., lambda v: v * v, [x])  # x**2
    ret2 = while_loop_v2(lambda v: v < 16., lambda v: v * v, [ret1])  # x**4
    grad = gradients_impl.gradients(ret2, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session() as sess:
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  def testDoubleDerivative(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., lambda v: v**2, [x])  # x**4
    grad = gradients_impl.gradients(ret, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  def testPruning(self):
    x = constant_op.constant(1)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=x.dtype, element_shape=x.shape)

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5

    def Body(x, tl):
      return x + 1, list_ops.tensor_list_push_back(tl, x)

    outputs = while_loop_v1(Cond, Body, [x, tensor_list])

    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(outputs[0])

    def GetOptimizedGraph():
      mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.CopyFrom(
          rewriter_config_pb2.RewriterConfig(
              constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
              memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
      return tf_optimizer.OptimizeGraph(config, mg)

    g = GetOptimizedGraph()
    self.assertEqual(len([n for n in g.node if n.op == "Enter"]), 1)

    stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
    train_op.append(stack)
    g = GetOptimizedGraph()
    self.assertEqual(len([n for n in g.node if n.op == "Enter"]), 2)

  def testCaptureExternalTensorInCond(self):
    x = constant_op.constant(2.)
    y = constant_op.constant(1.)
    ret = while_loop_v2(lambda v: v + y < 9., lambda v: v * 3., [x])
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  def testCaptureExternalTensorInBody(self):
    x = constant_op.constant(2.)
    y = constant_op.constant(3.)
    ret = while_loop_v2(lambda v: v < 8., lambda v: v * y, [x])
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  def testLoopWithTensorListPushBack(self):
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      tl = list_ops.tensor_list_push_back(tl, x)
      tl = list_ops.tensor_list_push_back(tl, constant_op.constant(100.))
      return x**2., tl

    ret = while_loop_v2(Cond, Body, [x, tensor_list])
    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  def testDuplicateAccumulator(self):
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      # There is an accumulator in the loop already so we should not add
      # another.
      tl = list_ops.tensor_list_push_back(tl, x)
      return x**2., tl

    ret = while_loop_v2(Cond, Body, [x, tensor_list])

    for op in ops.get_default_graph().get_operations():
      if op.type == "While":
        while_op = op

    body_graph = while_v2._get_body_graph(while_op)
    # body_graph.inputs: [counter_arg, x_arg, tl_arg, *accumulators]
    x_input_t = body_graph.inputs[1]
    accumulator_count = len(
        [c for c in x_input_t.consumers() if c.type == "TensorListPushBack"])
    self.assertEqual(accumulator_count, 1)

    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @parameterized.named_parameters(
      ("UnknownShape", None),
      ("PartiallyDefinedShape", [None, 2]),
      ("FullyDefinedShape", [1, 2]),
  )
  def testAccumulatorElementShape(self, shape):

    def MatchShape(actual_tensor_shape):
      # Compare the shapes, treating None dimensions as equal. We do not
      # directly check actual_tensor_shape and tf.TensorShape(shape) for
      # equality because tf.Dimension.__eq__ returns None if either dimension is
      # None.
      if shape is None:
        self.assertIsNone(actual_tensor_shape.dims)
      else:
        self.assertListEqual(actual_tensor_shape.as_list(), shape)

    def GetAccumulatorForInputAtIndex(while_op, idx):
      body_graph = while_v2._get_body_graph(while_op)
      y_input_t = body_graph.inputs[idx]
      push_back_node = [c for c in y_input_t.consumers()
                        if c.type == "TensorListPushBack"][0]
      output_idx = body_graph.outputs.index(push_back_node.outputs[0])
      return while_op.outputs[output_idx]

    x = constant_op.constant(2.)
    y = array_ops.placeholder(dtype=dtypes.float32, shape=shape)

    # Forward pass.
    ret = while_loop_v2(lambda v, u: v < 8., lambda v, u: (v * v, u), [x, y])
    while_op = ret[0].op.inputs[0].op
    # Get the TensorList output of While op containing the accumulated values
    # of y.
    # while_op.inputs: [counter_arg, x_arg, y_arg, *accumulators]
    output = GetAccumulatorForInputAtIndex(while_op, 2)
    _, val = list_ops.tensor_list_pop_back(output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

    # Gradient pass.
    grad = gradients_impl.gradients(ret[1], y)
    grad_while_op = grad[0].op.inputs[0].op
    # Get the TensorList output of gradient While op containing the accumulated
    # values of grad_y.
    # grad_while_op.inputs:
    # [counter_arg, total_iters_arg, grad_x_arg, grad_y_arg, *other_args]
    grad_output = GetAccumulatorForInputAtIndex(grad_while_op, 3)
    _, val = list_ops.tensor_list_pop_back(grad_output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

  def _createWhile(self, name):
    """Helper function testDefaultName."""
    output = while_v2.while_loop(lambda i: i < 3, lambda i: i + 1,
                                 [constant_op.constant(0)])
    while_op = output.op.inputs[0].op
    self.assertEqual(while_op.type, "While")
    return while_op

  def testDefaultName(self):
    with ops.Graph().as_default():
      while_op = self._createWhile(None)
      self.assertEqual(while_op.name, "while")
      self.assertRegexpMatches(
          while_op.get_attr("cond").name, r"while_cond_\d*")
      self.assertRegexpMatches(
          while_op.get_attr("body").name, r"while_body_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        while1_op = self._createWhile("")
        self.assertEqual(while1_op.name, "foo/while")
        self.assertRegexpMatches(
            while1_op.get_attr("cond").name, r"foo_while_cond_\d*")
        self.assertRegexpMatches(
            while1_op.get_attr("body").name, r"foo_while_body_\d*")

        while2_op = self._createWhile(None)
        self.assertEqual(while2_op.name, "foo/while_1")
        self.assertRegexpMatches(
            while2_op.get_attr("cond").name, r"foo_while_1_cond_\d*")
        self.assertRegexpMatches(
            while2_op.get_attr("body").name, r"foo_while_1_body_\d*")

  @test_util.enable_control_flow_v2
  def testWhileAndTensorArray(self):
    with self.cached_session() as sess:
      param = constant_op.constant(2.0)
      y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
      # map_fn uses TensorArray internally.
      r = functional_ops.map_fn(lambda x: math_ops.multiply(x, param), y0)
      self.assertAllClose([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], self.evaluate(r))
      r = gradients_impl.gradients(r, param)[0]
      self.assertAllClose(21.0, self.evaluate(r))

  def testNestedWhile(self):
    # Compute sum of geometric progression: n^0 + n^1 + ... + n^m
    # We compute the pow using a while loop.
    n = constant_op.constant(3.)
    m = constant_op.constant(5.)
    sum_of_powers = constant_op.constant(0.)

    def Body(i, previous_sum):
      prod = constant_op.constant(1.)
      return i - 1., previous_sum + while_loop_v2(
          lambda c, _: c > 0, lambda c, v: (c - 1., v * n), [i, prod])[1]

    result = while_loop_v2(lambda i, _: i >= 0, Body, [m, sum_of_powers])[1]
    grad = gradients_impl.gradients(result, [n])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(result), 364.)
      self.assertSequenceEqual(self.evaluate(grad), [547.])

  def testIdentityNodeInBody(self):

    def Body(v):
      v = array_ops.identity(v)
      v = array_ops.identity(v)
      return v * v

    x = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., Body, [x])
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  def testNestedWhileAndTensorArray(self):
    n = constant_op.constant(3.0)

    def Body(row, ta, n):

      def InnerBody(row, col, ta, n):
        # Note: row and col are 1-based.
        ta = ta.write(
            math_ops.cast(n * (row - 1.) + col - 1., dtypes.int32), row * col)
        return row, col + 1., ta, n

      # TODO(b/118457764): Remove n from loop_vars from both loops once fixed.
      ta = while_loop_v2(lambda _, col, _1, n: col <= n, InnerBody,
                         [row, constant_op.constant(1.), ta, n])[2]
      return row + 1., ta, n

    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=9)
    ta = while_loop_v2(lambda row, _, _1: row <= n, Body,
                       [constant_op.constant(1.), ta, n])[1]

    output = array_ops.reshape(ta.stack(), [3, 3])
    self.assertAllEqual(
        self.evaluate(output), [[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    # TODO(b/117675481): This does not work with current TA. Enable with new TA.
    # grad = gradients_impl.gradients(output, [n])
    # self.assertEqual(self.evaluate(grad), 3.5)


def ScalarShape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


if __name__ == "__main__":
  test.main()
