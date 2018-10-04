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

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
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
      self.assertEqual(sess.run(ret), 16.)
      self.assertSequenceEqual(sess.run(grad), [32.])

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
      self.assertSequenceEqual(sess.run(ret), [45., 3.])
      self.assertSequenceEqual(sess.run(grad), [9.])

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
      self.assertSequenceEqual(sess.run(ret), [120., 23.])
      self.assertSequenceEqual(sess.run(gradx_0), [39.])
      self.assertSequenceEqual(sess.run(gradx_1), [4.])
      self.assertSequenceEqual(sess.run(gradx_2), [43.])
      self.assertSequenceEqual(sess.run(grady_0), [55.])
      self.assertSequenceEqual(sess.run(grady_1), [6.])
      self.assertSequenceEqual(sess.run(grady_2), [61.])

  def testMultipleWhileLoops(self):
    x = constant_op.constant(2.)
    ret1 = while_loop_v2(lambda v: v < 4., lambda v: v * v, [x])  # x**2
    ret2 = while_loop_v2(lambda v: v < 16., lambda v: v * v, ret1)  # x**4
    grad = gradients_impl.gradients(ret2, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session() as sess:
      self.assertSequenceEqual(sess.run(grad), [32.])
      self.assertSequenceEqual(sess.run(grad_grad), [48.])

  def testDoubleDerivative(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., lambda v: v**2, [x])  # x**4
    grad = gradients_impl.gradients(ret, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret), 16.)
      self.assertSequenceEqual(sess.run(grad), [32.])
      self.assertSequenceEqual(sess.run(grad_grad), [48.])

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
      rewriter_config = rewriter_config_pb2.RewriterConfig(
          constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
      return tf_optimizer.OptimizeGraph(rewriter_config, mg)

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
      self.assertEqual(sess.run(ret), 18.)
      self.assertSequenceEqual(sess.run(grad), [9.])

  def testCaptureExternalTensorInBody(self):
    x = constant_op.constant(2.)
    y = constant_op.constant(3.)
    ret = while_loop_v2(lambda v: v < 8., lambda v: v * y, [x])
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret), 18.)
      self.assertSequenceEqual(sess.run(grad), [9.])

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
      self.assertSequenceEqual(sess.run(grad), [32.])

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
      self.assertSequenceEqual(sess.run(grad), [32.])

  @parameterized.named_parameters(
      ("UnknownShape", None),
      ("PartiallyDefinedShape", [None, 2]),
      ("FullyDefinedShape", [1, 2]),
  )
  def testTensorListOutputElementShape(self, shape):

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
    while_op = ret[0].op
    # Get the TensorList output of While op containing the accumulated values
    # of y.
    # while_op.inputs: [counter_arg, x_arg, y_arg, *accumulators]
    output = GetAccumulatorForInputAtIndex(while_op, 2)
    _, val = list_ops.tensor_list_pop_back(output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

    # Gradient pass.
    grad = gradients_impl.gradients(ret[1], y)
    grad_while_op = grad[0].op
    # Get the TensorList output of gradient While op containing the accumulated
    # values of grad_y.
    # grad_while_op.inputs:
    # [counter_arg, total_iters_arg, grad_x_arg, grad_y_arg, *other_args]
    grad_output = GetAccumulatorForInputAtIndex(grad_while_op, 4)
    _, val = list_ops.tensor_list_pop_back(grad_output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)


def ScalarShape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


if __name__ == "__main__":
  test.main()
