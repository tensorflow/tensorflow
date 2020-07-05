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
# =============================================================================

"""Tests for tpu_function helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import training_loop


class TPUContextTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testIsInContext(self):
    """Test that control_flow_util can check that we're in a TPU context."""
    z1 = array_ops.identity(1)
    pivot = control_flow_ops.no_op()
    context = tpu.TPUReplicateContext(b"context", 1, pivot=pivot)
    context.Enter()
    z2 = array_ops.identity(1)
    context.Exit()
    self.assertFalse(control_flow_util.IsInXLAContext(z1.op))
    self.assertTrue(control_flow_util.IsInXLAContext(z2.op))


class TPULayerRewriteTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testUsingInfeedQueueWithRegularizer(self):
    """Test that Layer regularizers can reference data created in loops."""

    def make_regularizer(scale):
      return lambda inputs: scale * math_ops.reduce_sum(math_ops.square(inputs))

    def training_step(inputs, scale):
      outputs = convolutional.conv2d(
          inputs,
          filters=16,
          kernel_size=(3, 3),
          data_format="channels_first",
          kernel_regularizer=make_regularizer(scale))
      loss = math_ops.reduce_mean(math_ops.square(outputs))
      return loss.op

    inputs = array_ops.zeros(shape=(128, 32, 32, 16))
    scale = array_ops.ones(shape=())
    infeed = tpu_feed.InfeedQueue(
        tuple_types=[dtypes.float32, dtypes.float32],
        tuple_shapes=[inputs.shape, scale.shape])

    def loop():
      return training_loop.repeat(5, training_step, infeed_queue=infeed)

    # This should not throw an error.
    tpu.rewrite(loop)

class TPUGraphPruneTest(test.TestCase):

  def test_prune_unconnected_ops(self):
    with ops.Graph().as_default():
      a = array_ops.placeholder(dtype=dtypes.float32, name="a")
      b = array_ops.placeholder(dtype=dtypes.float32, name="b")
      constant_op.constant(1.0, name="constant")
      x = variable_scope.get_variable(
          name="x",
          dtype=dtypes.float32,
          shape=[],
          use_resource=True,
          initializer=init_ops.constant_initializer(2.0))
      y = variable_scope.get_variable(
          name="y",
          dtype=dtypes.float32,
          shape=[],
          use_resource=True,
          initializer=init_ops.constant_initializer(3.0))
      math_ops.add(a, b)
      math_ops.add(x, y)
      graph_def = ops.get_default_graph().as_graph_def()

      for node in graph_def.node:
        # Attach a TPU_REPLICATE_ATTR to each node.
        node.attr[tpu._TPU_REPLICATE_ATTR].s = b"0"
        # Rewire placeholder "a" and variable "y" leaving them unconnected.
        for (input_index, node_input) in enumerate(node.input):
          if node_input == "b":
            node.input[input_index] = "constant"
          if node_input == "y":
            node.input[input_index] = "x"

    with ops.Graph().as_default() as graph:
      # Reimport the graph and prune unconnected ops.
      importer.import_graph_def(graph_def)
      tpu.prune_unconnected_ops_from_xla(ops.get_default_graph())

      # Verify that ops "a" and "x" still have TPU_REPLICATE_ATTR.
      a = graph.get_operation_by_name("import/a").get_attr(
          tpu._TPU_REPLICATE_ATTR)
      self.assertEqual(b"0", a)
      x = graph.get_operation_by_name("import/x").get_attr(
          tpu._TPU_REPLICATE_ATTR)
      self.assertEqual(b"0", x)
      # Verify that ops "b" and "y" have TPU_REPLICATE_ATTR removed.
      with self.assertRaisesRegex(
          ValueError,
          "Operation \'import/b\' has no attr named \'_tpu_replicate\'"):
        graph.get_operation_by_name("import/b").get_attr(
            tpu._TPU_REPLICATE_ATTR)
      with self.assertRaisesRegex(
          ValueError,
          "Operation \'import/y\' has no attr named \'_tpu_replicate\'"):
        graph.get_operation_by_name("import/y").get_attr(
            tpu._TPU_REPLICATE_ATTR)

def do_einsum():
  a = array_ops.placeholder(dtype=dtypes.float32, name="a", shape=[2, 3, 4])
  b = array_ops.placeholder(dtype=dtypes.float32, name="b", shape=[2, 4, 5])
  return special_math_ops.einsum("abc,acd->abd", a, b)


def find_einsum(g):
  graph_def = g.as_graph_def()
  for node in graph_def.node:
    if node.op == "Einsum":
      return True
  return False


def find_xla_einsum(g):
  graph_def = g.as_graph_def()
  for node in graph_def.node:
    if node.op == "XlaEinsum":
      return True
  return False


class TPUXlaEinsumTest(test.TestCase):

  def test_tpu_rewrite_uses_xla_einsum(self):
    with ops.Graph().as_default() as g:
      tpu.rewrite(do_einsum)
      self.assertTrue(find_einsum(g) or find_xla_einsum(g))

  def test_default_does_not_use_xla_einsum(self):
    with ops.Graph().as_default() as g:
      do_einsum()
      self.assertFalse(find_xla_einsum(g))


if __name__ == "__main__":
  test.main()
