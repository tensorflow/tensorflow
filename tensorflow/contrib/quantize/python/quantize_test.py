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
"""Unit tests for quantizing a Tensorflow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest

conv2d = layers.conv2d


class QuantizeTest(test_util.TensorFlowTestCase):

  def testInsertQuantOpFailsWhenOpsNotConnected(self):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      inputs = array_ops.zeros((batch_size, height, width, depth))
      conv = conv2d(inputs, 32, [5, 5], stride=2, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=None, scope='test')
      relu = nn_ops.relu6(inputs)

    context = quantize._QuantizeContext(graph=graph, weight_bits=8,
                                        weight_narrow_range=True,
                                        activation_bits=8)
    # Inserting a quantization op between two unconnected ops should fail with
    # ValueError.
    with self.assertRaises(ValueError) as err:
      context._InsertQuantOp('test', conv.op, [relu.op], 'FailingQuantOp')
    self.assertEqual(
        str(err.exception), 'Some inputs not quantized for ops: [Relu6]')

  def _WeightInit(self, stddev):
    """Returns truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initialized that initialzes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev)

  def _AssertInputOpsAre(self, op, in_op_names):
    """Asserts that all inputs to op come from in_op_names (disregarding order).

    Args:
      op: Operation to check inputs for.
      in_op_names: List of strings, operations where all op's inputs should
        come from.
    """
    expected_inputs = [in_op_name + ':0' for in_op_name in in_op_names]
    self.assertItemsEqual([t.name for t in op.inputs], expected_inputs)

  def _AssertOutputGoesToOps(self, op, graph, out_op_names):
    """Asserts that outputs from op go to out_op_names (and perhaps others).

    Args:
      op: Operation to check outputs for.
      graph: Graph where output operations are located.
      out_op_names: List of strings, operations where op's outputs should go.
    """
    for out_op_name in out_op_names:
      out_op = graph.get_operation_by_name(out_op_name)
      self.assertIn(op.outputs[0].name, [str(t.name) for t in out_op.inputs])

if __name__ == '__main__':
  googletest.main()
