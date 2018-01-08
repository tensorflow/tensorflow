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
"""Unit tests for the quantize_graph graph rewriting API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class QuantizeGraphTest(test_util.TensorFlowTestCase):

  # We have a lot of other tests that test the details of the rewrite, here we
  # just the specific features of the quantize_graph API.
  def testReturnedElementsTraining(self):
    self._TestReturnElements(True)

  def testReturnedElementsEval(self):
    self._TestReturnElements(False)

  def _TestReturnElements(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      a = constant_op.constant(1.0)
      b = variables.Variable(2.0)
      c = a + b
    elements = [a, b, c.op]
    if is_training:
      q_graph, returned_elements = quantize_graph.create_training_graph(
          graph, elements=elements)
    else:
      q_graph, returned_elements = quantize_graph.create_eval_graph(
          graph, elements=elements)
    # Make sure q_graph is different from graph.
    self.assertTrue(graph != q_graph)
    # Check that the returned elements are part of the new graph.
    for returned_element in returned_elements:
      self.assertEqual(q_graph, returned_element.graph)
    # Check that the elements match with the one from the input graph.
    for element, returned_element in zip(elements, returned_elements):
      self.assertEqual(element.name, returned_element.name)

  def testNoReturnElementsTraining(self):
    self._TestNoReturnElements(True)

  def testNoReturnElementsEval(self):
    self._TestNoReturnElements(False)

  def _TestNoReturnElements(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      a = constant_op.constant(1.0)
      b = variables.Variable(2.0)
      _ = a + b
    if is_training:
      q_graph = quantize_graph.create_training_graph(graph)
    else:
      q_graph = quantize_graph.create_eval_graph(graph)
    # Check that quantize_graph didn't return a tuple when elements isn't
    # provided.
    self.assertTrue(isinstance(q_graph, ops.Graph))
    # Make sure q_graph is different from graph.
    self.assertTrue(graph != q_graph)

  def testDeviceNameTraining(self):
    self._TestDeviceName(True)

  def testDeviceNameEval(self):
    self._TestDeviceName(False)

  def _TestDeviceName(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      inputs = array_ops.zeros((batch_size, height, width, depth))
      conv = layers.conv2d(
          inputs,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=None,
          scope='test')
      _ = nn_ops.relu6(conv)

    device_name = '/job:oink/task:0/device:CPU:0'
    if is_training:
      q_graph = quantize_graph.create_training_graph(
          graph, device_name_or_function=device_name)
    else:
      q_graph = quantize_graph.create_eval_graph(
          graph, device_name_or_function=device_name)

    orig_variable_names = set(
        [v.name for v in graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
    q_variables = q_graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    # Ensure that variables were added.
    self.assertTrue(len(orig_variable_names) < len(q_variables))
    # All added variables should have the specified device name.
    for var in q_variables:
      if var.name not in orig_variable_names:
        self.assertEqual(var.device, device_name)

  def _WeightInit(self, stddev):
    """Returns truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initialized that initialzes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev)


if __name__ == '__main__':
  googletest.main()
