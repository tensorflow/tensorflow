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

from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class QuantizeTest(test_util.TensorFlowTestCase):

  # We have a lot of other tests that test the details of the rewrite, here we
  # just the specific features of the quantize_graph API.
  def testReturnedElementsTraining(self):
    graph = ops.Graph()
    with graph.as_default():
      a = constant_op.constant(1.0)
      b = variables.Variable(2.0)
      c = a + b
    elements = [a, b, c.op]
    for element in elements:
      print(element)
    q_graph, returned_elements = quantize_graph.create_training_graph(
        graph, elements=elements)
    # Make sure q_graph is different from graph.
    self.assertTrue(graph != q_graph)
    # Check that the returned elements are part of the new graph.
    for returned_element in returned_elements:
      self.assertEqual(q_graph, returned_element.graph)
    # Check that the elements match with the one from the input graph.
    for element, returned_element in zip(elements, returned_elements):
      self.assertEqual(element.name, returned_element.name)

  # We have a lot of other tests that test the details of the rewrite, here we
  # just the specific features of the quantize_graph API.
  def testReturnedElementsEval(self):
    graph = ops.Graph()
    with graph.as_default():
      a = constant_op.constant(1.0)
      b = variables.Variable(2.0)
      c = a + b
    elements = [a, b, c.op]
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


if __name__ == '__main__':
  googletest.main()
