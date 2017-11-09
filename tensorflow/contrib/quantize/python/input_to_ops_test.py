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
"""Unit tests for InputToOps class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class InputToOpsTest(test_util.TensorFlowTestCase):

  def testNoConsumerOperations(self):
    graph = ops.Graph()
    with graph.as_default():
      input_tensor = array_ops.zeros((1, 2, 3, 4))

    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_operations = input_to_ops_map.ConsumerOperations(input_tensor.op)

    self.assertEqual(0, len(consumer_operations))

  def testOneConsumerOperation(self):
    graph = ops.Graph()
    with graph.as_default():
      input_tensor = array_ops.zeros((1, 2, 3, 4))
      output_tensor = nn_ops.relu6(input_tensor)

    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_operations = input_to_ops_map.ConsumerOperations(input_tensor.op)

    self.assertEqual(consumer_operations, {output_tensor.op})

  def testSeveralConsumerOperations(self):
    graph = ops.Graph()
    with graph.as_default():
      input_tensor = array_ops.zeros((1, 2, 3, 4))
      output_tensor_1 = nn_ops.relu6(input_tensor)
      output_tensor_2 = input_tensor + output_tensor_1
      output_tensor_3 = input_tensor * output_tensor_2

    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_operations = input_to_ops_map.ConsumerOperations(input_tensor.op)

    self.assertEqual(consumer_operations,
                     {output_tensor_1.op, output_tensor_2.op,
                      output_tensor_3.op})

if __name__ == '__main__':
  googletest.main()
