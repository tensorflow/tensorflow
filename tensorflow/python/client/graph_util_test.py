# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.python.client.graph_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import tensorflow as tf

from tensorflow.python.client import graph_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_ops
# pylint: disable=unused-import
from tensorflow.python.ops import math_ops
# pylint: enable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import googletest


class DeviceFunctionsTest(googletest.TestCase):

  def testPinToCpu(self):
    with ops.Graph().as_default() as g, g.device(graph_util.pin_to_cpu):
      const_a = constant_op.constant(5.0)
      const_b = constant_op.constant(10.0)
      add_c = const_a + const_b
      var_v = state_ops.variable_op([], dtype=dtypes.float32)
      assign_c_to_v = state_ops.assign(var_v, add_c)
      const_string = constant_op.constant("on a cpu")
      dynamic_stitch_int_result = data_flow_ops.dynamic_stitch(
          [[0, 1, 2], [2, 3]], [[12, 23, 34], [1, 2]])
      dynamic_stitch_float_result = data_flow_ops.dynamic_stitch(
          [[0, 1, 2], [2, 3]], [[12.0, 23.0, 34.0], [1.0, 2.0]])
    self.assertEqual(const_a.device, "/device:CPU:0")
    self.assertEqual(const_b.device, "/device:CPU:0")
    self.assertEqual(add_c.device, "/device:CPU:0")
    self.assertEqual(var_v.device, "/device:CPU:0")
    self.assertEqual(assign_c_to_v.device, "/device:CPU:0")
    self.assertEqual(const_string.device, "/device:CPU:0")
    self.assertEqual(dynamic_stitch_int_result.device, "/device:CPU:0")
    self.assertEqual(dynamic_stitch_float_result.device, "/device:CPU:0")

  def testPinRequiredOpsOnCPU(self):
    with ops.Graph().as_default() as g, g.device(
        graph_util.pin_variables_on_cpu):
      const_a = constant_op.constant(5.0)
      const_b = constant_op.constant(10.0)
      add_c = const_a + const_b
      var_v = state_ops.variable_op([], dtype=dtypes.float32)
      assign_c_to_v = state_ops.assign(var_v, add_c)
      dynamic_stitch_int_result = data_flow_ops.dynamic_stitch(
          [[0, 1, 2], [2, 3]], [[12, 23, 34], [1, 2]])
      dynamic_stitch_float_result = data_flow_ops.dynamic_stitch(
          [[0, 1, 2], [2, 3]], [[12.0, 23.0, 34.0], [1.0, 2.0]])
      # Non-variable ops shuld not specify a device
      self.assertEqual(const_a.device, None)
      self.assertEqual(const_b.device, None)
      self.assertEqual(add_c.device, None)
      # Variable ops specify a device
      self.assertEqual(var_v.device, "/device:CPU:0")
      self.assertEqual(assign_c_to_v.device, "/device:CPU:0")

  def testTwoDeviceFunctions(self):
    with ops.Graph().as_default() as g:
      var_0 = state_ops.variable_op([1], dtype=dtypes.float32)
      with g.device(graph_util.pin_variables_on_cpu):
        var_1 = state_ops.variable_op([1], dtype=dtypes.float32)
      var_2 = state_ops.variable_op([1], dtype=dtypes.float32)
      var_3 = state_ops.variable_op([1], dtype=dtypes.float32)
      with g.device(graph_util.pin_variables_on_cpu):
        var_4 = state_ops.variable_op([1], dtype=dtypes.float32)
        with g.device("/device:GPU:0"):
          var_5 = state_ops.variable_op([1], dtype=dtypes.float32)
        var_6 = state_ops.variable_op([1], dtype=dtypes.float32)

    self.assertEqual(var_0.device, None)
    self.assertEqual(var_1.device, "/device:CPU:0")
    self.assertEqual(var_2.device, None)
    self.assertEqual(var_3.device, None)
    self.assertEqual(var_4.device, "/device:CPU:0")
    self.assertEqual(var_5.device, "/device:GPU:0")
    self.assertEqual(var_6.device, "/device:CPU:0")

  def testExplicitDevice(self):
    with ops.Graph().as_default() as g:
      const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/job:ps"):
        const_5 = constant_op.constant(5.0)

    self.assertEqual(const_0.device, None)
    self.assertEqual(const_1.device, "/device:GPU:0")
    self.assertEqual(const_2.device, "/device:GPU:1")
    self.assertEqual(const_3.device, "/device:CPU:0")
    self.assertEqual(const_4.device, "/device:CPU:1")
    self.assertEqual(const_5.device, "/job:ps")

  def testDefaultDevice(self):
    with ops.Graph().as_default() as g, g.device(
        graph_util.pin_variables_on_cpu):
      with g.device("/job:ps"):
        const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/replica:0"):
        const_5 = constant_op.constant(5.0)

    self.assertEqual(const_0.device, "/job:ps")
    self.assertEqual(const_1.device, "/device:GPU:0")
    self.assertEqual(const_2.device, "/device:GPU:1")
    self.assertEqual(const_3.device, "/device:CPU:0")
    self.assertEqual(const_4.device, "/device:CPU:1")
    self.assertEqual(const_5.device, "/replica:0")

  def testExtractSubGraph(self):
    graph_def = tf.GraphDef()
    n1 = graph_def.node.add()
    n1.name = "n1"
    n1.input.extend(["n5"])
    n2 = graph_def.node.add()
    n2.name = "n2"
    # Take the first output of the n1 node as the input.
    n2.input.extend(["n1:0"])
    n3 = graph_def.node.add()
    n3.name = "n3"
    # Add a control input (which isn't really needed by the kernel, but
    # rather to enforce execution order between nodes).
    n3.input.extend(["^n2"])
    n4 = graph_def.node.add()
    n4.name = "n4"

    # It is fine to have a loops in the graph as well.
    n5 = graph_def.node.add()
    n5.name = "n5"
    n5.input.extend(["n1"])

    sub_graph = graph_util.extract_sub_graph(graph_def, ["n3"])
    self.assertEqual("n1", sub_graph.node[0].name)
    self.assertEqual("n2", sub_graph.node[1].name)
    self.assertEqual("n3", sub_graph.node[2].name)
    self.assertEqual("n5", sub_graph.node[3].name)


if __name__ == "__main__":
  googletest.main()
