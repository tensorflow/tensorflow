"""Tests for tensorflow.python.client.graph_util."""
import tensorflow.python.platform

from tensorflow.python.client import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import types
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
      var_v = state_ops.variable_op([], dtype=types.float32)
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
      var_v = state_ops.variable_op([], dtype=types.float32)
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
      var_0 = state_ops.variable_op([1], dtype=types.float32)
      with g.device(graph_util.pin_variables_on_cpu):
        var_1 = state_ops.variable_op([1], dtype=types.float32)
      var_2 = state_ops.variable_op([1], dtype=types.float32)
      var_3 = state_ops.variable_op([1], dtype=types.float32)
      with g.device(graph_util.pin_variables_on_cpu):
        var_4 = state_ops.variable_op([1], dtype=types.float32)
        with g.device("/device:GPU:0"):
          var_5 = state_ops.variable_op([1], dtype=types.float32)
        var_6 = state_ops.variable_op([1], dtype=types.float32)

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


if __name__ == "__main__":
  googletest.main()
