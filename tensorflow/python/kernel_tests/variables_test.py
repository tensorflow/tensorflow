# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tf.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables


class VariablesTestCase(tf.test.TestCase):

  def testInitialization(self):
    with self.test_session():
      var0 = tf.Variable(0.0)
      self.assertEqual("Variable:0", var0.name)
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.get_shape())

      var1 = tf.Variable(1.1)
      self.assertEqual("Variable_1:0", var1.name)
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.get_shape())

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        var0.eval()

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        var1.eval()

      tf.initialize_all_variables().run()

      self.assertAllClose(0.0, var0.eval())
      self.assertAllClose(1.1, var1.eval())

  def testInitializationOrder(self):
    with self.test_session():
      rnd = tf.Variable(random_ops.random_uniform([3, 6]), name="rnd")
      self.assertEqual("rnd:0", rnd.name)
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.get_shape())

      dep = tf.Variable(rnd.initialized_value(), name="dep")
      self.assertEqual("dep:0", dep.name)
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.get_shape())

      # Currently have to set the shape manually for Add.
      added_val = rnd.initialized_value() + dep.initialized_value() + 2.0
      added_val.set_shape(rnd.get_shape())

      depdep = tf.Variable(added_val, name="depdep")
      self.assertEqual("depdep:0", depdep.name)
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.get_shape())

      tf.initialize_all_variables().run()

      self.assertAllClose(rnd.eval(), dep.eval())
      self.assertAllClose(rnd.eval() + dep.eval() + 2.0,
                          depdep.eval())

  def testIterable(self):
    with self.assertRaisesRegexp(TypeError, "not iterable"):
      for _ in tf.Variable(0.0):
        pass
    with self.assertRaisesRegexp(TypeError, "not iterable"):
      for _ in tf.Variable([0.0, 1.0]):
        pass

  def testAssignments(self):
    with self.test_session():
      var = tf.Variable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      tf.initialize_all_variables().run()
      self.assertAllClose(0.0, var.eval())

      self.assertAllClose(1.0, plus_one.eval())
      self.assertAllClose(1.0, var.eval())

      self.assertAllClose(-1.0, minus_one.eval())
      self.assertAllClose(-1.0, var.eval())

      self.assertAllClose(4.0, four.eval())
      self.assertAllClose(4.0, var.eval())

  def _countUpToTest(self, dtype):
    with self.test_session():
      zero = tf.constant(0, dtype=dtype)
      var = tf.Variable(zero)
      count_up_to = var.count_up_to(3)

      tf.initialize_all_variables().run()
      self.assertEqual(0, var.eval())

      self.assertEqual(0, count_up_to.eval())
      self.assertEqual(1, var.eval())

      self.assertEqual(1, count_up_to.eval())
      self.assertEqual(2, var.eval())

      self.assertEqual(2, count_up_to.eval())
      self.assertEqual(3, var.eval())

      with self.assertRaisesOpError("Reached limit of 3"):
        count_up_to.eval()
      self.assertEqual(3, var.eval())

      with self.assertRaisesOpError("Reached limit of 3"):
        count_up_to.eval()
      self.assertEqual(3, var.eval())

  def testCountUpToInt32(self):
    self._countUpToTest(tf.int32)

  def testCountUpToInt64(self):
    self._countUpToTest(tf.int64)

  def testControlDepsNone(self):
    with self.test_session():
      c = tf.constant(1.0)
      with tf.control_dependencies([c]):
        # d get the control dep.
        d = tf.constant(2.0)
        # variables do not.
        var_x = tf.Variable(2.0)
        # initialized_value do not either.
        inited_x = var_x.initialized_value()
      self.assertEqual([c.op], d.op.control_inputs)
      self.assertEqual([], var_x.initializer.control_inputs)
      self.assertEqual([], var_x.value().op.control_inputs)
      self.assertEqual([], var_x.ref().op.control_inputs)
      self.assertEqual([var_x.initializer], inited_x.op.control_inputs)

  def testControlFlow(self):
    with self.test_session() as sess:
      v0 = tf.Variable(0, name="v0")
      var_dict = {}
      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        v1 = tf.Variable(1, name="v1")
        var_dict["v1"] = v1
        return v1 + v0
      def var_in_else_clause():
        v2 = tf.Variable(2, name="v2")
        var_dict["v2"] = v2
        return v2 + v0
      add = control_flow_ops.cond(tf.less(v0, 10),
                                  var_in_then_clause,
                                  var_in_else_clause)
      v1 = var_dict["v1"]
      v2 = var_dict["v2"]
      # We should be able to initialize and run v1 and v2 without initializing
      # v0, even if the variable was created with a control dep on v0.
      sess.run(v1.initializer)
      self.assertEqual([1], sess.run(v1))
      sess.run(v2.initializer)
      self.assertEqual([2], sess.run(v2))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      sess.run(v0.initializer)
      sess.run(add)

  def testUseVariableAsTensor(self):
    with self.test_session():
      var_x = tf.Variable(2.0)
      var_y = tf.Variable(3.0)
      tf.initialize_all_variables().run()
      self.assertAllClose(2.0, var_x.eval())
      self.assertAllClose(3.0, var_y.eval())
      self.assertAllClose(5.0, tf.add(var_x, var_y).eval())

  def testZeroSizeVarSameAsConst(self):
    with self.test_session():
      zero_size_var = tf.Variable(tf.zeros([0, 2]))
      zero_size_const = tf.ones([2, 0])
      variable_mul = tf.matmul(zero_size_const, zero_size_var)
      const_mul = tf.matmul(zero_size_const, zero_size_const, transpose_b=True)
      tf.initialize_all_variables().run()
      variable_output = variable_mul.eval()
      self.assertAllClose(const_mul.eval(), variable_output)
      self.assertAllClose([[0., 0.], [0., 0.]], variable_output)

  def testCachingDevice(self):
    with self.test_session():
      var = tf.Variable(2.0)
      self.assertEqual(var.device, var.value().device)
      self.assertEqual(var.device, var.initialized_value().device)

      var_cached = tf.Variable(2.0, caching_device="/job:foo")
      self.assertFalse(var_cached.device.startswith("/job:foo"))
      self.assertTrue(var_cached.value().device.startswith("/job:foo"))
      self.assertTrue(
          var_cached.initialized_value().device.startswith("/job:foo"))

  def testCollections(self):
    with self.test_session():
      var_x = tf.Variable(2.0)
      var_y = tf.Variable(2.0, trainable=False)
      var_z = tf.Variable(2.0, trainable=True)
      var_t = tf.Variable(
          2.0, trainable=True,
          collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                       tf.GraphKeys.VARIABLES])
      self.assertEqual([var_x, var_y, var_z, var_t], tf.all_variables())
      self.assertEqual([var_x, var_z, var_t], tf.trainable_variables())

  def testOperators(self):
    with self.test_session():
      var_f = tf.Variable([2.0])
      add = var_f + 0.0
      radd = 1.0 + var_f
      sub = var_f - 1.0
      rsub = 1.0 - var_f
      mul = var_f * 10.0
      rmul = 10.0 * var_f
      div = var_f / 10.0
      rdiv = 10.0 / var_f
      lt = var_f < 3.0
      rlt = 3.0 < var_f
      le = var_f <= 2.0
      rle = 2.0 <= var_f
      gt = var_f > 3.0
      rgt = 3.0 > var_f
      ge = var_f >= 2.0
      rge = 2.0 >= var_f
      neg = -var_f
      abs_v = abs(var_f)

      var_i = tf.Variable([20])
      mod = var_i % 7
      rmod = 103 % var_i

      var_b = tf.Variable([True, False])
      and_v = operator.and_(var_b, [True, True])
      or_v = operator.or_(var_b, [False, True])
      xor_v = operator.xor(var_b, [False, False])
      invert_v = ~var_b

      rnd = np.random.rand(4, 4).astype("f")
      var_t = tf.Variable(rnd)
      slice_v = var_t[2, 0:0]

      tf.initialize_all_variables().run()
      self.assertAllClose([2.0], add.eval())
      self.assertAllClose([3.0], radd.eval())
      self.assertAllClose([1.0], sub.eval())
      self.assertAllClose([-1.0], rsub.eval())
      self.assertAllClose([20.0], mul.eval())
      self.assertAllClose([20.0], rmul.eval())
      self.assertAllClose([0.2], div.eval())
      self.assertAllClose([5.0], rdiv.eval())
      self.assertAllClose([-2.0], neg.eval())
      self.assertAllClose([2.0], abs_v.eval())
      self.assertAllClose([True], lt.eval())
      self.assertAllClose([False], rlt.eval())
      self.assertAllClose([True], le.eval())
      self.assertAllClose([True], rle.eval())
      self.assertAllClose([False], gt.eval())
      self.assertAllClose([True], rgt.eval())
      self.assertAllClose([True], ge.eval())
      self.assertAllClose([True], rge.eval())

      self.assertAllClose([6], mod.eval())
      self.assertAllClose([3], rmod.eval())

      self.assertAllClose([True, False], and_v.eval())
      self.assertAllClose([True, True], or_v.eval())
      self.assertAllClose([True, False], xor_v.eval())
      self.assertAllClose([False, True], invert_v.eval())

      self.assertAllClose(rnd[2, 0:0], slice_v.eval())

  def testSession(self):
    with self.test_session() as sess:
      var = tf.Variable([1, 12])
      tf.initialize_all_variables().run()
      self.assertAllClose([1, 12], sess.run(var))

  def testDevicePlacement(self):
    with self.test_session() as sess:
      with tf.device("/cpu:0"):
        var = tf.Variable([1, 12])
      init_value = var.initialized_value()
      init_op = tf.initialize_all_variables()
      self.assertEqual(var.op.device, init_value.device)
      self.assertEqual(var.op.device, init_op.device)
      sess.run(init_op)

  def testColocation(self):
    with tf.device("/job:ps"):
      var = tf.Variable(0, name="v")
    with tf.device("/job:worker/task:7"):
      assign_op = var.assign(1)
    self.assertDeviceEqual("/job:ps", assign_op.device)
    self.assertEqual([b"loc:@v"], assign_op.op.colocation_groups())

  def testInitializerFunction(self):
    value = [[-42], [133.7]]
    shape = [2, 1]
    with self.test_session():
      initializer = lambda: tf.constant(value)
      with self.assertRaises(ValueError):
        # Checks that dtype must be specified.
        tf.Variable(initializer)

      v1 = tf.Variable(initializer, dtype=tf.float32)
      self.assertEqual(shape, v1.get_shape())
      self.assertAllClose(value, v1.initial_value.eval())
      with self.assertRaises(tf.errors.FailedPreconditionError):
        v1.eval()

      v2 = tf.Variable(tf.neg(v1.initialized_value()), dtype=tf.float32)
      self.assertEqual(v1.get_shape(), v2.get_shape())
      self.assertAllClose(np.negative(value), v2.initial_value.eval())

      # Once v2.initial_value.eval() has been called, v1 has effectively been
      # initialized.
      self.assertAllClose(value, v1.eval())

      with self.assertRaises(tf.errors.FailedPreconditionError):
        v2.eval()
      tf.initialize_all_variables().run()
      self.assertAllClose(np.negative(value), v2.eval())

  def testInitializerFunctionDevicePlacement(self):
    with self.test_session():
      initializer = lambda: tf.constant(42.0)
      with tf.device("/cpu:100"):
        v1 = tf.Variable(initializer, dtype=tf.float32, name="v1")
      expected_device = "/device:CPU:100"
      expected_group_v1 = [b"loc:@v1"]
      self.assertEqual(expected_device, v1.op.device)
      self.assertEqual(expected_group_v1, v1.op.colocation_groups())
      for i in v1.initializer.inputs:
        self.assertEqual(expected_device, i.op.device)
        self.assertEqual(expected_group_v1, i.op.colocation_groups())

      v2 = tf.Variable(initializer, dtype=tf.float32, name="v2")
      expected_group_v2 = [b"loc:@v2"]
      self.assertEqual(expected_group_v2, v2.op.colocation_groups())
      for i in v2.initializer.inputs:
        self.assertEqual(expected_group_v2, i.op.colocation_groups())


class IsInitializedTest(tf.test.TestCase):

  def testNoVars(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      uninited = tf.report_uninitialized_variables()
      self.assertEqual(0, sess.run(uninited).size)

  def testAssertVariablesInitialized(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2], name="v")
      w = tf.Variable([3, 4], name="w")
      _ = v, w
      uninited = tf.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), sess.run(uninited))
      tf.initialize_all_variables().run()
      self.assertEqual(0, sess.run(uninited).size)

  def testVariableList(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2], name="v")
      w = tf.Variable([3, 4], name="w")
      uninited = tf.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), sess.run(uninited))
      sess.run(w.initializer)
      self.assertAllEqual(np.array([b"v"]), sess.run(uninited))
      v.initializer.run()
      self.assertEqual(0, sess.run(uninited).size)

  def testZeroSizeVarInitialized(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable(tf.zeros([0, 2]), name="v")
      uninited = tf.report_uninitialized_variables()
      v.initializer.run()  # not strictly necessary
      self.assertEqual(0, sess.run(uninited).size)

  def testTrainingWithZeroSizeVar(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      a = tf.Variable(tf.zeros([0, 2]))
      b = tf.Variable(tf.ones([2, 2]))
      objective = tf.reduce_sum(b + tf.matmul(a, a, transpose_a=True))
      tf.initialize_all_variables().run()
      do_opt = tf.train.GradientDescentOptimizer(0.1).minimize(objective)
      sess.run([do_opt])
      self.assertAllClose([[0.9, 0.9], [0.9, 0.9]], b.eval())


class ObsoleteIsInitializedTest(tf.test.TestCase):

  def testNoVars(self):
    with tf.Graph().as_default():
      self.assertEqual(None, tf.assert_variables_initialized())

  def testVariables(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2])
      w = tf.Variable([3, 4])
      _ = v, w
      inited = tf.assert_variables_initialized()
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        sess.run(inited)
      tf.initialize_all_variables().run()
      sess.run(inited)

  def testVariableList(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2])
      w = tf.Variable([3, 4])
      inited = tf.assert_variables_initialized([v])
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      sess.run(w.initializer)
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      v.initializer.run()
      inited.op.run()


class PartitionedVariableTest(tf.test.TestCase):

  def testPartitionedVariable(self):
    with tf.Graph().as_default():
      v0 = tf.Variable([0])
      v1 = tf.Variable([1])
      v0._set_save_slice_info(variables.Variable.SaveSliceInfo(
          v0.name, [2], [0], [1]))
      v1._set_save_slice_info(variables.Variable.SaveSliceInfo(
          v0.name, [2], [1], [1]))
      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      partitioned_variable = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v1, v0],
          partitions=partitions)

      concatenated = tf.convert_to_tensor(partitioned_variable)
      num_partitions = len(partitioned_variable)
      iterated_partitions = list(partitioned_variable)
      self.assertEqual(2, num_partitions)
      self.assertEqual([v0, v1], iterated_partitions)
      self.assertEqual([2], concatenated.get_shape())

  def testPartitionedVariableFailures(self):
    with tf.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "empty"):
        variables.PartitionedVariable(
            name="fail",
            shape=2,
            dtype=tf.int32,
            variable_list=[],
            partitions=[])

      with self.assertRaisesRegexp(ValueError, "must have a save_slice_info"):
        v0 = tf.Variable([0])
        partitions = [1]
        variables.PartitionedVariable(
            name="two_vars",
            shape=[1],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)

      with self.assertRaisesRegexp(ValueError, "full shapes must match"):
        v0 = tf.Variable([0])
        v1 = tf.Variable([1])
        v0._set_save_slice_info(variables.Variable.SaveSliceInfo(
            v0.name, [2], [0], [1]))
        v1._set_save_slice_info(variables.Variable.SaveSliceInfo(
            v0.name, [2], [1], [1]))
        partitions = [2]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[3],
            dtype=v0.dtype,
            variable_list=[v1, v0],
            partitions=partitions)

      with self.assertRaisesRegexp(ValueError, "must be positive"):
        v0 = tf.Variable([0])
        v0._set_save_slice_info(variables.Variable.SaveSliceInfo(
            v0.name, [2], [0], [1]))
        partitions = [0]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[2],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)


class VariableContainerTest(tf.test.TestCase):

  def testContainer(self):
    with tf.Graph().as_default():
      v0 = tf.Variable([0])
      with tf.container("l1"):
        v1 = tf.Variable([1])
        with tf.container("l2"):
          v2 = tf.Variable([2])
          special_v = state_ops.variable_op([1], tf.float32, container="l3")
        v3 = tf.Variable([3])
      v4 = tf.Variable([4])
    self.assertEqual(tf.compat.as_bytes(""), v0.op.get_attr("container"))
    self.assertEqual(tf.compat.as_bytes("l1"), v1.op.get_attr("container"))
    self.assertEqual(tf.compat.as_bytes("l2"), v2.op.get_attr("container"))
    self.assertEqual(tf.compat.as_bytes("l3"),
                     special_v.op.get_attr("container"))
    self.assertEqual(tf.compat.as_bytes("l1"), v3.op.get_attr("container"))
    self.assertEqual(tf.compat.as_bytes(""), v4.op.get_attr("container"))


if __name__ == "__main__":
  tf.test.main()
