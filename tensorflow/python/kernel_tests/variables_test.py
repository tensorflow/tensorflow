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

import functools
import operator

import numpy as np

from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import compat


class VariablesTestCase(test.TestCase):

  def testInitialization(self):
    with self.cached_session():
      var0 = variables.VariableV1(0.0)
      self.assertEqual("Variable:0", var0.name)
      self.assertEqual("Variable", var0._shared_name)
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.shape)

      var1 = variables.VariableV1(1.1)
      self.assertEqual("Variable_1:0", var1.name)
      self.assertEqual("Variable_1", var1._shared_name)
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.shape)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var0)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var1)

      variables.global_variables_initializer().run()

      self.assertAllClose(0.0, self.evaluate(var0))
      self.assertAllClose(1.1, self.evaluate(var1))

  def testInitializationOrder(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([3, 6]), name="rnd")
      self.assertEqual("rnd:0", rnd.name)
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.shape)

      dep = variables.Variable(rnd.initialized_value(), name="dep")
      self.assertEqual("dep:0", dep.name)
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.shape)

      # Currently have to set the shape manually for Add.
      added_val = rnd.initialized_value() + dep.initialized_value() + 2.0
      added_val.set_shape(rnd.get_shape())

      depdep = variables.Variable(added_val, name="depdep")
      self.assertEqual("depdep:0", depdep.name)
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.shape)

      variables.global_variables_initializer().run()

      self.assertAllClose(rnd.eval(), self.evaluate(dep))
      self.assertAllClose(rnd.eval() + self.evaluate(dep) + 2.0,
                          self.evaluate(depdep))

  def testIterable(self):
    with self.assertRaisesRegexp(TypeError, "not iterable"):
      for _ in variables.Variable(0.0):
        pass
    with self.assertRaisesRegexp(TypeError, "not iterable"):
      for _ in variables.Variable([0.0, 1.0]):
        pass

  def testAssignments(self):
    with self.cached_session():
      var = variables.Variable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      variables.global_variables_initializer().run()
      self.assertAllClose(0.0, self.evaluate(var))

      self.assertAllClose(1.0, self.evaluate(plus_one))
      self.assertAllClose(1.0, self.evaluate(var))

      self.assertAllClose(-1.0, self.evaluate(minus_one))
      self.assertAllClose(-1.0, self.evaluate(var))

      self.assertAllClose(4.0, self.evaluate(four))
      self.assertAllClose(4.0, self.evaluate(var))

  def testResourceAssignments(self):
    with self.session(use_gpu=True):
      var = resource_variable_ops.ResourceVariable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      variables.global_variables_initializer().run()
      self.assertAllClose(0.0, self.evaluate(var))

      self.evaluate(plus_one)
      self.assertAllClose(1.0, self.evaluate(var))

      self.evaluate(minus_one)
      self.assertAllClose(-1.0, self.evaluate(var))

      self.evaluate(four)
      self.assertAllClose(4.0, self.evaluate(var))

  def testZeroSizeStringAssign(self):
    with self.cached_session() as sess:
      array = variables.VariableV1(
          initial_value=array_ops.zeros((0,), dtype=dtypes.string),
          name="foo",
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES])
      self.evaluate(variables.local_variables_initializer())
      old_value = array.value()
      copy_op = array.assign(old_value)
      self.assertEqual([], list(self.evaluate(copy_op)))

  def _countUpToTest(self, dtype):
    with self.cached_session():
      zero = constant_op.constant(0, dtype=dtype)
      var = variables.Variable(zero)
      count_up_to = var.count_up_to(3)

      variables.global_variables_initializer().run()
      self.assertEqual(0, self.evaluate(var))

      self.assertEqual(0, self.evaluate(count_up_to))
      self.assertEqual(1, self.evaluate(var))

      self.assertEqual(1, self.evaluate(count_up_to))
      self.assertEqual(2, self.evaluate(var))

      self.assertEqual(2, self.evaluate(count_up_to))
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

  def testCountUpToInt32(self):
    self._countUpToTest(dtypes.int32)

  def testCountUpToInt64(self):
    self._countUpToTest(dtypes.int64)

  def testControlDepsNone(self):
    with self.cached_session():
      c = constant_op.constant(1.0)
      with ops.control_dependencies([c]):
        # d get the control dep.
        d = constant_op.constant(2.0)
        # variables do not.
        var_x = variables.VariableV1(2.0)
      self.assertEqual([c.op], d.op.control_inputs)
      self.assertEqual([], var_x.initializer.control_inputs)
      self.assertEqual([], var_x.value().op.control_inputs)
      self.assertEqual([], var_x._ref().op.control_inputs)  # pylint: disable=protected-access

  def testControlFlow(self):
    with self.cached_session() as sess:
      v0 = variables.Variable(0, name="v0")
      var_dict = {}

      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        v1 = variables.Variable(1, name="v1")
        var_dict["v1"] = v1
        return v1 + v0

      def var_in_else_clause():
        v2 = variables.Variable(2, name="v2")
        var_dict["v2"] = v2
        return v2 + v0

      add = control_flow_ops.cond(
          math_ops.less(v0, 10), var_in_then_clause, var_in_else_clause)
      v1 = var_dict["v1"]
      v2 = var_dict["v2"]
      # We should be able to initialize and run v1 and v2 without initializing
      # v0, even if the variable was created with a control dep on v0.
      self.evaluate(v1.initializer)
      self.assertEqual([1], self.evaluate(v1))
      self.evaluate(v2.initializer)
      self.assertEqual([2], self.evaluate(v2))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      self.evaluate(v0.initializer)
      sess.run(add)

  def testControlFlowInitialization(self):
    """Expects an error if an initializer is in a control-flow scope."""
    def cond(i, _):
      return i < 10

    def body(i, _):
      zero = array_ops.zeros([], dtype=dtypes.int32)
      v = variables.Variable(initial_value=zero)
      return (i + 1, v.read_value())

    with self.assertRaisesRegexp(ValueError, "inside a control-flow"):
      control_flow_ops.while_loop(cond, body, [0, 0])

  def testUseVariableAsTensor(self):
    with self.cached_session():
      var_x = variables.Variable(2.0)
      var_y = variables.Variable(3.0)
      variables.global_variables_initializer().run()
      self.assertAllClose(2.0, self.evaluate(var_x))
      self.assertAllClose(3.0, self.evaluate(var_y))
      self.assertAllClose(5.0, math_ops.add(var_x, var_y).eval())

  def testZeroSizeVarSameAsConst(self):
    with self.cached_session():
      zero_size_var = variables.Variable(array_ops.zeros([0, 2]))
      zero_size_const = array_ops.ones([2, 0])
      variable_mul = math_ops.matmul(zero_size_const, zero_size_var)
      const_mul = math_ops.matmul(
          zero_size_const, zero_size_const, transpose_b=True)
      variables.global_variables_initializer().run()
      variable_output = self.evaluate(variable_mul)
      self.assertAllClose(const_mul.eval(), variable_output)
      self.assertAllClose([[0., 0.], [0., 0.]], variable_output)

  def testCachingDevice(self):
    with self.cached_session():
      var = variables.Variable(2.0)
      self.assertEqual(var.device, var.value().device)
      self.assertEqual(var.device, var.initialized_value().device)

      var_cached = variables.Variable(2.0, caching_device="/job:foo")
      self.assertFalse(var_cached.device.startswith("/job:foo"))
      self.assertTrue(var_cached.value().device.startswith("/job:foo"))

  def testCollections(self):
    with self.cached_session():
      var_x = variables.VariableV1(2.0)
      var_y = variables.VariableV1(2.0, trainable=False)
      var_z = variables.VariableV1(2.0, trainable=True)
      var_t = variables.VariableV1(
          2.0,
          trainable=True,
          collections=[
              ops.GraphKeys.TRAINABLE_VARIABLES, ops.GraphKeys.GLOBAL_VARIABLES
          ])
      self.assertEqual([var_x, var_y, var_z, var_t],
                       variables.global_variables())
      self.assertEqual([var_x, var_z, var_t], variables.trainable_variables())

  def testCollectionsWithScope(self):
    with self.cached_session():
      with ops.name_scope("scope_1"):
        var_x = variables.VariableV1(2.0)
      with ops.name_scope("scope_2"):
        var_y = variables.VariableV1(2.0)

      self.assertEqual([var_x, var_y], variables.global_variables())
      self.assertEqual([var_x], variables.global_variables("scope_1"))
      self.assertEqual([var_y], variables.global_variables("scope_2"))

      self.assertEqual([var_x, var_y], variables.trainable_variables())
      self.assertEqual([var_x], variables.trainable_variables("scope_1"))
      self.assertEqual([var_y], variables.trainable_variables("scope_2"))

  def testOperatorWrapping(self):
    for attr in functools.WRAPPER_ASSIGNMENTS:
      self.assertEqual(
          getattr(variables.Variable.__add__, attr),
          getattr(ops.Tensor.__add__, attr))

  def testOperators(self):
    with self.cached_session():
      var_f = variables.Variable([2.0])
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

      var_i = variables.Variable([20])
      mod = var_i % 7
      rmod = 103 % var_i

      var_b = variables.Variable([True, False])
      and_v = operator.and_(var_b, [True, True])
      or_v = operator.or_(var_b, [False, True])
      xor_v = operator.xor(var_b, [False, False])
      invert_v = ~var_b

      rnd = np.random.rand(4, 4).astype("f")
      var_t = variables.Variable(rnd)
      slice_v = var_t[2, 0:0]

      var_m = variables.Variable([[2.0, 3.0]])
      matmul = var_m.__matmul__([[10.0], [20.0]])
      rmatmul = var_m.__rmatmul__([[10.0], [20.0]])

      variables.global_variables_initializer().run()
      self.assertAllClose([2.0], self.evaluate(add))
      self.assertAllClose([3.0], self.evaluate(radd))
      self.assertAllClose([1.0], self.evaluate(sub))
      self.assertAllClose([-1.0], self.evaluate(rsub))
      self.assertAllClose([20.0], self.evaluate(mul))
      self.assertAllClose([20.0], self.evaluate(rmul))
      self.assertAllClose([0.2], self.evaluate(div))
      self.assertAllClose([5.0], self.evaluate(rdiv))
      self.assertAllClose([-2.0], self.evaluate(neg))
      self.assertAllClose([2.0], self.evaluate(abs_v))
      self.assertAllClose([True], self.evaluate(lt))
      self.assertAllClose([False], self.evaluate(rlt))
      self.assertAllClose([True], self.evaluate(le))
      self.assertAllClose([True], self.evaluate(rle))
      self.assertAllClose([False], self.evaluate(gt))
      self.assertAllClose([True], self.evaluate(rgt))
      self.assertAllClose([True], self.evaluate(ge))
      self.assertAllClose([True], self.evaluate(rge))

      self.assertAllClose([6], self.evaluate(mod))
      self.assertAllClose([3], self.evaluate(rmod))

      self.assertAllClose([True, False], self.evaluate(and_v))
      self.assertAllClose([True, True], self.evaluate(or_v))
      self.assertAllClose([True, False], self.evaluate(xor_v))
      self.assertAllClose([False, True], self.evaluate(invert_v))

      self.assertAllClose(rnd[2, 0:0], self.evaluate(slice_v))

      self.assertAllClose([[80.0]], self.evaluate(matmul))
      self.assertAllClose([[20.0, 30.0], [40.0, 60.0]], self.evaluate(rmatmul))

  def testSession(self):
    with self.cached_session() as sess:
      var = variables.Variable([1, 12])
      variables.global_variables_initializer().run()
      self.assertAllClose([1, 12], self.evaluate(var))

  def testColocation(self):
    with ops.device("/job:ps"):
      var = variables.VariableV1(0, name="v")
    with ops.device("/job:worker/task:7"):
      assign_op = var.assign(1)
    self.assertDeviceEqual("/job:ps", assign_op.device)
    self.assertEqual([b"loc:@v"], assign_op.op.colocation_groups())

  def testInitializerFunction(self):
    value = [[-42], [133.7]]
    shape = [2, 1]
    with self.cached_session():
      initializer = lambda: constant_op.constant(value)

      v1 = variables.Variable(initializer, dtype=dtypes.float32)
      self.assertEqual(shape, v1.get_shape())
      self.assertEqual(shape, v1.shape)
      self.assertAllClose(value, v1.initial_value.eval())
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v1)

      v2 = variables.Variable(
          math_ops.negative(v1.initialized_value()), dtype=dtypes.float32)
      self.assertEqual(v1.get_shape(), v2.get_shape())
      self.assertEqual(v1.shape, v2.shape)
      self.assertAllClose(np.negative(value), v2.initial_value.eval())

      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v2)
      variables.global_variables_initializer().run()
      self.assertAllClose(np.negative(value), self.evaluate(v2))

  def testConstraintArg(self):
    constraint = lambda x: x
    v = variables.Variable(
        lambda: constant_op.constant(1.),
        constraint=constraint)
    self.assertEqual(v.constraint, constraint)

    constraint = 0
    with self.assertRaises(ValueError):
      v = variables.Variable(
          lambda: constant_op.constant(1.),
          constraint=constraint)

  def testNoRefDataRace(self):
    with self.cached_session():
      a = variables.Variable([1, 2, 3], dtype=dtypes.float32)
      b = variables.Variable(a.initialized_value() + 2)
      c = variables.Variable(b.initialized_value() + 2)
      variables.global_variables_initializer().run()
      self.assertAllEqual(a.eval(), [1, 2, 3])
      self.assertAllEqual(b.eval(), [3, 4, 5])
      self.assertAllEqual(c.eval(), [5, 6, 7])

  def testInitializerFunctionDevicePlacement(self):
    with self.cached_session():
      initializer = lambda: constant_op.constant(42.0)
      with ops.device("/cpu:100"):
        v1 = variables.Variable(initializer, dtype=dtypes.float32, name="v1")
      expected_device = "/device:CPU:100"
      expected_group_v1 = [b"loc:@v1"]
      self.assertEqual(expected_device, v1.op.device)
      self.assertEqual(expected_group_v1, v1.op.colocation_groups())
      for i in v1.initializer.inputs:
        self.assertEqual(expected_group_v1, i.op.colocation_groups())

      v2 = variables.Variable(initializer, dtype=dtypes.float32, name="v2")
      expected_group_v2 = [b"loc:@v2"]
      self.assertEqual(expected_group_v2, v2.op.colocation_groups())
      for i in v2.initializer.inputs:
        self.assertEqual(expected_group_v2, i.op.colocation_groups())

  def testVariableDefInitializedInstances(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v_def = variables.Variable(
          initial_value=constant_op.constant(3.0)).to_proto()

    with ops.Graph().as_default(), self.cached_session() as sess:
      # v describes a VariableDef-based variable without an initial value.
      v = variables.Variable(variable_def=v_def)
      self.assertEqual(3.0, sess.run(v.initialized_value()))

      # initialized_value should not rerun the initializer_op if the variable
      # has already been initialized elsewhere.
      sess.run(v.assign(1.0))
      self.assertEqual(1.0, v.initialized_value().eval())

    v_def.ClearField("initial_value_name")
    with ops.Graph().as_default(), self.cached_session() as sess:
      # Restoring a legacy VariableDef proto that does not have
      # initial_value_name set should still work.
      v = variables.Variable(variable_def=v_def)
      # We should also be able to re-export the variable to a new meta graph.
      self.assertProtoEquals(v_def, v.to_proto())
      # But attempts to use initialized_value will result in errors.
      with self.assertRaises(ValueError):
        sess.run(v.initialized_value())

  def testTrainableInProto(self):
    with ops.Graph().as_default():
      non_trainable_variable = variables.Variable(
          trainable=False,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          False,
          variables.Variable(variable_def=non_trainable_variable.to_proto())
          .trainable)
      trainable_variable = variables.Variable(
          trainable=True,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          True,
          variables.Variable(variable_def=trainable_variable.to_proto())
          .trainable)

  def testLoad(self):
    with self.cached_session():
      var = variables.Variable(np.zeros((5, 5), np.float32))
      variables.global_variables_initializer().run()
      var.load(np.ones((5, 5), np.float32))

      self.assertAllClose(np.ones((5, 5), np.float32), self.evaluate(var))

  def testRepr(self):
    var = variables.VariableV1(np.zeros((5, 5), np.float32), name="noop")
    self.assertEqual(
        "<tf.Variable 'noop:0' shape=(5, 5) dtype=float32_ref>",
        repr(var))

  def testVariableNamesPreserveNameScopesWithDefun(self):
    @function.defun
    def create_variable():
      with ops.name_scope("foo"):
        v = variables.Variable(0.0, name="bar")
      self.assertEqual(v.name, "foo/bar:0")
    with ops.get_default_graph().as_default():
      create_variable()


class IsInitializedTest(test.TestCase):

  def testNoVars(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      uninited = variables.report_uninitialized_variables()
      self.assertEqual(0, self.evaluate(uninited).size)

  def testAssertVariablesInitialized(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.Variable([1, 2], name="v")
      w = variables.Variable([3, 4], name="w")
      _ = v, w
      uninited = variables.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), self.evaluate(uninited))
      variables.global_variables_initializer().run()
      self.assertEqual(0, self.evaluate(uninited).size)

  def testVariableList(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.VariableV1([1, 2], name="v")
      w = variables.VariableV1([3, 4], name="w")
      uninited = variables.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), self.evaluate(uninited))
      self.evaluate(w.initializer)
      self.assertAllEqual(np.array([b"v"]), self.evaluate(uninited))
      v.initializer.run()
      self.assertEqual(0, self.evaluate(uninited).size)

  def testZeroSizeVarInitialized(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.Variable(array_ops.zeros([0, 2]), name="v")
      uninited = variables.report_uninitialized_variables()
      v.initializer.run()  # not strictly necessary
      self.assertEqual(0, self.evaluate(uninited).size)

  def testTrainingWithZeroSizeVar(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      a = variables.Variable(array_ops.zeros([0, 2]))
      b = variables.Variable(array_ops.ones([2, 2]))
      objective = math_ops.reduce_sum(b + math_ops.matmul(
          a, a, transpose_a=True))
      variables.global_variables_initializer().run()
      do_opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(
          objective)
      sess.run([do_opt])
      self.assertAllClose([[0.9, 0.9], [0.9, 0.9]], self.evaluate(b))


class ObsoleteIsInitializedTest(test.TestCase):

  def testNoVars(self):
    with ops.Graph().as_default():
      self.assertEqual(None, variables.assert_variables_initialized())

  def testVariables(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.VariableV1([1, 2])
      w = variables.VariableV1([3, 4])
      _ = v, w
      inited = variables.assert_variables_initialized()
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        sess.run(inited)
      variables.global_variables_initializer().run()
      sess.run(inited)

  def testVariableList(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.VariableV1([1, 2])
      w = variables.VariableV1([3, 4])
      inited = variables.assert_variables_initialized([v])
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      self.evaluate(w.initializer)
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      v.initializer.run()
      inited.op.run()


class PartitionedVariableTest(test.TestCase):

  def testPartitionedVariable(self):
    with ops.Graph().as_default():
      v0 = variables.Variable([0])
      v1 = variables.Variable([1])
      v0._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
      v1._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [1], [1]))
      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      partitioned_variable = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v1, v0],
          partitions=partitions)

      concatenated = ops.convert_to_tensor(partitioned_variable)
      num_partitions = len(partitioned_variable)
      iterated_partitions = list(partitioned_variable)
      self.assertEqual(2, num_partitions)
      self.assertEqual([v0, v1], iterated_partitions)
      self.assertEqual([2], partitioned_variable.get_shape())
      self.assertEqual([2], partitioned_variable.shape)
      self.assertEqual([2], concatenated.get_shape())
      self.assertEqual([2], concatenated.shape)

  def testPartitionedVariableFailures(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "empty"):
        variables.PartitionedVariable(
            name="fail",
            shape=2,
            dtype=dtypes.int32,
            variable_list=[],
            partitions=[])

      with self.assertRaisesRegexp(ValueError, "must have a save_slice_info"):
        v0 = variables.Variable([0])
        partitions = [1]
        variables.PartitionedVariable(
            name="two_vars",
            shape=[1],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)

      with self.assertRaisesRegexp(ValueError, "full shapes must match"):
        v0 = variables.Variable([0])
        v1 = variables.Variable([1])
        v0._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
        v1._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [1], [1]))
        partitions = [2]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[3],
            dtype=v0.dtype,
            variable_list=[v1, v0],
            partitions=partitions)

      with self.assertRaisesRegexp(ValueError, "must be positive"):
        v0 = variables.Variable([0])
        v0._set_save_slice_info(
            variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
        partitions = [0]

        variables.PartitionedVariable(
            name="two_vars",
            shape=[2],
            dtype=v0.dtype,
            variable_list=[v0],
            partitions=partitions)

  def testPartitionedVariableAssignments(self):
    with ops.Graph().as_default(), self.cached_session():
      v0 = variables.Variable(initial_value=[0.0])
      v1 = variables.Variable(initial_value=[1.0])
      v2 = variables.Variable(initial_value=[20.0])
      v3 = variables.Variable(initial_value=[30.0])
      v0._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
      v1._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v1.name, [2], [1], [1]))
      v2._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v2.name, [2], [0], [1]))
      v3._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v3.name, [2], [1], [1]))

      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      pv_0 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v0, v1],
          partitions=partitions)

      pv_1 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v2, v3],
          partitions=partitions)

      deltas_a = constant_op.constant([1.0, 2.0])
      deltas_b = constant_op.constant([3.0, 4.0])
      ones = array_ops.ones([2])
      plus_delta = pv_0.assign_add(deltas_a)
      minus_delta = pv_0.assign_sub(deltas_b)
      assign_ones = pv_0.assign(ones)

      c_0 = constant_op.constant([2.0])
      c_1 = constant_op.constant([3.0])
      assign_list = pv_1.assign([c_0, c_1])
      assign_part_value = pv_1.assign_add(assign_ones)
      assign_part_var = pv_1.assign_sub(pv_0)
      variables.global_variables_initializer().run()

      self.assertEqual([1.0], plus_delta[0].eval())
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([3.0], plus_delta[1].eval())
      self.assertEqual([3.0], self.evaluate(v1))

      self.assertEqual([-2.0], minus_delta[0].eval())
      self.assertEqual([-2.0], self.evaluate(v0))
      self.assertEqual([-1.0], minus_delta[1].eval())
      self.assertEqual([-1.0], self.evaluate(v1))

      self.assertEqual([1.0], assign_ones[0].eval())
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([1.0], assign_ones[1].eval())
      self.assertEqual([1.0], self.evaluate(v1))

      self.assertEqual([2.0], assign_list[0].eval())
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], assign_list[1].eval())
      self.assertEqual([3.0], self.evaluate(v3))

      self.assertEqual([3.0], assign_part_value[0].eval())
      self.assertEqual([3.0], self.evaluate(v2))
      self.assertEqual([4.0], assign_part_value[1].eval())
      self.assertEqual([4.0], self.evaluate(v3))

      self.assertEqual([2.0], assign_part_var[0].eval())
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], assign_part_var[1].eval())
      self.assertEqual([3.0], self.evaluate(v3))


class VariableContainerTest(test.TestCase):

  def testContainer(self):
    with ops.Graph().as_default():
      v0 = variables.Variable([0])
      with ops.container("l1"):
        v1 = variables.Variable([1])
        with ops.container("l2"):
          v2 = variables.Variable([2])
          special_v = gen_state_ops.variable(
              shape=[1],
              dtype=dtypes.float32,
              name="VariableInL3",
              container="l3",
              shared_name="")
        v3 = variables.Variable([3])
      v4 = variables.Variable([4])
    self.assertEqual(compat.as_bytes(""), v0.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l1"), v1.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l2"), v2.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l3"), special_v.op.get_attr("container"))
    self.assertEqual(compat.as_bytes("l1"), v3.op.get_attr("container"))
    self.assertEqual(compat.as_bytes(""), v4.op.get_attr("container"))


if __name__ == "__main__":
  test.main()
