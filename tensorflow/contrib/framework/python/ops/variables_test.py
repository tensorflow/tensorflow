# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""variables tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables as variables_lib2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import saver as saver_lib


class LocalVariableTest(test.TestCase):

  def test_local_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], variables_lib.local_variables())
      value0 = 42
      variables_lib2.local_variable(value0)
      value1 = 43
      variables_lib2.local_variable(value1)
      variables = variables_lib.local_variables()
      self.assertEquals(2, len(variables))
      self.assertRaises(errors_impl.OpError, sess.run, variables)
      variables_lib.variables_initializer(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))

  def testLocalVariableNameAndShape(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.local_variable([1, 1, 1, 1, 1], name='a')
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertListEqual([a], variables_lib2.get_local_variables())

  def testLocalVariableNotInAllVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.local_variable(0)
        self.assertFalse(a in variables_lib.global_variables())
        self.assertTrue(a in variables_lib.local_variables())

  def testLocalVariableNotInVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.local_variable(0)
        self.assertFalse(a in variables_lib2.get_variables_to_restore())
        self.assertTrue(a in variables_lib.local_variables())

  def testGetVariablesDontReturnsTransients(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        variables_lib2.local_variable(0)
      with variable_scope.variable_scope('B'):
        variables_lib2.local_variable(0)
      self.assertEquals([], variables_lib2.get_variables('A'))
      self.assertEquals([], variables_lib2.get_variables('B'))

  def testGetLocalVariablesReturnsTransients(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.local_variable(0)
      with variable_scope.variable_scope('B'):
        b = variables_lib2.local_variable(0)
      self.assertEquals([a], variables_lib2.get_local_variables('A'))
      self.assertEquals([b], variables_lib2.get_local_variables('B'))

  def testInitializedVariableValue(self):
    with self.test_session() as sess:
      a = variables_lib2.local_variable([0, 0, 0, 0, 0], name='a')
      sess.run(variables_lib.local_variables_initializer())
      self.assertAllEqual(a.eval(), [0] * 5)

  def testResourceVariable(self):
    a = variables_lib2.local_variable(0)
    b = variables_lib2.local_variable(0, use_resource=True)
    self.assertEqual(type(a), variables_lib.Variable)
    self.assertEqual(type(b), resource_variable_ops.ResourceVariable)


class GlobalVariableTest(test.TestCase):

  def test_global_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], variables_lib.global_variables())
      value0 = 42
      variables_lib2.global_variable(value0)
      value1 = 43
      variables_lib2.global_variable(value1)
      variables = variables_lib.global_variables()
      self.assertEquals(2, len(variables))
      with self.assertRaisesOpError(
          'Attempting to use uninitialized value Variable'):
        sess.run(variables)
      variables_lib.variables_initializer(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))

  def testVariableNameAndShape(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.global_variable([1, 1, 1, 1, 1], name='a')
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertListEqual([a], variables_lib.global_variables())

  def testGlobalVariableNotInLocalVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.global_variable(0)
        self.assertFalse(a in variables_lib.local_variables())
        self.assertTrue(a in variables_lib.global_variables())

  def testGlobalVariableInVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.global_variable(0)
        self.assertFalse(a in variables_lib.local_variables())
        self.assertTrue(a in variables_lib2.get_variables_to_restore())

  def testGetVariablesReturnsThem(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.global_variable(0)
      with variable_scope.variable_scope('B'):
        b = variables_lib2.global_variable(0)
      self.assertEquals([a], variables_lib2.get_variables('A'))
      self.assertEquals([b], variables_lib2.get_variables('B'))

  def testGetLocalVariablesDontReturnsThem(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        variables_lib2.global_variable(0)
      with variable_scope.variable_scope('B'):
        variables_lib2.global_variable(0)
      self.assertEquals([], variables_lib2.get_local_variables('A'))
      self.assertEquals([], variables_lib2.get_local_variables('B'))

  def testInitializedVariableValue(self):
    with self.test_session() as sess:
      a = variables_lib2.global_variable([0, 0, 0, 0, 0], name='a')
      sess.run(variables_lib.global_variables_initializer())
      self.assertAllEqual(a.eval(), [0] * 5)

  def testResourceVariable(self):
    a = variables_lib2.global_variable(0)
    b = variables_lib2.global_variable(0, use_resource=True)
    self.assertEqual(type(a), variables_lib.Variable)
    self.assertEqual(type(b), resource_variable_ops.ResourceVariable)


class GlobalStepTest(test.TestCase):

  def _assert_global_step(self, global_step, expected_dtype=dtypes.int64):
    self.assertEquals('%s:0' % ops.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEquals(expected_dtype, global_step.dtype.base_dtype)
    self.assertEquals([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    with ops.Graph().as_default() as g:
      self.assertEquals(None, variables_lib2.get_global_step())
      variables_lib.Variable(
          0.0,
          trainable=False,
          dtype=dtypes.float32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(TypeError, 'does not have integer type',
                              variables_lib2.get_global_step)
    self.assertRaisesRegexp(TypeError, 'does not have integer type',
                            variables_lib2.get_global_step, g)

  def test_invalid_shape(self):
    with ops.Graph().as_default() as g:
      self.assertEquals(None, variables_lib2.get_global_step())
      variables_lib.Variable(
          [0],
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(TypeError, 'not scalar',
                              variables_lib2.get_global_step)
    self.assertRaisesRegexp(TypeError, 'not scalar',
                            variables_lib2.get_global_step, g)

  def test_create_global_step(self):
    self.assertEquals(None, variables_lib2.get_global_step())
    with ops.Graph().as_default() as g:
      global_step = variables_lib2.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegexp(ValueError, 'already exists',
                              variables_lib2.create_global_step)
      self.assertRaisesRegexp(ValueError, 'already exists',
                              variables_lib2.create_global_step, g)
      self._assert_global_step(variables_lib2.create_global_step(ops.Graph()))

  def test_get_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertEquals(None, variables_lib2.get_global_step())
      variables_lib.Variable(
          0,
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self._assert_global_step(
          variables_lib2.get_global_step(), expected_dtype=dtypes.int32)
    self._assert_global_step(
        variables_lib2.get_global_step(g), expected_dtype=dtypes.int32)

  def test_get_or_create_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertEquals(None, variables_lib2.get_global_step())
      self._assert_global_step(variables_lib2.get_or_create_global_step())
      self._assert_global_step(variables_lib2.get_or_create_global_step(g))


class VariablesTest(test.TestCase):

  def testCreateVariable(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertTrue(a in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
        self.assertFalse(a in ops.get_collection(ops.GraphKeys.MODEL_VARIABLES))
        self.assertFalse(a in variables_lib.local_variables())

  def testGetVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.variable('a', [5])
      self.assertEquals([a, b], variables_lib2.get_variables())
      self.assertEquals([a], variables_lib2.get_variables('A'))
      self.assertEquals([b], variables_lib2.get_variables('B'))

  def testGetVariablesWithScope(self):
    with self.test_session():
      with variable_scope.variable_scope('A') as var_scope:
        a = variables_lib2.variable('a', [5])
        b = variables_lib2.variable('b', [5])
      self.assertSetEqual(
          set([a, b]), set(variables_lib2.get_variables(var_scope)))

  def testGetVariablesSuffix(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('A'):
        b = variables_lib2.variable('b', [5])
      self.assertEquals([a], variables_lib2.get_variables(suffix='a'))
      self.assertEquals([b], variables_lib2.get_variables(suffix='b'))

  def testGetVariableWithSingleVar(self):
    with self.test_session():
      with variable_scope.variable_scope('parent'):
        a = variables_lib2.variable('child', [5])
      self.assertEquals(a, variables_lib2.get_unique_variable('parent/child'))

  def testGetVariableWithDistractors(self):
    with self.test_session():
      with variable_scope.variable_scope('parent'):
        a = variables_lib2.variable('child', [5])
        with variable_scope.variable_scope('child'):
          variables_lib2.variable('grandchild1', [7])
          variables_lib2.variable('grandchild2', [9])
      self.assertEquals(a, variables_lib2.get_unique_variable('parent/child'))

  def testGetVariableThrowsExceptionWithNoMatch(self):
    var_name = 'cant_find_me'
    with self.test_session():
      with self.assertRaises(ValueError):
        variables_lib2.get_unique_variable(var_name)

  def testGetThrowsExceptionWithChildrenButNoMatch(self):
    var_name = 'parent/child'
    with self.test_session():
      with variable_scope.variable_scope(var_name):
        variables_lib2.variable('grandchild1', [7])
        variables_lib2.variable('grandchild2', [9])
      with self.assertRaises(ValueError):
        variables_lib2.get_unique_variable(var_name)

  def testGetVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.variable('a', [5])
      self.assertEquals([a, b], variables_lib2.get_variables_to_restore())

  def testIncludeGetVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.variable('a', [5])
      self.assertEquals([a, b], variables_lib2.get_variables())
      self.assertEquals([a], variables_lib2.get_variables_to_restore(['A']))

  def testExcludeGetVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.variable('a', [5])
      self.assertEquals([a, b], variables_lib2.get_variables())
      self.assertEquals(
          [a], variables_lib2.get_variables_to_restore(exclude=['B']))

  def testWrongIncludeGetVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.variable('a', [5])
      self.assertEquals([a, b], variables_lib2.get_variables())
      self.assertEquals([], variables_lib2.get_variables_to_restore(['a']))

  def testGetMixedVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        b = variables_lib2.variable('b', [5])
      with variable_scope.variable_scope('B'):
        c = variables_lib2.variable('c', [5])
        d = variables_lib2.variable('d', [5])
      self.assertEquals([a, b, c, d], variables_lib2.get_variables())
      self.assertEquals(
          [a, c],
          variables_lib2.get_variables_to_restore(include=['A/a', 'B/c']))

  def testExcludeGetMixedVariablesToRestore(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        b = variables_lib2.variable('b', [5])
      with variable_scope.variable_scope('B'):
        c = variables_lib2.variable('c', [5])
        d = variables_lib2.variable('d', [5])
      self.assertEquals([a, b, c, d], variables_lib2.get_variables())
      self.assertEquals(
          [b, d],
          variables_lib2.get_variables_to_restore(exclude=['A/a', 'B/c']))

  def testReuseVariable(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [])
      with variable_scope.variable_scope('A', reuse=True):
        b = variables_lib2.variable('a', [])
      self.assertEquals(a, b)
      self.assertListEqual([a], variables_lib2.get_variables())

  def testVariableWithRegularizer(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [], regularizer=nn_ops.l2_loss)
      loss = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertDeviceEqual(loss.device, a.device)

  def testVariableWithRegularizerColocate(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable(
            'a', [], device='gpu:0', regularizer=nn_ops.l2_loss)
      loss = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertDeviceEqual(loss.device, a.device)

  def testVariableWithDevice(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [], device='cpu:0')
        b = variables_lib2.variable('b', [], device='cpu:1')
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')

  def testVariableWithDeviceFromScope(self):
    with self.test_session():
      with ops.device('/cpu:0'):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [], device='cpu:1')
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')

  def testVariableWithDeviceFunction(self):

    class DevFn(object):

      def __init__(self):
        self.counter = -1

      def __call__(self, op):
        self.counter += 1
        return 'cpu:%d' % self.counter

    with self.test_session():
      with arg_scope([variables_lib2.variable], device=DevFn()):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [])
        c = variables_lib2.variable('c', [], device='cpu:12')
        d = variables_lib2.variable('d', [])
        with ops.device('cpu:99'):
          e_init = constant_op.constant(12)
        e = variables_lib2.variable('e', initializer=e_init)
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertEqual(a.initial_value.op.colocation_groups(),
                       a.op.colocation_groups())
      self.assertDeviceEqual(b.device, 'cpu:1')
      self.assertEqual(b.initial_value.op.colocation_groups(),
                       b.op.colocation_groups())
      self.assertDeviceEqual(c.device, 'cpu:12')
      self.assertEqual(c.initial_value.op.colocation_groups(),
                       c.op.colocation_groups())
      self.assertDeviceEqual(d.device, 'cpu:2')
      self.assertEqual(d.initial_value.op.colocation_groups(),
                       d.op.colocation_groups())
      self.assertDeviceEqual(e.device, 'cpu:3')
      self.assertDeviceEqual(e.initial_value.device, 'cpu:99')

  def testVariableWithReplicaDeviceSetter(self):
    with self.test_session():
      with ops.device(device_setter.replica_device_setter(ps_tasks=2)):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [])
        c = variables_lib2.variable('c', [], device='cpu:12')
        d = variables_lib2.variable('d', [])
        with ops.device('cpu:99'):
          e_init = constant_op.constant(12)
        e = variables_lib2.variable('e', initializer=e_init)
      # The values below highlight how the replica_device_setter puts initial
      # values on the worker job, and how it merges explicit devices.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertEqual(a.initial_value.op.colocation_groups(),
                       a.op.colocation_groups())
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertEqual(b.initial_value.op.colocation_groups(),
                       b.op.colocation_groups())
      self.assertDeviceEqual(c.device, '/job:ps/task:0/cpu:12')
      self.assertEqual(c.initial_value.op.colocation_groups(),
                       c.op.colocation_groups())
      self.assertDeviceEqual(d.device, '/job:ps/task:1/cpu:0')
      self.assertEqual(d.initial_value.op.colocation_groups(),
                       d.op.colocation_groups())
      self.assertDeviceEqual(e.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/job:worker/cpu:99')

  def testVariableWithVariableDeviceChooser(self):

    with ops.Graph().as_default():
      device_fn = variables_lib2.VariableDeviceChooser(num_tasks=2)
      with arg_scope([variables_lib2.variable], device=device_fn):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [])
        c = variables_lib2.variable('c', [], device='cpu:12')
        d = variables_lib2.variable('d', [])
        with ops.device('cpu:99'):
          e_init = constant_op.constant(12)
        e = variables_lib2.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertEqual(a.initial_value.op.colocation_groups(),
                       a.op.colocation_groups())
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertEqual(b.initial_value.op.colocation_groups(),
                       b.op.colocation_groups())
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertEqual(c.initial_value.op.colocation_groups(),
                       c.op.colocation_groups())
      self.assertDeviceEqual(d.device, '/job:ps/task:0/cpu:0')
      self.assertEqual(d.initial_value.op.colocation_groups(),
                       d.op.colocation_groups())
      self.assertDeviceEqual(e.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')

  def testVariableGPUPlacement(self):

    with ops.Graph().as_default():
      device_fn = variables_lib2.VariableDeviceChooser(device_type='GPU')
      with arg_scope([variables_lib2.variable], device=device_fn):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [])
        c = variables_lib2.variable('c', [], device='cpu:12')
        d = variables_lib2.variable('d', [])
        with ops.device('cpu:99'):
          e_init = constant_op.constant(12)
        e = variables_lib2.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/device:GPU:0')
      self.assertEqual(a.initial_value.op.colocation_groups(),
                       a.op.colocation_groups())
      self.assertDeviceEqual(b.device, '/device:GPU:0')
      self.assertEqual(b.initial_value.op.colocation_groups(),
                       b.op.colocation_groups())
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertEqual(c.initial_value.op.colocation_groups(),
                       c.op.colocation_groups())
      self.assertDeviceEqual(d.device, '/device:GPU:0')
      self.assertEqual(d.initial_value.op.colocation_groups(),
                       d.op.colocation_groups())
      self.assertDeviceEqual(e.device, '/device:GPU:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')


class ModelVariablesTest(test.TestCase):

  def testNameAndShape(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.model_variable('a', [5])
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertListEqual([a], variables_lib2.get_model_variables('A'))

  def testNotInLocalVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.model_variable('a', [5])
        self.assertTrue(a in variables_lib.global_variables())
        self.assertTrue(a in ops.get_collection(ops.GraphKeys.MODEL_VARIABLES))
        self.assertFalse(a in variables_lib.local_variables())

  def testGetVariablesReturns(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.model_variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.model_variable('a', [5])
      self.assertEquals([a], variables_lib2.get_variables('A'))
      self.assertEquals([b], variables_lib2.get_variables('B'))

  def testGetModelVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.model_variable('a', [5])
      with variable_scope.variable_scope('B'):
        b = variables_lib2.model_variable('a', [5])
      self.assertEquals([a], variables_lib2.get_model_variables('A'))
      self.assertEquals([b], variables_lib2.get_model_variables('B'))

  def testGetTrainableVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        variables_lib2.local_variable([5])
        a = variables_lib.Variable([5])
      with variable_scope.variable_scope('B'):
        variables_lib2.local_variable([5])
        b = variables_lib.Variable([5])
      self.assertEquals([a], variables_lib2.get_trainable_variables('A'))
      self.assertEquals([b], variables_lib2.get_trainable_variables('B'))

  def testGetLocalVariables(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        _ = variables_lib2.model_variable('a', [5])
      with variable_scope.variable_scope('B'):
        _ = variables_lib2.model_variable('a', [5])
      self.assertEquals([], variables_lib2.get_local_variables('A'))
      self.assertEquals([], variables_lib2.get_local_variables('B'))

  def testInitializedVariableValue(self):
    with self.test_session() as sess:
      a = variables_lib2.model_variable(
          'a', [5], initializer=init_ops.ones_initializer())
      sess.run(variables_lib.global_variables_initializer())
      self.assertAllEqual(a.eval(), [1] * 5)

  def testDeviceFn(self):

    class DevFn(object):

      def __init__(self):
        self.counter = -1

      def __call__(self, op):
        self.counter += 1
        return '/cpu:%d' % self.counter

    with ops.Graph().as_default():
      with arg_scope([variables_lib2.model_variable], device=DevFn()):
        a = variables_lib2.model_variable('a', [5])
        b = variables_lib2.model_variable('b', [20])
        self.assertDeviceEqual(a.device, '/cpu:0')
        self.assertEqual(a.initial_value.op.colocation_groups(),
                         a.op.colocation_groups())
        self.assertDeviceEqual(b.device, '/cpu:1')
        self.assertEqual(b.initial_value.op.colocation_groups(),
                         b.op.colocation_groups())

  def testVariableWithVariableDeviceChooser(self):

    with ops.Graph().as_default():
      device_fn = variables_lib2.VariableDeviceChooser()
      with arg_scope([variables_lib2.model_variable], device=device_fn):
        a = variables_lib2.model_variable('a', [5])
        b = variables_lib2.model_variable('b', [20])
        self.assertDeviceEqual(a.device, 'cpu:0')
        self.assertEqual(a.initial_value.op.colocation_groups(),
                         a.op.colocation_groups())
        self.assertDeviceEqual(b.device, 'cpu:0')
        self.assertEqual(a.initial_value.op.colocation_groups(),
                         a.op.colocation_groups())


class GetVariablesCollections(test.TestCase):

  def testVariableCollection(self):
    with self.test_session():
      a = variables_lib2.variable('a', [], collections='A')
      b = variables_lib2.variable('b', [], collections='B')
      self.assertEquals(a, ops.get_collection('A')[0])
      self.assertEquals(b, ops.get_collection('B')[0])

  def testVariableCollections(self):
    with self.test_session():
      a = variables_lib2.variable('a', [], collections=['A', 'C'])
      b = variables_lib2.variable('b', [], collections=['B', 'C'])
      self.assertEquals(a, ops.get_collection('A')[0])
      self.assertEquals(b, ops.get_collection('B')[0])
      self.assertListEqual([a, b], ops.get_collection('C'))

  def testVariableCollectionsWithArgScope(self):
    with self.test_session():
      with arg_scope([variables_lib2.variable], collections='A'):
        a = variables_lib2.variable('a', [])
        b = variables_lib2.variable('b', [])
      self.assertListEqual([a, b], ops.get_collection('A'))

  def testVariableCollectionsWithArgScopeNested(self):
    with self.test_session():
      with arg_scope([variables_lib2.variable], collections='A'):
        a = variables_lib2.variable('a', [])
        with arg_scope([variables_lib2.variable], collections='B'):
          b = variables_lib2.variable('b', [])
      self.assertEquals(a, ops.get_collection('A')[0])
      self.assertEquals(b, ops.get_collection('B')[0])

  def testVariableCollectionsWithArgScopeNonNested(self):
    with self.test_session():
      with arg_scope([variables_lib2.variable], collections='A'):
        a = variables_lib2.variable('a', [])
      with arg_scope([variables_lib2.variable], collections='B'):
        b = variables_lib2.variable('b', [])
      variables_lib2.variable('c', [])
      self.assertListEqual([a], ops.get_collection('A'))
      self.assertListEqual([b], ops.get_collection('B'))

  def testVariableRestoreWithArgScopeNested(self):
    with self.test_session():
      a = variables_lib2.variable('a', [])
      with arg_scope(
          [variables_lib2.variable], trainable=False, collections=['A', 'B']):
        b = variables_lib2.variable('b', [])
      c = variables_lib2.variable('c', [], trainable=False)
    self.assertEquals([a, c], variables_lib2.get_variables_to_restore())
    self.assertEquals([a], variables_lib.trainable_variables())
    self.assertEquals([b], ops.get_collection('A'))
    self.assertEquals([b], ops.get_collection('B'))


class GetVariablesBySuffixTest(test.TestCase):

  def testGetVariableGivenNameScoped(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        b = variables_lib2.variable('b', [5])
        self.assertEquals([a], variables_lib2.get_variables_by_suffix('a'))
        self.assertEquals([b], variables_lib2.get_variables_by_suffix('b'))

  def testGetVariableWithScope(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        fooa = variables_lib2.variable('fooa', [5])
      with variable_scope.variable_scope('B'):
        a2 = variables_lib2.variable('a', [5])
      matched_variables = variables_lib2.get_variables_by_suffix('a')
      self.assertEquals([a, fooa, a2], matched_variables)
      matched_variables = variables_lib2.get_variables_by_suffix('/a')
      self.assertEquals([a, a2], matched_variables)
      matched_variables = variables_lib2.get_variables_by_suffix('a', scope='A')
      self.assertEquals([a, fooa], matched_variables)

  def testGetVariableWithoutScope(self):
    with self.test_session():
      a = variables_lib2.variable('a', [5])
      fooa = variables_lib2.variable('fooa', [5])
      b_a = variables_lib2.variable('B/a', [5])
      matched_variables = variables_lib2.get_variables_by_suffix('a')
      self.assertEquals([a, fooa, b_a], matched_variables)
      matched_variables = variables_lib2.get_variables_by_suffix('fooa')
      self.assertEquals([fooa], matched_variables)


class GetVariablesByNameTest(test.TestCase):

  def testGetVariableGivenNameScoped(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        b = variables_lib2.variable('b', [5])
        self.assertEquals([a], variables_lib2.get_variables_by_name('a'))
        self.assertEquals([b], variables_lib2.get_variables_by_name('b'))

  def testGetVariableWithScope(self):
    with self.test_session():
      with variable_scope.variable_scope('A'):
        a = variables_lib2.variable('a', [5])
        fooa = variables_lib2.variable('fooa', [5])
      with variable_scope.variable_scope('B'):
        a2 = variables_lib2.variable('a', [5])
      matched_variables = variables_lib2.get_variables_by_name('a')
      self.assertEquals([a, a2], matched_variables)
      matched_variables = variables_lib2.get_variables_by_name('fooa')
      self.assertEquals([fooa], matched_variables)
      matched_variables = variables_lib2.get_variables_by_name('/a')
      self.assertEquals([], matched_variables)
      matched_variables = variables_lib2.get_variables_by_name('a', scope='A')
      self.assertEquals([a], matched_variables)

  def testGetVariableWithoutScope(self):
    with self.test_session():
      a = variables_lib2.variable('a', [5])
      fooa = variables_lib2.variable('fooa', [5])
      b_a = variables_lib2.variable('B/a', [5])
      matched_variables = variables_lib2.get_variables_by_name('a')
      self.assertEquals([a, b_a], matched_variables)
      matched_variables = variables_lib2.get_variables_by_name('fooa')
      self.assertEquals([fooa], matched_variables)


class GetVariableFullNameTest(test.TestCase):

  def testVariable(self):
    my_var0 = variables_lib2.variable('my_var0', shape=[])
    full_name = variables_lib2.get_variable_full_name(my_var0)
    self.assertEquals(full_name, my_var0.op.name)

  def testPartitionedVariable(self):
    input_full_name = 'my_var0'
    partitioner = partitioned_variables.variable_axis_size_partitioner(2)
    my_var0 = variables_lib2.variable(
        'my_var0', shape=[2, 2], partitioner=partitioner)
    for part_var in list(my_var0):
      computed_full_name = variables_lib2.get_variable_full_name(part_var)
      self.assertEquals(input_full_name, computed_full_name)


class AssignFromValuesTest(test.TestCase):

  def testNoScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = init_ops.truncated_normal_initializer(stddev=.1)
      var0 = variables_lib2.variable(
          'my_var0', shape=[1, 3, 1], initializer=initializer)
      var1 = variables_lib2.variable(
          'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_var0': init_value0, 'my_var1': init_value1}
      assign_op, feed_dict = variables_lib2.assign_from_values(
          var_names_to_values)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      sess.run(assign_op, feed_dict)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)

  def testWithScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = init_ops.truncated_normal_initializer(stddev=.1)

      with variable_scope.variable_scope('my_model/my_layer0'):
        var0 = variables_lib2.variable(
            'my_var0', shape=[1, 3, 1], initializer=initializer)
      with variable_scope.variable_scope('my_model/my_layer1'):
        var1 = variables_lib2.variable(
            'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {
          'my_model/my_layer0/my_var0': init_value0,
          'my_model/my_layer1/my_var1': init_value1
      }
      assign_op, feed_dict = variables_lib2.assign_from_values(
          var_names_to_values)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      sess.run(assign_op, feed_dict)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)


class AssignFromValuesFnTest(test.TestCase):

  def testNoScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = init_ops.truncated_normal_initializer(stddev=.1)
      var0 = variables_lib2.variable(
          'my_var0', shape=[1, 3, 1], initializer=initializer)
      var1 = variables_lib2.variable(
          'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_var0': init_value0, 'my_var1': init_value1}
      init_fn = variables_lib2.assign_from_values_fn(var_names_to_values)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)

  def testWithScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = init_ops.truncated_normal_initializer(stddev=.1)

      with variable_scope.variable_scope('my_model/my_layer0'):
        var0 = variables_lib2.variable(
            'my_var0', shape=[1, 3, 1], initializer=initializer)
      with variable_scope.variable_scope('my_model/my_layer1'):
        var1 = variables_lib2.variable(
            'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {
          'my_model/my_layer0/my_var0': init_value0,
          'my_model/my_layer1/my_var1': init_value1
      }
      init_fn = variables_lib2.assign_from_values_fn(var_names_to_values)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)


class AssignFromCheckpointTest(test.TestCase):

  def create_checkpoint_from_values(self,
                                    var_names_to_values,
                                    checkpoint_dir,
                                    global_step=None):
    """Creates a checkpoint from a mapping of name to values in model_dir.

    Args:
      var_names_to_values: a map from variable names to values.
      checkpoint_dir: the directory where the checkpoint will be saved.
      global_step: the global step used to save the checkpoint.

    Returns:
      the model_path to the checkpoint.
    """
    var_list = []
    with session.Session('', graph=ops.Graph()) as sess:
      # Create a set of variables to save in the checkpoint.
      for var_name in var_names_to_values:
        var_value = var_names_to_values[var_name]
        var_list.append(variables_lib.Variable(var_value, name=var_name))
      saver = saver_lib.Saver(var_list)
      init_op = variables_lib.variables_initializer(var_list)
      sess.run(init_op)
      # Save the initialized values in the file at 'checkpoint_dir'
      return saver.save(sess, checkpoint_dir, global_step=global_step)

  def testLoadExistingVariables(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                     'load_existing_variables'))

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[])
      var1 = variables_lib2.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      op, feed_dict = variables_lib2.assign_from_checkpoint(model_path,
                                                            vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      sess.run(op, feed_dict)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  # Tests restoring PartitionedVariables and tests using a dictionary
  # of lists as the assign_from_checkpoint() var_list param.
  def testLoadPartitionedVariables(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(
        self.get_temp_dir(), 'load_partitioned_variables'))

    init_value0 = np.array([[10.0, 11.0], [12.0, 13.0]])
    init_value1 = np.array([20.0])  # Partitioned into 1 part, edge case.
    var_names_to_values = {'var0': init_value0, 'var1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      # var0 and var1 are PartitionedVariables.
      partitioner = partitioned_variables.variable_axis_size_partitioner(2)
      var0 = variables_lib2.variable(
          'var0', shape=init_value0.shape, partitioner=partitioner)
      var0full = variables_lib2.variable(
          'var0full', shape=init_value0.shape)
      var1 = variables_lib2.variable(
          'var1', shape=init_value1.shape, partitioner=partitioner)

      # Convert var0 and var1 into a list of underlying variables.
      vars_to_restore = {'var0': list(var0) + [var0full], 'var1': list(var1)}
      op, feed_dict = variables_lib2.assign_from_checkpoint(model_path,
                                                            vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      sess.run(op, feed_dict)

      # Request and test the variable values. PartitionedVariables can't
      # be evaled so we wrap them in an identity.
      self.assertTrue(np.array_equal(
          init_value0, array_ops.identity(var0).eval()))
      self.assertTrue(np.array_equal(
          init_value0, var0full.eval()))
      self.assertTrue(np.array_equal(
          init_value1, array_ops.identity(var1).eval()))

  def testRaisesValueErrorIfAVariableIsntFound(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(
        self.get_temp_dir(), 'raises_value_error_if_var_isnt_found'))

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session():
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[])
      var1 = variables_lib2.variable('my_var1', shape=[])

      vars_to_restore = {'v0_fake': var0, 'v1': var1}

      with self.assertRaises(ValueError):
        variables_lib2.assign_from_checkpoint(model_path, vars_to_restore)

  def testInitFromCheckpointWithScopes(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(
        self.get_temp_dir(), 'init_from_checkpoint_with_scopes'))

    init_value0 = np.asarray(
        [1.0, 3.0, 9.0], dtype=np.float32).reshape((1, 3, 1))
    init_value1 = np.asarray(
        [2.0, 4.0, 6.0, 8.0], dtype=np.float32).reshape((2, 1, 2))

    var_names_to_values = {'layer0/v0': init_value0, 'layer1/v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      with variable_scope.variable_scope('my_model/my_layer0'):
        var0 = variables_lib2.variable('my_var0', shape=init_value0.shape)
      with variable_scope.variable_scope('my_model/my_layer1'):
        var1 = variables_lib2.variable('my_var1', shape=init_value1.shape)

      vars_to_restore = {'layer0/v0': var0, 'layer1/v1': var1}
      op, feed_dict = variables_lib2.assign_from_checkpoint(model_path,
                                                            vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      sess.run(op, feed_dict)

      # Request and test the variable values:
      self.assertAllEqual(init_value0, var0.eval())
      self.assertAllEqual(init_value1, var1.eval())


class AssignFromCheckpointFnTest(test.TestCase):

  def create_checkpoint_from_values(self,
                                    var_names_to_values,
                                    checkpoint_dir,
                                    global_step=None):
    """Creates a checkpoint from a mapping of name to values in model_dir.

    Args:
      var_names_to_values: a map from variable names to values.
      checkpoint_dir: the directory where the checkpoint will be saved.
      global_step: the global step used to save the checkpoint.

    Returns:
      the model_path to the checkpoint.
    """
    var_list = []
    with session.Session('', graph=ops.Graph()) as sess:
      # Create a set of variables to save in the checkpoint.
      for var_name in var_names_to_values:
        var_value = var_names_to_values[var_name]
        var_list.append(variables_lib.Variable(var_value, name=var_name))
      saver = saver_lib.Saver(var_list)
      init_op = variables_lib.variables_initializer(var_list)
      sess.run(init_op)
      # Save the initialized values in the file at 'checkpoint_dir'
      return saver.save(sess, checkpoint_dir, global_step=global_step)

  def testLoadExistingVariables(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                     'load_existing_variables'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[])
      var1 = variables_lib2.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = variables_lib2.assign_from_checkpoint_fn(model_path,
                                                         vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testLoadExistingVariablesDifferentShapeDefaultDoesNotAllowReshape(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(
        self.get_temp_dir(), 'load_existing_vars_no_reshape'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = [[10.0, 11.0]]
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[2, 1])
      var1 = variables_lib2.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = variables_lib2.assign_from_checkpoint_fn(model_path,
                                                         vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      with self.assertRaises(errors_impl.InvalidArgumentError):
        init_fn(sess)

  def testLoadExistingVariablesDifferentShapeAllowReshape(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(
        self.get_temp_dir(),
        'load_existing_variables_different_shape_allow_reshape'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = [[10.0, 11.0]]
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[2, 1])
      var1 = variables_lib2.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = variables_lib2.assign_from_checkpoint_fn(
          model_path, vars_to_restore, reshape_variables=True)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertAllEqual(np.transpose(np.array(init_value0)), var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testNotFoundError(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                     'not_found_error'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[])
      var1 = variables_lib2.variable('my_var1', shape=[])
      var2 = variables_lib2.variable('my_var2', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1, 'v2': var2}
      init_fn = variables_lib2.assign_from_checkpoint_fn(model_path,
                                                         vars_to_restore)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      with self.assertRaises(errors_impl.NotFoundError):
        init_fn(sess)

  def testMissingVariablesList(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                     'missing_variables_list'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('v0', shape=[])
      var1 = variables_lib2.variable('v1', shape=[])
      var2 = variables_lib2.variable('v2', shape=[])

      vars_to_restore = [var0, var1, var2]
      init_fn = variables_lib2.assign_from_checkpoint_fn(
          model_path, vars_to_restore, ignore_missing_vars=True)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testMissingVariablesDict(self):
    model_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                     'missing_variables_dict'))
    if gfile.Exists(model_dir):
      gfile.DeleteRecursively(model_dir)

    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = variables_lib2.variable('my_var0', shape=[])
      var1 = variables_lib2.variable('my_var1', shape=[])
      var2 = variables_lib2.variable('my_var2', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1, 'v2': var2}
      init_fn = variables_lib2.assign_from_checkpoint_fn(
          model_path, vars_to_restore, ignore_missing_vars=True)

      # Initialize the variables.
      sess.run(variables_lib.global_variables_initializer())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())


class ZeroInitializerOpTest(test.TestCase):

  def _testZeroInitializer(self, shape, initializer, use_init):
    var = variables_lib.Variable(initializer)
    var_zero = variables_lib2.zero_initializer(var)
    with self.test_session() as sess:
      with self.assertRaisesOpError('Attempting to use uninitialized value'):
        var.eval()
      if use_init:
        sess.run(var.initializer)
        with self.assertRaisesOpError('input is already initialized'):
          var_zero.eval()
        self.assertAllClose(np.ones(shape), var.eval())
      else:
        var_zero.eval()
        self.assertAllClose(np.zeros(shape), var.eval())

  def testZeroInitializer(self):
    for dtype in (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64):
      for use_init in (False, True):
        self._testZeroInitializer(
            [10, 20], array_ops.ones(
                [10, 20], dtype=dtype), use_init)


class FilterVariablesTest(test.TestCase):

  def setUp(self):
    g = ops.Graph()
    with g.as_default():
      var_list = []
      var_list.append(variables_lib.Variable(0, name='conv1/weights'))
      var_list.append(variables_lib.Variable(0, name='conv1/biases'))
      var_list.append(variables_lib.Variable(0, name='conv2/weights'))
      var_list.append(variables_lib.Variable(0, name='conv2/biases'))
      var_list.append(variables_lib.Variable(0, name='clfs/weights'))
      var_list.append(variables_lib.Variable(0, name='clfs/biases'))
      self._var_list = var_list

  def _test_filter_variables(self,
                             expected_var_names,
                             include_patterns=None,
                             exclude_patterns=None,
                             reg_search=True):
    filtered_var_list = variables_lib2.filter_variables(
        self._var_list,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        reg_search=reg_search)

    filtered_var_names = [var.op.name for var in filtered_var_list]

    for name in filtered_var_names:
      self.assertIn(name, expected_var_names)
    for name in expected_var_names:
      self.assertIn(name, filtered_var_names)
    self.assertEqual(len(filtered_var_names), len(expected_var_names))

  def testNoFiltering(self):
    self._test_filter_variables(expected_var_names=[
        'conv1/weights', 'conv1/biases', 'conv2/weights', 'conv2/biases',
        'clfs/weights', 'clfs/biases'
    ])

  def testIncludeBiases(self):
    self._test_filter_variables(
        expected_var_names=['conv1/biases', 'conv2/biases', 'clfs/biases'],
        include_patterns=['biases'])

  def testExcludeWeights(self):
    self._test_filter_variables(
        expected_var_names=['conv1/biases', 'conv2/biases', 'clfs/biases'],
        exclude_patterns=['weights'])

  def testExcludeWeightsAndConv1(self):
    self._test_filter_variables(
        expected_var_names=['conv2/biases', 'clfs/biases'],
        exclude_patterns=['weights', 'conv1'])

  def testTwoIncludePatternsEnsureNoVariablesTwiceInFilteredList(self):
    self._test_filter_variables(
        expected_var_names=[
            'conv1/weights', 'conv1/biases', 'conv2/weights', 'clfs/weights'
        ],
        include_patterns=['conv1', 'weights'])

  def testIncludeConv1ExcludeBiases(self):
    self._test_filter_variables(
        expected_var_names=['conv1/weights'],
        include_patterns=['conv1'],
        exclude_patterns=['biases'])

  def testRegMatchIncludeBiases(self):
    self._test_filter_variables(
        expected_var_names=['conv1/biases', 'conv2/biases', 'clfs/biases'],
        include_patterns=['.*biases'],
        reg_search=False)

  def testRegMatchIncludeBiasesWithIncompleteRegExpHasNoMatches(self):
    self._test_filter_variables(
        expected_var_names=[], include_patterns=['biases'], reg_search=False)


if __name__ == '__main__':
  test.main()
