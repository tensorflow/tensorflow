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

import numpy as np
import tensorflow as tf


class LocalVariableTest(tf.test.TestCase):

  def test_local_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], tf.local_variables())
      value0 = 42
      tf.contrib.framework.local_variable(value0)
      value1 = 43
      tf.contrib.framework.local_variable(value1)
      variables = tf.local_variables()
      self.assertEquals(2, len(variables))
      self.assertRaises(tf.OpError, sess.run, variables)
      tf.initialize_variables(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))

  def testLocalVariableNameAndShape(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.local_variable([1, 1, 1, 1, 1], name='a')
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertListEqual([a], tf.contrib.framework.get_local_variables())

  def testLocalVariableNotInAllVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.local_variable(0)
        self.assertFalse(a in tf.all_variables())
        self.assertTrue(a in tf.local_variables())

  def testLocalVariableNotInVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.local_variable(0)
        self.assertFalse(a in tf.contrib.framework.get_variables_to_restore())
        self.assertTrue(a in tf.local_variables())

  def testGetVariablesDontReturnsTransients(self):
    with self.test_session():
      with tf.variable_scope('A'):
        tf.contrib.framework.local_variable(0)
      with tf.variable_scope('B'):
        tf.contrib.framework.local_variable(0)
      self.assertEquals([], tf.contrib.framework.get_variables('A'))
      self.assertEquals([], tf.contrib.framework.get_variables('B'))

  def testGetLocalVariablesReturnsTransients(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.local_variable(0)
      with tf.variable_scope('B'):
        b = tf.contrib.framework.local_variable(0)
      self.assertEquals([a], tf.contrib.framework.get_local_variables('A'))
      self.assertEquals([b], tf.contrib.framework.get_local_variables('B'))

  def testInitializedVariableValue(self):
    with self.test_session() as sess:
      a = tf.contrib.framework.local_variable([0, 0, 0, 0, 0], name='a')
      sess.run(tf.initialize_local_variables())
      self.assertAllEqual(a.eval(), [0]*5)


class GlobalStepTest(tf.test.TestCase):

  def _assert_global_step(self, global_step, expected_dtype=tf.int64):
    self.assertEquals('%s:0' % tf.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEquals(expected_dtype, global_step.dtype.base_dtype)
    self.assertEquals([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          0.0, trainable=False, dtype=tf.float32, name=tf.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(
          TypeError, 'does not have integer type',
          tf.contrib.framework.get_global_step)
    self.assertRaisesRegexp(
        TypeError, 'does not have integer type',
        tf.contrib.framework.get_global_step, g)

  def test_invalid_shape(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          [0], trainable=False, dtype=tf.int32, name=tf.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(
          TypeError, 'not scalar',
          tf.contrib.framework.get_global_step)
    self.assertRaisesRegexp(
        TypeError, 'not scalar',
        tf.contrib.framework.get_global_step, g)

  def test_create_global_step(self):
    self.assertEquals(None, tf.contrib.framework.get_global_step())
    with tf.Graph().as_default() as g:
      global_step = tf.contrib.framework.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegexp(
          ValueError, 'already exists', tf.contrib.framework.create_global_step)
      self.assertRaisesRegexp(
          ValueError, 'already exists', tf.contrib.framework.create_global_step,
          g)
      self._assert_global_step(
          tf.contrib.framework.create_global_step(tf.Graph()))

  def test_get_global_step(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      tf.Variable(
          0, trainable=False, dtype=tf.int32, name=tf.GraphKeys.GLOBAL_STEP)
      self._assert_global_step(
          tf.contrib.framework.get_global_step(), expected_dtype=tf.int32)
    self._assert_global_step(
        tf.contrib.framework.get_global_step(g), expected_dtype=tf.int32)

  def test_get_or_create_global_step(self):
    with tf.Graph().as_default() as g:
      self.assertEquals(None, tf.contrib.framework.get_global_step())
      self._assert_global_step(
          tf.contrib.framework.get_or_create_global_step())
      self._assert_global_step(
          tf.contrib.framework.get_or_create_global_step(g))


class VariablesTest(tf.test.TestCase):

  def testCreateVariable(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertTrue(a in tf.get_collection(tf.GraphKeys.VARIABLES))
        self.assertFalse(a in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        self.assertFalse(a in tf.local_variables())

  def testGetVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.variable('a', [5])
      self.assertEquals([a, b], tf.contrib.framework.get_variables())
      self.assertEquals([a], tf.contrib.framework.get_variables('A'))
      self.assertEquals([b], tf.contrib.framework.get_variables('B'))

  def testGetVariablesSuffix(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('A'):
        b = tf.contrib.framework.variable('b', [5])
      self.assertEquals([a], tf.contrib.framework.get_variables(suffix='a'))
      self.assertEquals([b], tf.contrib.framework.get_variables(suffix='b'))

  def testGetVariableWithSingleVar(self):
    with self.test_session():
      with tf.variable_scope('parent'):
        a = tf.contrib.framework.variable('child', [5])
      self.assertEquals(
          a, tf.contrib.framework.get_unique_variable('parent/child'))

  def testGetVariableWithDistractors(self):
    with self.test_session():
      with tf.variable_scope('parent'):
        a = tf.contrib.framework.variable('child', [5])
        with tf.variable_scope('child'):
          tf.contrib.framework.variable('grandchild1', [7])
          tf.contrib.framework.variable('grandchild2', [9])
      self.assertEquals(
          a, tf.contrib.framework.get_unique_variable('parent/child'))

  def testGetVariableThrowsExceptionWithNoMatch(self):
    var_name = 'cant_find_me'
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.framework.get_unique_variable(var_name)

  def testGetThrowsExceptionWithChildrenButNoMatch(self):
    var_name = 'parent/child'
    with self.test_session():
      with tf.variable_scope(var_name):
        tf.contrib.framework.variable('grandchild1', [7])
        tf.contrib.framework.variable('grandchild2', [9])
      with self.assertRaises(ValueError):
        tf.contrib.framework.get_unique_variable(var_name)

  def testGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.variable('a', [5])
      self.assertEquals([a, b],
                        tf.contrib.framework.get_variables_to_restore())

  def testIncludeGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.variable('a', [5])
      self.assertEquals([a, b], tf.contrib.framework.get_variables())
      self.assertEquals([a],
                        tf.contrib.framework.get_variables_to_restore(['A']))

  def testExcludeGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.variable('a', [5])
      self.assertEquals([a, b], tf.contrib.framework.get_variables())
      self.assertEquals([a],
                        tf.contrib.framework.get_variables_to_restore(
                            exclude=['B']))

  def testWrongIncludeGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.variable('a', [5])
      self.assertEquals([a, b], tf.contrib.framework.get_variables())
      self.assertEquals([],
                        tf.contrib.framework.get_variables_to_restore(['a']))

  def testGetMixedVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        b = tf.contrib.framework.variable('b', [5])
      with tf.variable_scope('B'):
        c = tf.contrib.framework.variable('c', [5])
        d = tf.contrib.framework.variable('d', [5])
      self.assertEquals([a, b, c, d], tf.contrib.framework.get_variables())
      self.assertEquals([a, c],
                        tf.contrib.framework.get_variables_to_restore(
                            include=['A/a', 'B/c']))

  def testExcludeGetMixedVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        b = tf.contrib.framework.variable('b', [5])
      with tf.variable_scope('B'):
        c = tf.contrib.framework.variable('c', [5])
        d = tf.contrib.framework.variable('d', [5])
      self.assertEquals([a, b, c, d], tf.contrib.framework.get_variables())
      self.assertEquals([b, d],
                        tf.contrib.framework.get_variables_to_restore(
                            exclude=['A/a', 'B/c']))

  def testReuseVariable(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [])
      with tf.variable_scope('A', reuse=True):
        b = tf.contrib.framework.variable('a', [])
      self.assertEquals(a, b)
      self.assertListEqual([a], tf.contrib.framework.get_variables())

  def testVariableWithRegularizer(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [], regularizer=tf.nn.l2_loss)
      loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertDeviceEqual(loss.device, a.device)

  def testVariableWithRegularizerColocate(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [], device='gpu:0',
                                          regularizer=tf.nn.l2_loss)
      loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertDeviceEqual(loss.device, a.device)

  def testVariableWithDevice(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [], device='cpu:0')
        b = tf.contrib.framework.variable('b', [], device='cpu:1')
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')

  def testVariableWithDeviceFromScope(self):
    with self.test_session():
      with tf.device('/cpu:0'):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [], device='cpu:1')
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
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          device=DevFn()):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [])
        c = tf.contrib.framework.variable('c', [], device='cpu:12')
        d = tf.contrib.framework.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = tf.contrib.framework.variable('e', initializer=e_init)
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(a.initial_value.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')
      self.assertDeviceEqual(b.initial_value.device, 'cpu:1')
      self.assertDeviceEqual(c.device, 'cpu:12')
      self.assertDeviceEqual(c.initial_value.device, 'cpu:12')
      self.assertDeviceEqual(d.device, 'cpu:2')
      self.assertDeviceEqual(d.initial_value.device, 'cpu:2')
      self.assertDeviceEqual(e.device, 'cpu:3')
      self.assertDeviceEqual(e.initial_value.device, 'cpu:99')

  def testVariableWithReplicaDeviceSetter(self):
    with self.test_session():
      with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [])
        c = tf.contrib.framework.variable('c', [], device='cpu:12')
        d = tf.contrib.framework.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = tf.contrib.framework.variable('e', initializer=e_init)
      # The values below highlight how the replica_device_setter puts initial
      # values on the worker job, and how it merges explicit devices.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(a.initial_value.device, a.device)
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(b.initial_value.device, b.device)
      self.assertDeviceEqual(c.device, '/job:ps/task:0/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, c.device)
      self.assertDeviceEqual(d.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(d.initial_value.device, d.device)
      self.assertDeviceEqual(e.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/job:worker/cpu:99')

  def testVariableWithVariableDeviceChooser(self):

    with tf.Graph().as_default():
      device_fn = tf.contrib.framework.VariableDeviceChooser(num_tasks=2)
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          device=device_fn):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [])
        c = tf.contrib.framework.variable('c', [], device='cpu:12')
        d = tf.contrib.framework.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = tf.contrib.framework.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(a.initial_value.device, a.device)
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(b.initial_value.device, b.device)
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, c.device)
      self.assertDeviceEqual(d.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(d.initial_value.device, d.device)
      self.assertDeviceEqual(e.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')

  def testVariableGPUPlacement(self):

    with tf.Graph().as_default():
      device_fn = tf.contrib.framework.VariableDeviceChooser(device_type='GPU')
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          device=device_fn):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [])
        c = tf.contrib.framework.variable('c', [], device='cpu:12')
        d = tf.contrib.framework.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = tf.contrib.framework.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/gpu:0')
      self.assertDeviceEqual(a.initial_value.device, a.device)
      self.assertDeviceEqual(b.device, '/gpu:0')
      self.assertDeviceEqual(b.initial_value.device, b.device)
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, c.device)
      self.assertDeviceEqual(d.device, '/gpu:0')
      self.assertDeviceEqual(d.initial_value.device, d.device)
      self.assertDeviceEqual(e.device, '/gpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')


class ModelVariablesTest(tf.test.TestCase):

  def testNameAndShape(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.model_variable('a', [5])
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])
        self.assertListEqual([a], tf.contrib.framework.get_model_variables('A'))

  def testNotInLocalVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.model_variable('a', [5])
        self.assertTrue(a in tf.all_variables())
        self.assertTrue(a in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        self.assertFalse(a in tf.local_variables())

  def testGetVariablesReturns(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.model_variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.model_variable('a', [5])
      self.assertEquals([a], tf.contrib.framework.get_variables('A'))
      self.assertEquals([b], tf.contrib.framework.get_variables('B'))

  def testGetModelVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.model_variable('a', [5])
      with tf.variable_scope('B'):
        b = tf.contrib.framework.model_variable('a', [5])
      self.assertEquals([a], tf.contrib.framework.get_model_variables('A'))
      self.assertEquals([b], tf.contrib.framework.get_model_variables('B'))

  def testGetLocalVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        _ = tf.contrib.framework.model_variable('a', [5])
      with tf.variable_scope('B'):
        _ = tf.contrib.framework.model_variable('a', [5])
      self.assertEquals([], tf.contrib.framework.get_local_variables('A'))
      self.assertEquals([], tf.contrib.framework.get_local_variables('B'))

  def testInitializedVariableValue(self):
    with self.test_session() as sess:
      a = tf.contrib.framework.model_variable(
          'a', [5], initializer=tf.ones_initializer)
      sess.run(tf.initialize_all_variables())
      self.assertAllEqual(a.eval(), [1]*5)

  def testDeviceFn(self):
    class DevFn(object):

      def __init__(self):
        self.counter = -1

      def __call__(self, op):
        self.counter += 1
        return '/cpu:%d' % self.counter

    with tf.Graph().as_default():
      with tf.contrib.framework.arg_scope([tf.contrib.framework.model_variable],
                                          device=DevFn()):
        a = tf.contrib.framework.model_variable('a', [5])
        b = tf.contrib.framework.model_variable('b', [20])
        self.assertDeviceEqual(a.device, '/cpu:0')
        self.assertDeviceEqual(a.initial_value.device, '/cpu:0')
        self.assertDeviceEqual(b.device, '/cpu:1')
        self.assertDeviceEqual(b.initial_value.device, '/cpu:1')

  def testVariableWithVariableDeviceChooser(self):

    with tf.Graph().as_default():
      device_fn = tf.contrib.framework.VariableDeviceChooser()
      with tf.contrib.framework.arg_scope([tf.contrib.framework.model_variable],
                                          device=device_fn):
        a = tf.contrib.framework.model_variable('a', [5])
        b = tf.contrib.framework.model_variable('b', [20])
        self.assertDeviceEqual(a.device, 'cpu:0')
        self.assertDeviceEqual(a.initial_value.device, a.device)
        self.assertDeviceEqual(b.device, 'cpu:0')
        self.assertDeviceEqual(b.initial_value.device, b.device)


class GetVariablesCollections(tf.test.TestCase):

  def testVariableCollection(self):
    with self.test_session():
      a = tf.contrib.framework.variable('a', [], collections='A')
      b = tf.contrib.framework.variable('b', [], collections='B')
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])

  def testVariableCollections(self):
    with self.test_session():
      a = tf.contrib.framework.variable('a', [], collections=['A', 'C'])
      b = tf.contrib.framework.variable('b', [], collections=['B', 'C'])
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])
      self.assertListEqual([a, b], tf.get_collection('C'))

  def testVariableCollectionsWithArgScope(self):
    with self.test_session():
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          collections='A'):
        a = tf.contrib.framework.variable('a', [])
        b = tf.contrib.framework.variable('b', [])
      self.assertListEqual([a, b], tf.get_collection('A'))

  def testVariableCollectionsWithArgScopeNested(self):
    with self.test_session():
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          collections='A'):
        a = tf.contrib.framework.variable('a', [])
        with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                            collections='B'):
          b = tf.contrib.framework.variable('b', [])
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])

  def testVariableCollectionsWithArgScopeNonNested(self):
    with self.test_session():
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          collections='A'):
        a = tf.contrib.framework.variable('a', [])
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          collections='B'):
        b = tf.contrib.framework.variable('b', [])
      tf.contrib.framework.variable('c', [])
      self.assertListEqual([a], tf.get_collection('A'))
      self.assertListEqual([b], tf.get_collection('B'))

  def testVariableRestoreWithArgScopeNested(self):
    with self.test_session():
      a = tf.contrib.framework.variable('a', [])
      with tf.contrib.framework.arg_scope([tf.contrib.framework.variable],
                                          trainable=False,
                                          collections=['A', 'B']):
        b = tf.contrib.framework.variable('b', [])
      c = tf.contrib.framework.variable('c', [], trainable=False)
    self.assertEquals([a, c], tf.contrib.framework.get_variables_to_restore())
    self.assertEquals([a], tf.trainable_variables())
    self.assertEquals([b], tf.get_collection('A'))
    self.assertEquals([b], tf.get_collection('B'))


class GetVariablesBySuffixTest(tf.test.TestCase):

  def testGetVariableGivenNameScoped(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        b = tf.contrib.framework.variable('b', [5])
        self.assertEquals([a],
                          tf.contrib.framework.get_variables_by_suffix('a'))
        self.assertEquals([b],
                          tf.contrib.framework.get_variables_by_suffix('b'))

  def testGetVariableWithScope(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        fooa = tf.contrib.framework.variable('fooa', [5])
      with tf.variable_scope('B'):
        a2 = tf.contrib.framework.variable('a', [5])
      matched_variables = tf.contrib.framework.get_variables_by_suffix('a')
      self.assertEquals([a, fooa, a2], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_suffix('/a')
      self.assertEquals([a, a2], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_suffix(
          'a', scope='A')
      self.assertEquals([a, fooa], matched_variables)

  def testGetVariableWithoutScope(self):
    with self.test_session():
      a = tf.contrib.framework.variable('a', [5])
      fooa = tf.contrib.framework.variable('fooa', [5])
      b_a = tf.contrib.framework.variable('B/a', [5])
      matched_variables = tf.contrib.framework.get_variables_by_suffix('a')
      self.assertEquals([a, fooa, b_a], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_suffix('fooa')
      self.assertEquals([fooa], matched_variables)


class GetVariablesByNameTest(tf.test.TestCase):

  def testGetVariableGivenNameScoped(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        b = tf.contrib.framework.variable('b', [5])
        self.assertEquals([a], tf.contrib.framework.get_variables_by_name('a'))
        self.assertEquals([b], tf.contrib.framework.get_variables_by_name('b'))

  def testGetVariableWithScope(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = tf.contrib.framework.variable('a', [5])
        fooa = tf.contrib.framework.variable('fooa', [5])
      with tf.variable_scope('B'):
        a2 = tf.contrib.framework.variable('a', [5])
      matched_variables = tf.contrib.framework.get_variables_by_name('a')
      self.assertEquals([a, a2], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_name('fooa')
      self.assertEquals([fooa], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_name('/a')
      self.assertEquals([], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_name('a',
                                                                     scope='A')
      self.assertEquals([a], matched_variables)

  def testGetVariableWithoutScope(self):
    with self.test_session():
      a = tf.contrib.framework.variable('a', [5])
      fooa = tf.contrib.framework.variable('fooa', [5])
      b_a = tf.contrib.framework.variable('B/a', [5])
      matched_variables = tf.contrib.framework.get_variables_by_name('a')
      self.assertEquals([a, b_a], matched_variables)
      matched_variables = tf.contrib.framework.get_variables_by_name('fooa')
      self.assertEquals([fooa], matched_variables)


class AssignFromValuesTest(tf.test.TestCase):

  def testNoScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = tf.truncated_normal_initializer(stddev=.1)
      var0 = tf.contrib.framework.variables.variable(
          'my_var0', shape=[1, 3, 1], initializer=initializer)
      var1 = tf.contrib.framework.variables.variable(
          'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_var0': init_value0, 'my_var1': init_value1}
      assign_op, feed_dict = tf.contrib.framework.variables.assign_from_values(
          var_names_to_values)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

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
      initializer = tf.truncated_normal_initializer(stddev=.1)

      with tf.variable_scope('my_model/my_layer0'):
        var0 = tf.contrib.framework.variables.variable(
            'my_var0', shape=[1, 3, 1], initializer=initializer)
      with tf.variable_scope('my_model/my_layer1'):
        var1 = tf.contrib.framework.variables.variable(
            'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_model/my_layer0/my_var0': init_value0,
                             'my_model/my_layer1/my_var1': init_value1}
      assign_op, feed_dict = tf.contrib.framework.variables.assign_from_values(
          var_names_to_values)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      sess.run(assign_op, feed_dict)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)


class AssignFromValuesFnTest(tf.test.TestCase):

  def testNoScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0]).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0]).reshape((2, 1, 2))

    with self.test_session() as sess:
      initializer = tf.truncated_normal_initializer(stddev=.1)
      var0 = tf.contrib.framework.variable(
          'my_var0', shape=[1, 3, 1], initializer=initializer)
      var1 = tf.contrib.framework.variable(
          'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_var0': init_value0, 'my_var1': init_value1}
      init_fn = tf.contrib.framework.assign_from_values_fn(var_names_to_values)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

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
      initializer = tf.truncated_normal_initializer(stddev=.1)

      with tf.variable_scope('my_model/my_layer0'):
        var0 = tf.contrib.framework.variable(
            'my_var0', shape=[1, 3, 1], initializer=initializer)
      with tf.variable_scope('my_model/my_layer1'):
        var1 = tf.contrib.framework.variable(
            'my_var1', shape=[2, 1, 2], initializer=initializer)

      var_names_to_values = {'my_model/my_layer0/my_var0': init_value0,
                             'my_model/my_layer1/my_var1': init_value1}
      init_fn = tf.contrib.framework.assign_from_values_fn(var_names_to_values)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      var0, var1 = sess.run([var0, var1])
      self.assertAllEqual(init_value0, var0)
      self.assertAllEqual(init_value1, var1)


class AssignFromCheckpointTest(tf.test.TestCase):

  def create_checkpoint_from_values(self, var_names_to_values, checkpoint_dir,
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
    with tf.Session('', graph=tf.Graph()) as sess:
      # Create a set of variables to save in the checkpoint.
      for var_name in var_names_to_values:
        var_value = var_names_to_values[var_name]
        var_list.append(tf.Variable(var_value, name=var_name))
      saver = tf.train.Saver(var_list)
      init_op = tf.initialize_variables(var_list)
      sess.run(init_op)
      # Save the initialized values in the file at 'checkpoint_dir'
      return saver.save(sess, checkpoint_dir, global_step=global_step)

  def testLoadExistingVariables(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variables.variable('my_var0', shape=[])
      var1 = tf.contrib.framework.variables.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      op, feed_dict = tf.contrib.framework.variables.assign_from_checkpoint(
          model_path, vars_to_restore)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      sess.run(op, feed_dict)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testRaisesValueErrorIfAVariableIsntFound(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session():
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variables.variable('my_var0', shape=[])
      var1 = tf.contrib.framework.variables.variable('my_var1', shape=[])

      vars_to_restore = {'v0_fake': var0, 'v1': var1}

      with self.assertRaises(ValueError):
        tf.contrib.framework.variables.assign_from_checkpoint(model_path,
                                                              vars_to_restore)

  def testInitFromCheckpointWithScopes(self):
    init_value0 = np.asarray([1.0, 3.0, 9.0],
                             dtype=np.float32).reshape((1, 3, 1))
    init_value1 = np.asarray([2.0, 4.0, 6.0, 8.0],
                             dtype=np.float32).reshape((2, 1, 2))

    var_names_to_values = {'layer0/v0': init_value0, 'layer1/v1': init_value1}
    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      with tf.variable_scope('my_model/my_layer0'):
        var0 = tf.contrib.framework.variables.variable('my_var0',
                                                       shape=init_value0.shape)
      with tf.variable_scope('my_model/my_layer1'):
        var1 = tf.contrib.framework.variables.variable('my_var1',
                                                       shape=init_value1.shape)

      vars_to_restore = {'layer0/v0': var0, 'layer1/v1': var1}
      op, feed_dict = tf.contrib.framework.variables.assign_from_checkpoint(
          model_path,
          vars_to_restore)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      sess.run(op, feed_dict)

      # Request and test the variable values:
      self.assertAllEqual(init_value0, var0.eval())
      self.assertAllEqual(init_value1, var1.eval())


class AssignFromCheckpointFnTest(tf.test.TestCase):

  def create_checkpoint_from_values(self, var_names_to_values, checkpoint_dir,
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
    with tf.Session('', graph=tf.Graph()) as sess:
      # Create a set of variables to save in the checkpoint.
      for var_name in var_names_to_values:
        var_value = var_names_to_values[var_name]
        var_list.append(tf.Variable(var_value, name=var_name))
      saver = tf.train.Saver(var_list)
      init_op = tf.initialize_variables(var_list)
      sess.run(init_op)
      # Save the initialized values in the file at 'checkpoint_dir'
      return saver.save(sess, checkpoint_dir, global_step=global_step)

  def testLoadExistingVariables(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('my_var0', shape=[])
      var1 = tf.contrib.framework.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path, vars_to_restore)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testLoadExistingVariablesDifferentShapeDefaultDoesNotAllowReshape(self):
    init_value0 = [[10.0, 11.0]]
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('my_var0', shape=[2, 1])
      var1 = tf.contrib.framework.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path, vars_to_restore)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      with self.assertRaises(tf.errors.InvalidArgumentError):
        init_fn(sess)

  def testLoadExistingVariablesDifferentShapeAllowReshape(self):
    init_value0 = [[10.0, 11.0]]
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('my_var0', shape=[2, 1])
      var1 = tf.contrib.framework.variable('my_var1', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1}
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path, vars_to_restore, reshape_variables=True)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertAllEqual(np.transpose(np.array(init_value0)), var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testNotFoundError(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('my_var0', shape=[])
      var1 = tf.contrib.framework.variable('my_var1', shape=[])
      var2 = tf.contrib.framework.variable('my_var2', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1, 'v2': var2}
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path,
          vars_to_restore)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      with self.assertRaises(tf.errors.NotFoundError):
        init_fn(sess)

  def testMissingVariablesList(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('v0', shape=[])
      var1 = tf.contrib.framework.variable('v1', shape=[])
      var2 = tf.contrib.framework.variable('v2', shape=[])

      vars_to_restore = [var0, var1, var2]
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path,
          vars_to_restore,
          ignore_missing_vars=True)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

  def testMissingVariablesDict(self):
    init_value0 = 10.0
    init_value1 = 20.0
    var_names_to_values = {'v0': init_value0, 'v1': init_value1}

    model_dir = os.path.join(self.get_temp_dir(), 'model')
    with self.test_session() as sess:
      model_path = self.create_checkpoint_from_values(var_names_to_values,
                                                      model_dir)
      var0 = tf.contrib.framework.variable('my_var0', shape=[])
      var1 = tf.contrib.framework.variable('my_var1', shape=[])
      var2 = tf.contrib.framework.variable('my_var2', shape=[])

      vars_to_restore = {'v0': var0, 'v1': var1, 'v2': var2}
      init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path,
          vars_to_restore,
          ignore_missing_vars=True)

      # Initialize the variables.
      sess.run(tf.initialize_all_variables())

      # Perform the assignment.
      init_fn(sess)

      # Request and test the variable values:
      self.assertEqual(init_value0, var0.eval())
      self.assertEqual(init_value1, var1.eval())

if __name__ == '__main__':
  tf.test.main()
