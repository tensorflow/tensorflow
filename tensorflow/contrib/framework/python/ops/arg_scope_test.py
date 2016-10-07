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
"""arg_scope tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.contrib.framework.add_arg_scope
def func1(*args, **kwargs):
  return (args, kwargs)


@tf.contrib.framework.add_arg_scope
def func2(*args, **kwargs):
  return (args, kwargs)


@tf.contrib.framework.add_arg_scope
def func3(args, a=None, b=1, c=2):
  """Some cool doc string."""
  return (args, a, b, c)


def _key_op(op):
  return getattr(op, '_key_op', str(op))


class ArgScopeTest(tf.test.TestCase):

  def testEmptyArgScope(self):
    with self.test_session():
      with tf.contrib.framework.arg_scope([]) as sc:
        self.assertEqual(sc, {})

  def testClearArgScope(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    key_op = _key_op(func1)
    func1_scope = {key_op: func1_kwargs.copy()}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]) as sc1:
        self.assertEqual(sc1, func1_scope)
        with tf.contrib.framework.arg_scope({}) as sc2:
          self.assertEqual(sc2, {})
        with tf.contrib.framework.arg_scope([]) as current_arg_scope:
          self.assertEqual(current_arg_scope, func1_scope)

  def testNonDecorated(self):
    def my_func(t, a=None):
      return (t, a)
    with self.assertRaises(ValueError):
      with tf.contrib.framework.arg_scope([my_func], a=1):
        pass

  def testUnexpectedArg(self):
    with self.assertRaises(TypeError):
      with tf.contrib.framework.arg_scope([func3], d=1):
        func3(1)

  def testCurrentArgScope(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    key_op = _key_op(func1)
    current_scope = {key_op: func1_kwargs.copy()}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]) as scope:
        self.assertDictEqual(scope, current_scope)

  def testArgScopedArguments(self):
    func3_kwargs = ('a', 'b', 'c')
    self.assertEquals(tf.contrib.framework.arg_scoped_arguments(func3),
                      func3_kwargs)

  def testCurrentArgScopeNested(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_kwargs = {'b': 2, 'd': [2]}
    key = _key_op
    current_scope = {key(func1): func1_kwargs.copy(),
                     key(func2): func2_kwargs.copy()}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]):
        with tf.contrib.framework.arg_scope([func2], b=2, d=[2]) as scope:
          self.assertDictEqual(scope, current_scope)

  def testReuseArgScope(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    key_op = _key_op(func1)
    current_scope = {key_op: func1_kwargs.copy()}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1],
                                          a=1, b=None, c=[1]) as scope1:
        pass
      with tf.contrib.framework.arg_scope(scope1) as scope:
        self.assertDictEqual(scope, current_scope)

  def testReuseArgScopeNested(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_kwargs = {'b': 2, 'd': [2]}
    key = _key_op
    current_scope1 = {key(func1): func1_kwargs.copy()}
    current_scope2 = {key(func1): func1_kwargs.copy(),
                      key(func2): func2_kwargs.copy()}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1],
                                          a=1, b=None, c=[1]) as scope1:
        with tf.contrib.framework.arg_scope([func2], b=2, d=[2]) as scope2:
          pass
      with tf.contrib.framework.arg_scope(scope1):
        with tf.contrib.framework.arg_scope([]) as current_arg_scope:
          self.assertDictEqual(current_arg_scope, current_scope1)
      with tf.contrib.framework.arg_scope(scope2):
        with tf.contrib.framework.arg_scope([]) as current_arg_scope:
          self.assertDictEqual(current_arg_scope, current_scope2)

  def testSimpleArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with self.test_session():
      with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testSimpleArgScopeWithTuple(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with self.test_session():
      with tf.contrib.framework.arg_scope((func1,), a=1, b=None, c=[1]):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testOverwriteArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': 2, 'c': [1]}
    with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]):
      args, kwargs = func1(0, b=2)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testNestedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with tf.contrib.framework.arg_scope([func1], a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      func1_kwargs['b'] = 2
      with tf.contrib.framework.arg_scope([func1], b=2):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testSharedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with tf.contrib.framework.arg_scope([func1, func2], a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      args, kwargs = func2(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testSharedArgScopeTuple(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with tf.contrib.framework.arg_scope((func1, func2), a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      args, kwargs = func2(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testPartiallySharedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_args = (1,)
    func2_kwargs = {'a': 1, 'b': None, 'd': [2]}
    with tf.contrib.framework.arg_scope([func1, func2], a=1, b=None):
      with tf.contrib.framework.arg_scope([func1], c=[1]):
        with tf.contrib.framework.arg_scope([func2], d=[2]):
          args, kwargs = func1(0)
          self.assertTupleEqual(args, func1_args)
          self.assertDictEqual(kwargs, func1_kwargs)
          args, kwargs = func2(1)
          self.assertTupleEqual(args, func2_args)
          self.assertDictEqual(kwargs, func2_kwargs)

  def testDocString(self):
    self.assertEqual(func3.__doc__, 'Some cool doc string.')


if __name__ == '__main__':
  tf.test.main()
