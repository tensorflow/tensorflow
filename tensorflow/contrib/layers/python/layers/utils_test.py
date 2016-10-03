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
"""Tests for regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import utils


class ConstantValueTest(tf.test.TestCase):

  def test_value(self):
    for v in [True, False, 1, 0, 1.0]:
      value = utils.constant_value(v)
      self.assertEqual(value, v)

  def test_constant(self):
    for v in [True, False, 1, 0, 1.0]:
      c = tf.constant(v)
      value = utils.constant_value(c)
      self.assertEqual(value, v)
      with self.test_session():
        self.assertEqual(c.eval(), v)

  def test_variable(self):
    for v in [True, False, 1, 0, 1.0]:
      with tf.Graph().as_default() as g, self.test_session(g) as sess:
        x = tf.Variable(v)
        value = utils.constant_value(x)
        self.assertEqual(value, None)
        sess.run(tf.initialize_all_variables())
        self.assertEqual(x.eval(), v)

  def test_placeholder(self):
    for v in [True, False, 1, 0, 1.0]:
      p = tf.placeholder(np.dtype(type(v)), [])
      x = tf.identity(p)
      value = utils.constant_value(p)
      self.assertEqual(value, None)
      with self.test_session():
        self.assertEqual(x.eval(feed_dict={p: v}), v)


class StaticCondTest(tf.test.TestCase):

  def test_value(self):
    fn1 = lambda: 'fn1'
    fn2 = lambda: 'fn2'
    expected = lambda v: 'fn1' if v else 'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      self.assertEqual(o, expected(v))

  def test_constant(self):
    fn1 = lambda: tf.constant('fn1')
    fn2 = lambda: tf.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))

  def test_variable(self):
    fn1 = lambda: tf.Variable('fn1')
    fn2 = lambda: tf.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session() as sess:
        sess.run(tf.initialize_all_variables())
        self.assertEqual(o.eval(), expected(v))

  def test_tensors(self):
    fn1 = lambda: tf.constant(0) - tf.constant(1)
    fn2 = lambda: tf.constant(0) - tf.constant(2)
    expected = lambda v: -1 if v else -2
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))


class SmartCondStaticTest(tf.test.TestCase):

  def test_value(self):
    fn1 = lambda: 'fn1'
    fn2 = lambda: 'fn2'
    expected = lambda v: 'fn1' if v else 'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(tf.constant(v), fn1, fn2)
      self.assertEqual(o, expected(v))

  def test_constant(self):
    fn1 = lambda: tf.constant('fn1')
    fn2 = lambda: tf.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(tf.constant(v), fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))

  def test_variable(self):
    fn1 = lambda: tf.Variable('fn1')
    fn2 = lambda: tf.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(tf.constant(v), fn1, fn2)
      with self.test_session() as sess:
        sess.run(tf.initialize_all_variables())
        self.assertEqual(o.eval(), expected(v))

  def test_tensors(self):
    fn1 = lambda: tf.constant(0) - tf.constant(1)
    fn2 = lambda: tf.constant(0) - tf.constant(2)
    expected = lambda v: -1 if v else -2
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(tf.constant(v), fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))


class SmartCondDynamicTest(tf.test.TestCase):

  def test_value(self):
    fn1 = lambda: tf.convert_to_tensor('fn1')
    fn2 = lambda: tf.convert_to_tensor('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = tf.placeholder(tf.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_constant(self):
    fn1 = lambda: tf.constant('fn1')
    fn2 = lambda: tf.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = tf.placeholder(tf.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_variable(self):
    fn1 = lambda: tf.Variable('fn1')
    fn2 = lambda: tf.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = tf.placeholder(tf.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session() as sess:
        sess.run(tf.initialize_all_variables())
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_tensors(self):
    fn1 = lambda: tf.constant(0) - tf.constant(1)
    fn2 = lambda: tf.constant(0) - tf.constant(2)
    expected = lambda v: -1 if v else -2
    p = tf.placeholder(tf.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))


class CollectNamedOutputsTest(tf.test.TestCase):

  def test_collect(self):
    t1 = tf.constant(1.0, name='t1')
    t2 = tf.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    self.assertEqual(tf.get_collection('end_points'), [t1, t2])

  def test_aliases(self):
    t1 = tf.constant(1.0, name='t1')
    t2 = tf.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    self.assertEqual(t1.alias, 'a1')
    self.assertEqual(t2.alias, 'a2')

  def test_gather_aliases(self):
    t1 = tf.constant(1.0, name='t1')
    t2 = tf.constant(2.0, name='t2')
    t3 = tf.constant(2.0, name='t3')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    tf.add_to_collection('end_points', t3)
    aliases = utils.gather_tensors_alias(tf.get_collection('end_points'))
    self.assertListEqual(aliases, ['a1', 'a2', 't3'])


if __name__ == '__main__':
  tf.test.main()
