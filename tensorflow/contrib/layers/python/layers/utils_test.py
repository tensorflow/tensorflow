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

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ConstantValueTest(test.TestCase):

  def test_value(self):
    for v in [True, False, 1, 0, 1.0]:
      value = utils.constant_value(v)
      self.assertEqual(value, v)

  def test_constant(self):
    for v in [True, False, 1, 0, 1.0]:
      c = constant_op.constant(v)
      value = utils.constant_value(c)
      self.assertEqual(value, v)
      with self.test_session():
        self.assertEqual(c.eval(), v)

  def test_variable(self):
    for v in [True, False, 1, 0, 1.0]:
      with ops.Graph().as_default() as g, self.test_session(g) as sess:
        x = variables.Variable(v)
        value = utils.constant_value(x)
        self.assertEqual(value, None)
        sess.run(variables.global_variables_initializer())
        self.assertEqual(x.eval(), v)

  def test_placeholder(self):
    for v in [True, False, 1, 0, 1.0]:
      p = array_ops.placeholder(np.dtype(type(v)), [])
      x = array_ops.identity(p)
      value = utils.constant_value(p)
      self.assertEqual(value, None)
      with self.test_session():
        self.assertEqual(x.eval(feed_dict={p: v}), v)


class StaticCondTest(test.TestCase):

  def test_value(self):
    fn1 = lambda: 'fn1'
    fn2 = lambda: 'fn2'
    expected = lambda v: 'fn1' if v else 'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      self.assertEqual(o, expected(v))

  def test_constant(self):
    fn1 = lambda: constant_op.constant('fn1')
    fn2 = lambda: constant_op.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))

  def test_variable(self):
    fn1 = lambda: variables.Variable('fn1')
    fn2 = lambda: variables.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertEqual(o.eval(), expected(v))

  def test_tensors(self):
    fn1 = lambda: constant_op.constant(0) - constant_op.constant(1)
    fn2 = lambda: constant_op.constant(0) - constant_op.constant(2)
    expected = lambda v: -1 if v else -2
    for v in [True, False, 1, 0]:
      o = utils.static_cond(v, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))


class SmartCondStaticTest(test.TestCase):

  def test_value(self):
    fn1 = lambda: 'fn1'
    fn2 = lambda: 'fn2'
    expected = lambda v: 'fn1' if v else 'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(constant_op.constant(v), fn1, fn2)
      self.assertEqual(o, expected(v))

  def test_constant(self):
    fn1 = lambda: constant_op.constant('fn1')
    fn2 = lambda: constant_op.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(constant_op.constant(v), fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))

  def test_variable(self):
    fn1 = lambda: variables.Variable('fn1')
    fn2 = lambda: variables.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(constant_op.constant(v), fn1, fn2)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertEqual(o.eval(), expected(v))

  def test_tensors(self):
    fn1 = lambda: constant_op.constant(0) - constant_op.constant(1)
    fn2 = lambda: constant_op.constant(0) - constant_op.constant(2)
    expected = lambda v: -1 if v else -2
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(constant_op.constant(v), fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(), expected(v))


class SmartCondDynamicTest(test.TestCase):

  def test_value(self):
    fn1 = lambda: ops.convert_to_tensor('fn1')
    fn2 = lambda: ops.convert_to_tensor('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = array_ops.placeholder(dtypes.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_constant(self):
    fn1 = lambda: constant_op.constant('fn1')
    fn2 = lambda: constant_op.constant('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = array_ops.placeholder(dtypes.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_variable(self):
    fn1 = lambda: variables.Variable('fn1')
    fn2 = lambda: variables.Variable('fn2')
    expected = lambda v: b'fn1' if v else b'fn2'
    p = array_ops.placeholder(dtypes.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))

  def test_tensors(self):
    fn1 = lambda: constant_op.constant(0) - constant_op.constant(1)
    fn2 = lambda: constant_op.constant(0) - constant_op.constant(2)
    expected = lambda v: -1 if v else -2
    p = array_ops.placeholder(dtypes.bool, [])
    for v in [True, False, 1, 0]:
      o = utils.smart_cond(p, fn1, fn2)
      with self.test_session():
        self.assertEqual(o.eval(feed_dict={p: v}), expected(v))


class CollectNamedOutputsTest(test.TestCase):

  def test_collect(self):
    t1 = constant_op.constant(1.0, name='t1')
    t2 = constant_op.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    self.assertEqual(ops.get_collection('end_points'), [t1, t2])

  def test_aliases(self):
    t1 = constant_op.constant(1.0, name='t1')
    t2 = constant_op.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    self.assertEqual(t1.aliases, ['a1'])
    self.assertEqual(t2.aliases, ['a2'])

  def test_multiple_aliases(self):
    t1 = constant_op.constant(1.0, name='t1')
    t2 = constant_op.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a11', t1)
    utils.collect_named_outputs('end_points', 'a12', t1)
    utils.collect_named_outputs('end_points', 'a21', t2)
    utils.collect_named_outputs('end_points', 'a22', t2)
    self.assertEqual(t1.aliases, ['a11', 'a12'])
    self.assertEqual(t2.aliases, ['a21', 'a22'])

  def test_gather_aliases(self):
    t1 = constant_op.constant(1.0, name='t1')
    t2 = constant_op.constant(2.0, name='t2')
    t3 = constant_op.constant(2.0, name='t3')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a2', t2)
    ops.add_to_collection('end_points', t3)
    aliases = utils.gather_tensors_aliases(ops.get_collection('end_points'))
    self.assertEqual(aliases, ['a1', 'a2', 't3'])

  def test_convert_collection_to_dict(self):
    t1 = constant_op.constant(1.0, name='t1')
    t2 = constant_op.constant(2.0, name='t2')
    utils.collect_named_outputs('end_points', 'a1', t1)
    utils.collect_named_outputs('end_points', 'a21', t2)
    utils.collect_named_outputs('end_points', 'a22', t2)
    end_points = utils.convert_collection_to_dict('end_points')
    self.assertEqual(end_points['a1'], t1)
    self.assertEqual(end_points['a21'], t2)
    self.assertEqual(end_points['a22'], t2)


class NPositiveIntegersTest(test.TestCase):

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      utils.n_positive_integers('3', [1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(3.3, [1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(-1, [1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(0, [1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(1, [1, 2])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(1, [-1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(1, [0])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(1, [0])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(2, [1])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(2, [1, 2, 3])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(2, ['hello', 2])

    with self.assertRaises(ValueError):
      utils.n_positive_integers(2, tensor_shape.TensorShape([2, 3, 1]))

    with self.assertRaises(ValueError):
      utils.n_positive_integers(3, tensor_shape.TensorShape([2, None, 1]))

    with self.assertRaises(ValueError):
      utils.n_positive_integers(3, tensor_shape.TensorShape(None))

  def test_valid_input(self):
    self.assertEqual(utils.n_positive_integers(1, 2), (2,))
    self.assertEqual(utils.n_positive_integers(2, 2), (2, 2))
    self.assertEqual(utils.n_positive_integers(2, (2, 3)), (2, 3))
    self.assertEqual(utils.n_positive_integers(3, (2, 3, 1)), (2, 3, 1))
    self.assertEqual(utils.n_positive_integers(3, (2, 3, 1)), (2, 3, 1))
    self.assertEqual(
        utils.n_positive_integers(3, tensor_shape.TensorShape([2, 3, 1])),
        (2, 3, 1))


if __name__ == '__main__':
  test.main()
