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
"""Tests for control_flow module."""

# Unfortunately pylint has false positives when nonlocal is present.
# pylint:disable=unused-variable

import collections
import re
import sys

import numpy as np
import six

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import variables as variable_operators
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _unranked_item(value):
  rand_rank = random_ops.random_uniform(
      shape=(), minval=3, maxval=4, dtype=dtypes.int32)
  rand_shape = array_ops.ones([rand_rank], dtype=dtypes.int32)
  return array_ops.fill(rand_shape, value)


def _partial_shaped_bools():
  rand_vect = math_ops.range(
      random_ops.random_uniform(
          shape=(), minval=2, maxval=3, dtype=dtypes.int32))
  return array_ops.expand_dims_v2(rand_vect, 0) < 0


class ForLoopTest(testing.AutoGraphTestCase):

  def test_tensor(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    self.assertEqual(s, (1234,))

  def test_range_tensor(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_explicit_limit_delta(self):
    def body(i):
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(-17, -3, 5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (-171207,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_explicit_limit_negative_delta(self):
    def body(i):
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        math_ops.range(17, 3, -5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (171207,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_random_delta(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    random_one = random_ops.random_uniform((), 1, 2, dtype=dtypes.int32)
    control_flow.for_stmt(
        math_ops.range(0, 5, random_one),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('StatelessWhile')

  def test_range_tensor_random_negative_delta(self):
    def body(i):
      nonlocal s
      s = s * 100 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    random_neg_five = random_ops.random_uniform((), -5, -4, dtype=dtypes.int32)
    control_flow.for_stmt(
        math_ops.range(17, 3, random_neg_five),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'iterate_names': 'i'})

    self.assertEqual(s, (171207,))
    self.assertOpCreated('StatelessWhile')

  def test_tensor_with_extra_test_object_vars(self):
    class MutableObject(object):
      field_1 = constant_op.constant(0, dtype=dtypes.int32)
      field_2 = constant_op.constant(1, dtype=dtypes.int32)
    state = MutableObject()

    def body(i):
      state.field_1 += i
      state.field_2 *= i

    def get_state():
      return state.field_1, state.field_2

    def set_state(loop_vars):
      state.field_1, state.field_2 = loop_vars

    control_flow.for_stmt(
        iter_=constant_op.constant([1, 2, 3, 4]),
        body=body,
        extra_test=lambda: state.field_1 < 6,
        get_state=get_state,
        set_state=set_state,
        symbol_names=('state.field_1', 'state.field_2'),
        opts={})

    self.assertEqual((state.field_1, state.field_2), (6, 6))
    self.assertOpCreated('StatelessWhile')

  def test_python(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        range(5),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, 1234)
    self.assertNoOpsCreated()

  def test_python_generator_with_extra_test(self):
    def new_generator():
      for i in range(1, 5):
        yield i

    gen = new_generator()
    def run_loop():
      s = 0
      c = 0

      def body(i):
        nonlocal s, c
        s = s * 10 + i
        c += 1

      control_flow.for_stmt(
          gen,
          extra_test=lambda: c == 0,  # Break after first iteration
          body=body,
          get_state=None,
          set_state=None,
          symbol_names=('s', 'c'),
          opts={})
      return s, c

    self.assertEqual(run_loop(), (1, 1))
    self.assertEqual(run_loop(), (2, 1))
    self.assertEqual(run_loop(), (3, 1))

    self.assertEqual(next(gen), 4)

    self.assertNoOpsCreated()

  def test_python_generator_with_extra_test_no_iterations(self):
    def new_generator():
      for i in range(5):
        yield i

    gen = new_generator()
    def run_loop():
      s = 0

      def body(i):
        nonlocal s
        s = s * 10 + i

      control_flow.for_stmt(
          gen,
          extra_test=lambda: False,  # Break before loop
          body=body,
          get_state=None,
          set_state=None,
          symbol_names=('s',),
          opts={})
      return s

    self.assertEqual(run_loop(), 0)
    self.assertEqual(run_loop(), 0)

    self.assertEqual(next(gen), 0)

    self.assertNoOpsCreated()

  def test_tf_dataset(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (1234,))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: s < 3,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (12,))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test_collection_vars(self):
    def body(i):
      nonlocal s
      l[0] += i
      s += i

    def set_state(loop_vars):
      nonlocal s
      l[0], s = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    l = [constant_op.constant(0, dtype=dtypes.int64)]
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: s < 3,
        body=body,
        get_state=lambda: (l[0], s),
        set_state=set_state,
        symbol_names=('l[0]', 's'),
        opts={})

    self.assertEqual((l[0], s), (3, 3))
    self.assertOpCreated('ScanDataset')

  def test_dataset_with_extra_test_iteration_limiting(self):
    def body(it):
      nonlocal i
      with ops.control_dependencies((control_flow_ops.Assert(i < 3, (i,)),)):
        i = it

    def set_state(loop_vars):
      nonlocal i
      i, = loop_vars

    i = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda: i < 3,
        body=body,
        get_state=lambda: (i,),
        set_state=set_state,
        symbol_names=('i',),
        opts={})

    self.assertEqual(i, (3,))
    self.assertOpCreated('ScanDataset')

  def test_tf_dataset_no_loop_vars(self):
    def body(i):
      v.assign(v.read_value() * 10 + i)

    v = self.variable('v', 0, dtypes.int64)

    control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    self.assertEqual(v.read_value(), 1234)
    self.assertOpCreated('ScanDataset')

  def test_tf_iterator(self):
    def body(i):
      nonlocal s
      s = s * 10 + i

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0, dtype=dtypes.int64)
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, 1234)
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_shape_invariants(self):
    def body(i):
      nonlocal s
      s = array_ops.concat([s, [i]], 0)

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = constant_op.constant([], dtype=dtypes.int64)
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={'shape_invariants': [(s, tensor_shape.TensorShape([None]))]})

    self.assertAllEqual(s, [0, 1, 2, 3, 4])
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_shape_invariants_with_nested_structures(self):
    def body(i):
      nonlocal s
      nonlocal t
      s = array_ops.concat([s, [i]], 0)
      t = Test(var=t.var + 1)

    def set_state(loop_vars):
      nonlocal s
      nonlocal t
      s, t = loop_vars

    s = constant_op.constant([], dtype=dtypes.int64)
    Test = collections.namedtuple('Test', ['var'])
    t = Test(var=constant_op.constant([0], dtype=dtypes.int64))
    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (s, t),
        set_state=set_state,
        symbol_names=('s', 't'),
        opts={'shape_invariants': [(s, tensor_shape.TensorShape([None]))]})

    self.assertAllEqual(s, [0, 1, 2, 3, 4])
    self.assertEqual(t.var, [5])
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_iterator_no_loop_vars(self):
    def body(i):
      v.assign(v.read_value() * 10 + i)

    v = self.variable('v', 0, dtypes.int64)

    control_flow.for_stmt(
        iter(dataset_ops.Dataset.range(5)),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    self.assertEqual(v.read_value(), 1234)
    self.assertOpCreated('IteratorGetNextAsOptional')

  def test_tf_ragged_tensor(self):
    def body(i):
      nonlocal s
      s = s * 10 + i[0]

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    control_flow.for_stmt(
        ragged_factory_ops.constant([[1], [2, 4], [3]]),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (123,))
    self.assertOpCreated('StatelessWhile')

  def test_tf_ragged_tensor_higher_dimensional(self):
    def body(i):
      nonlocal s
      s = s * 10 + i[0][0]

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = 0
    ragged_3d = [
        [[1], [1, 1], [1]],
        [[2], [2]],
    ]
    control_flow.for_stmt(
        ragged_factory_ops.constant(ragged_3d),
        extra_test=None,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})

    self.assertEqual(s, (12,))
    self.assertOpCreated('StatelessWhile')

  def test_tf_ragged_tensor_no_loop_vars(self):
    v = self.variable('v', 0, dtypes.int32)

    def body(i):
      v.assign(v.read_value() * 10 + i[0])

    control_flow.for_stmt(
        ragged_factory_ops.constant([[1], [2, 4], [3]]),
        extra_test=None,
        body=body,
        get_state=lambda: (),
        set_state=lambda _: None,
        symbol_names=(),
        opts={})

    # Note: 123 = ((0*10 + 1)*10+2)*10+3 (first element of each row).
    self.assertEqual(v.read_value(), 123)
    self.assertOpCreated('While')

  def _basic_loop(self, init_value, body_fn):
    def body(i):
      nonlocal s
      s = body_fn(i, s)

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = init_value
    control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda: True,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    return s

  def test_tensor_illegal_input(self):
    with self.assertRaisesRegex(ValueError, '\'s\' may not be None'):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, '\'s\' must be defined'):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(ValueError, '\'s\' is None at the end'):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(TypeError, '\'s\'.* dtype float32 after'):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    with self.assertRaisesRegex(ValueError, r'\'s\'.* shape \(1,\) after'):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))


class WhileLoopTest(testing.AutoGraphTestCase):

  def test_tensor(self):

    def body():
      nonlocal i, s
      s = s * 10 + i
      i += 1

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = 0
    n = constant_op.constant(5)
    s = 0
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(s, 1234)
    self.assertOpCreated('StatelessWhile')

  def test_tensor_creating_variable(self):

    def body():
      nonlocal i, s
      i = constant_op.constant(2)
      s = i ** 5

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 2)
    self.assertEqual(s, 32)
    self.assertOpCreated('StatelessWhile')
    # Check that the temporary staging of the body did not create extra ops.
    # Node naming is inconsistent between V1 and V2.
    self.assertGraphContains(r'(while/)?pow$', 1)

  def test_tensor_creating_dynamic_shape_variable(self):

    def body():
      nonlocal i, y
      i += 1
      y = random_ops.random_uniform([i])

    def set_state(loop_vars):
      nonlocal i, y
      i, y = loop_vars

    i = constant_op.constant(0)
    y = variable_operators.Undefined('y')
    control_flow.while_stmt(
        test=lambda: math_ops.less(i, 3),
        body=body,
        get_state=lambda: (i, y),
        set_state=set_state,
        symbol_names=('i', 'y'),
        opts={})

    self.assertEqual(i, 3)
    self.assertLess(y[0], 3)

  def test_tensor_creating_dynamic_shape_variable_preserves_shape_invar(self):

    def body():
      nonlocal i, y
      i += 1
      y = array_ops.zeros([1])

    def set_state(loop_vars):
      nonlocal i, y
      i, y = loop_vars

    i = constant_op.constant(0)
    y = variable_operators.Undefined('y')
    control_flow.while_stmt(
        test=lambda: math_ops.less(i, 3),
        body=body,
        get_state=lambda: (i, y),
        set_state=set_state,
        symbol_names=('i', 'y'),
        opts={'shape_invariants': ((y, tensor_shape.TensorShape([1])),)})

    self.evaluate(y)

  def test_tensor_creating_complex_variable(self):

    def body():
      nonlocal i, s
      i = {'a': constant_op.constant(2), 'b': {'c': constant_op.constant(1)}}
      s = i['a'] ** 5

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertDictEqual(i, {'a': 2, 'b': {'c': 1}})
    self.assertEqual(s, 32)
    self.assertOpCreated('StatelessWhile')
    # Check that the temporary staging of the body did not create extra ops.
    # Node naming is inconsistent between V1 and V2.
    self.assertGraphContains(r'(while/)?pow$', 1)

  def test_tensor_creating_variable_of_dynamic_shape(self):

    def body():
      nonlocal i, s
      i = array_ops.ones(
          [random_ops.random_uniform(minval=1, maxval=4, shape=()), 7])
      s = math_ops.reduce_sum(i)

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0.0)
    control_flow.while_stmt(
        test=lambda: math_ops.equal(s, 0),
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i[0][0], 1)
    self.assertGreaterEqual(s, 7)
    self.assertOpCreated('While')  # Not stateless because of the random op.

  def test_tensor_with_side_effecting_condition(self):
    v = self.variable('v', 0, dtypes.int32)

    def cond():
      v.assign(v.read_value() * 10 + i)
      return i < n

    def body():
      nonlocal i
      i += 1

    def set_state(loop_vars):
      nonlocal i
      i, = loop_vars

    i = 0
    n = constant_op.constant(5)
    control_flow.while_stmt(
        test=cond,
        body=body,
        get_state=lambda: (i,),
        set_state=set_state,
        symbol_names=('i',),
        opts={})

    self.assertEqual(i, (5,))
    self.assertEqual(v, (12345,))
    self.assertOpCreated('While')

  def test_tensor_failing_to_determine_placeholder(self):

    class UserType:
      pass

    def body():
      nonlocal v
      v = UserType()

    def set_state(loop_vars):
      nonlocal v
      v, = loop_vars

    v = variable_operators.Undefined('v')

    with self.assertRaisesRegex(
        ValueError,
        re.compile('must be defined.*tried to define.*unsupported type',
                   re.DOTALL)):
      control_flow.while_stmt(
          test=lambda: constant_op.constant(True),
          body=body,
          get_state=lambda: (v,),
          set_state=set_state,
          symbol_names=('v',),
          opts={})

  def test_tensor_failing_to_stage_loop_body(self):

    def body():
      nonlocal i, s
      i = constant_op.constant(2)
      raise ValueError('testing')
      s = i ** 5  # pylint: disable=unreachable

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = variable_operators.Undefined('i')
    s = constant_op.constant(0)

    with self.assertRaisesRegex(
        ValueError,
        re.compile('must be defined.*tried to define.*testing', re.DOTALL)):
      control_flow.while_stmt(
          test=lambda: math_ops.equal(s, 0),
          body=body,
          get_state=lambda: (i, s),
          set_state=set_state,
          symbol_names=('i', 's'),
          opts={})

  def test_tensor_with_python_state(self):
    class MutableObject(object):
      field = constant_op.constant(0, dtype=dtypes.int32)
    state = MutableObject()

    def body():
      nonlocal i
      state.field = state.field * 10 + i
      i += 1

    def set_state(loop_vars):
      nonlocal i
      i, state.field = loop_vars

    i = 0
    n = constant_op.constant(5)
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, state.field),
        set_state=set_state,
        symbol_names=('i', 'state.field'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(state.field, 1234)
    self.assertOpCreated('StatelessWhile')

  def test_python(self):
    def body():
      nonlocal i, s
      s = s * 10 + i
      i += 1

    i = 0
    s = 0
    n = 5
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=None,
        set_state=None,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(s, 1234)
    self.assertNoOpsCreated()

  def test_python_with_tensor_state(self):
    def body():
      nonlocal i, s
      s = s * 10 + i
      i += 1

    i = 0
    s = constant_op.constant(0)
    n = 5
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=None,
        set_state=None,
        symbol_names=('i', 's'),
        opts={})

    self.assertEqual(i, 5)
    self.assertEqual(s, 1234)
    self.assertOpsNotCreated(('While', 'StatelessWhile'))

  def test_python_while_infinite(self):
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
      with self.assertRaisesRegex(ValueError, 'iteration limit'):
        control_flow.while_stmt(
            test=lambda: True,
            body=lambda: None,
            get_state=None,
            set_state=None,
            symbol_names=(),
            opts={})

  def test_python_for_infinite(self):
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
      with self.assertRaisesRegex(ValueError, 'iteration limit'):
        control_flow.for_stmt(
            iter_=range(101),
            extra_test=None,
            body=lambda i: None,
            get_state=None,
            set_state=None,
            symbol_names=(),
            opts={})

  def test_python_while_large_unroll_warning(self):
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(
        control_flow, 'INEFFICIENT_UNROLL_MIN_ITERATIONS', 10):
      with ops.Graph().as_default():
        out_capturer = six.StringIO()
        with test.mock.patch.object(sys, 'stdout', out_capturer):
          with test.mock.patch.object(ag_logging, 'echo_log_to_stdout', True):
            def custom_iterator():
              for i in range(11):
                c = constant_op.constant(i)
                yield c

            i = 0
            control_flow.for_stmt(
                iter_=custom_iterator(),
                extra_test=None,
                body=lambda i: None,
                get_state=None,
                set_state=None,
                symbol_names=(),
                opts={})
        self.assertTrue(re.match(
            r'.* Large unrolled loop.*Const.*', out_capturer.getvalue()))

  def test_python_for_large_unroll_warning(self):
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(
        control_flow, 'INEFFICIENT_UNROLL_MIN_ITERATIONS', 10):
      with ops.Graph().as_default():
        out_capturer = six.StringIO()
        with test.mock.patch.object(sys, 'stdout', out_capturer):
          with test.mock.patch.object(ag_logging, 'echo_log_to_stdout', True):
            def body():
              nonlocal i
              gen_math_ops.add(i, 1)
              i += 1

            i = 0
            control_flow.while_stmt(
                test=lambda: i < 100,
                body=body,
                get_state=None,
                set_state=None,
                symbol_names=('i',),
                opts={})
        self.assertTrue(re.match(
            r'.* Large unrolled loop.*Add.*', out_capturer.getvalue()))

  def _basic_loop(self, init_value, body_fn):

    def body():
      nonlocal i, s
      s = body_fn(i, s)
      i += 1

    def set_state(loop_vars):
      nonlocal i, s
      i, s = loop_vars

    i = 0
    n = constant_op.constant(5)
    s = init_value
    control_flow.while_stmt(
        test=lambda: i < n,
        body=body,
        get_state=lambda: (i, s),
        set_state=set_state,
        symbol_names=('i', 's'),
        opts={})
    return s

  def test_tensor_illegal_input(self):
    with self.assertRaisesRegex(ValueError, "'s' may not be None"):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, "'s' must be defined"):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(ValueError, "'s' is None at the end"):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(TypeError, "'s'.* dtype float32 after"):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    with self.assertRaisesRegex(ValueError, r"'s'.* shape \(1,\) after"):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))

  def _fixed_while_loop(self, cond_fn):
    def test_():
      return cond_fn(s)

    def body():
      nonlocal s
      s += 1

    def set_state(loop_vars):
      nonlocal s
      s, = loop_vars

    s = constant_op.constant(0)
    control_flow.while_stmt(
        test=test_,
        body=body,
        get_state=lambda: (s,),
        set_state=set_state,
        symbol_names=('s',),
        opts={})
    return s

  def _assertFixedLoopResult(self, cond, expected):
    def test_fn():
      return self._fixed_while_loop(cond)
    self.assertEqual(test_fn(), expected)

  def test_tensor_legal_cond_scalar(self):
    self._assertFixedLoopResult(lambda s: constant_op.constant(False), 0)
    self._assertFixedLoopResult(lambda s: s < 2, 2)

  def test_tensor_legal_cond_single_element_nd(self):
    self._assertFixedLoopResult(lambda s: constant_op.constant([[False]]), 0)
    self._assertFixedLoopResult(lambda s: _unranked_item(False), 0)

  def _assertCondCheckFails(self, cond):
    with self.assertRaisesRegex(
        ValueError, 'condition of while loop expected to be `tf.bool`'):
      self._fixed_while_loop(cond)

  def test_tensor_illegal_cond_not_bool(self):
    self._assertCondCheckFails(lambda s: constant_op.constant(1))
    self._assertCondCheckFails(lambda s: s)

  def test_tensor_illegal_cond_not_single_element(self):
    self._assertCondCheckFails(lambda s: constant_op.constant([1, 2, 3]))
    self._assertCondCheckFails(lambda s: constant_op.constant([True, False]))

  def test_tensor_illegal_cond_not_single_element_dynamic_shape(self):
    self._fixed_while_loop(lambda s: _partial_shaped_bools())
    # TODO(mdan): This error is quite bad. Measure the cost of an assertion.
    self.assertRaisesRuntime(
        errors_impl.InvalidArgumentError, 'requested shape has 1')


class IfStmtTest(testing.AutoGraphTestCase):

  def test_tensor(self):

    def test_fn(cond):
      def body():
        nonlocal i
        i = constant_op.constant(1)

      def orelse():
        nonlocal i
        i = constant_op.constant(-1)

      def set_state(cond_vars):
        nonlocal i
        i, = cond_vars

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i,),
          set_state=set_state,
          symbol_names=('i',),
          nouts=1)
      return i

    self.assertEqual(test_fn(constant_op.constant(True)), 1)
    self.assertEqual(test_fn(constant_op.constant(False)), -1)
    self.assertOpCreated('StatelessIf')

  def test_tensor_no_outputs(self):

    def test_fn(cond):
      def body():
        nonlocal i
        i = constant_op.constant(1)

      def orelse():
        nonlocal i
        i = constant_op.constant(-1.0)

      def set_state(cond_vars):
        nonlocal i
        i, = cond_vars

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i,),
          set_state=set_state,
          symbol_names=('i',),
          nouts=0)
      return i

    self.assertIsNone(test_fn(constant_op.constant(True)))
    self.assertIsNone(test_fn(constant_op.constant(False)))
    self.assertOpCreated('StatelessIf')

  def test_tensor_multiple_returns(self):

    def test_fn(cond):
      def body():
        nonlocal i, j
        i = constant_op.constant(1)
        j = constant_op.constant(2)

      def orelse():
        nonlocal i, j
        i = constant_op.constant(-1)
        j = constant_op.constant(-2)

      def set_state(cond_vars):
        nonlocal i, j
        i, j = cond_vars

      i, j = None, None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=lambda: (i, j),
          set_state=set_state,
          symbol_names=('i', 'j'),
          nouts=2)
      return i, j

    self.assertEqual(test_fn(constant_op.constant(True)), (1, 2))
    self.assertEqual(test_fn(constant_op.constant(False)), (-1, -2))
    self.assertOpCreated('StatelessIf')

  def test_python(self):

    def test_fn(cond):
      def body():
        nonlocal i
        i = 1

      def orelse():
        nonlocal i
        i = -1

      i = None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=None,
          set_state=None,
          symbol_names=('i',),
          nouts=1)
      return i

    self.assertEqual(test_fn(True), 1)
    self.assertEqual(test_fn(False), -1)
    self.assertNoOpsCreated()

  def test_python_multiple_returns(self):

    def test_fn(cond):
      def body():
        nonlocal i, j
        i = 1
        j = 2

      def orelse():
        nonlocal i, j
        i = -1
        j = -2

      i, j = None, None
      control_flow.if_stmt(
          cond=cond,
          body=body,
          orelse=orelse,
          get_state=None,
          set_state=None,
          symbol_names=('i', 'j'),
          nouts=2)
      return i, j

    self.assertEqual(test_fn(True), (1, 2))
    self.assertEqual(test_fn(False), (-1, -2))
    self.assertNoOpsCreated()

  def _basic_cond(self, body_fn, else_fn):
    def body():
      nonlocal x
      x = body_fn()

    def orelse():
      nonlocal x
      x = else_fn()

    def set_state(cond_vars):
      nonlocal x
      x, = cond_vars

    x = 0
    control_flow.if_stmt(
        cond=constant_op.constant(True),
        body=body,
        orelse=orelse,
        get_state=lambda: (x,),
        set_state=set_state,
        symbol_names=('x',),
        nouts=1)
    return x

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(
        ValueError, "'x' is None at the end of the main branch"):
      self._basic_cond(lambda: None, lambda: 1)
    with self.assertRaisesRegex(
        ValueError, "'x' is None at the end of the else branch"):
      self._basic_cond(lambda: 1, lambda: None)

  def test_tensor_undefined_output(self):
    with self.assertRaisesRegex(
        ValueError, "'x' must also be initialized in the main branch"):
      self._basic_cond(lambda: variable_operators.Undefined('x'), lambda: 1)
    with self.assertRaisesRegex(
        ValueError, "'x' must also be initialized in the else branch"):
      self._basic_cond(lambda: 1, lambda: variable_operators.Undefined('s'))

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(
        TypeError, "'x' has dtype int32.*but.*float32"):
      self._basic_cond(lambda: 1, lambda: 1.0)

  def _fixed_cond(self, cond_val):
    def body():
      nonlocal x
      x = 1

    def orelse():
      nonlocal x
      x = -1

    def set_state(cond_vars):
      nonlocal x
      x, = cond_vars

    x = 0
    control_flow.if_stmt(
        cond=cond_val,
        body=body,
        orelse=orelse,
        get_state=lambda: (x,),
        set_state=set_state,
        symbol_names=('x',),
        nouts=1)
    return x

  def _assertFixedCondResult(self, cond, expected):
    def test_fn():
      return self._fixed_cond(cond)
    self.assertEqual(test_fn(), expected)

  def test_tensor_legal_cond_scalar(self):
    self._assertFixedCondResult(constant_op.constant(True), 1)
    self._assertFixedCondResult(constant_op.constant(False), -1)

  def test_tensor_legal_cond_single_element_nd(self):
    self._assertFixedCondResult(constant_op.constant([[True]]), 1)
    self._assertFixedCondResult(constant_op.constant([[False]]), -1)
    self._assertFixedCondResult(_unranked_item(True), 1)
    self._assertFixedCondResult(_unranked_item(False), -1)

  def _assertCondCheckFails(self, cond):
    with self.assertRaisesRegex(
        ValueError, 'condition of if statement expected to be `tf.bool`'):
      self._fixed_cond(cond)

  def test_tensor_illegal_cond_not_bool(self):
    self._assertCondCheckFails(constant_op.constant(1))

  def test_tensor_illegal_cond_not_single_element(self):
    self._assertCondCheckFails(constant_op.constant([1, 2, 3]))
    self._assertCondCheckFails(constant_op.constant([True, False]))

  def test_tensor_illegal_cond_not_single_element_dynamic_shape(self):
    self._fixed_cond(_partial_shaped_bools())
    # TODO(mdan): This error is quite bad. Measure the cost of an assertion.
    self.assertRaisesRuntime(
        errors_impl.InvalidArgumentError, 'requested shape has 1')

if __name__ == '__main__':
  test.main()
