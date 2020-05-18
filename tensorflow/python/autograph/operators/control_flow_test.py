# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

import numpy as np
import six

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import variables as variable_operators
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ForLoopTest(test.TestCase):

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
    self.assertEqual(self.evaluate(s), (1234,))

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
    self.assertEqual(self.evaluate(s), (1234,))

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
    self.assertEqual(self.evaluate(s), (-171207,))

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
    self.assertEqual(self.evaluate(s), (171207,))

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
    self.assertEqual(self.evaluate(s), (1234,))

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
    self.assertEqual(self.evaluate(s), (171207,))

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
    self.assertEqual(self.evaluate((state.field_1, state.field_2)), (6, 6))

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
    self.assertEqual(self.evaluate(s), (1234,))

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
    self.assertEqual(self.evaluate(s), (12,))

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
    self.assertEqual(self.evaluate((l[0], s)), (3, 3))

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
    self.assertEqual(self.evaluate(i), (3,))

  def test_tf_dataset_no_loop_vars(self):
    def body(i):
      v.assign(v.read_value() * 10 + i)

    v = variables.Variable(0, dtype=dtypes.int64)
    self.evaluate(v.initializer)

    # tf.function required for the automatic control dependencies, and because
    # ops test for its presence.
    @def_function.function
    def test_fn():
      control_flow.for_stmt(
          dataset_ops.Dataset.range(5),
          extra_test=None,
          body=body,
          get_state=lambda: (),
          set_state=lambda _: None,
          symbol_names=(),
          opts={})

    self.evaluate(test_fn())
    self.assertEqual(self.evaluate(v.read_value()), 1234)

  def test_tf_iterator(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function
    def test_fn():
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
      return s
    self.assertAllEqual(test_fn(), 1234)

  def test_tf_iterator_shape_invariants(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function
    def test_fn():
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
      return s
    self.assertAllEqual(test_fn(), [0, 1, 2, 3, 4])

  def test_tf_iterator_no_loop_vars(self):
    def body(i):
      v.assign(v.read_value() * 10 + i)

    v = variables.Variable(0, dtype=dtypes.int64)
    self.evaluate(v.initializer)

    # tf.function required for the automatic control dependencies.
    @def_function.function
    def test_fn():
      control_flow.for_stmt(
          iter(dataset_ops.Dataset.range(5)),
          extra_test=None,
          body=body,
          get_state=lambda: (),
          set_state=lambda _: None,
          symbol_names=(),
          opts={})

    self.evaluate(test_fn())
    self.assertEqual(self.evaluate(v.read_value()), 1234)

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
    self.assertEqual(self.evaluate(s), (123,))

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
    self.assertEqual(self.evaluate(s), (12,))

  def test_tf_ragged_tensor_no_loop_vars(self):
    v = variables.Variable(0, dtype=dtypes.int32)
    self.evaluate(v.initializer)

    def body(i):
      v.assign(v.read_value() * 10 + i[0])

    # tf.function required for the automatic control dependencies.
    @def_function.function(autograph=False)
    def test_fn():
      control_flow.for_stmt(
          ragged_factory_ops.constant([[1], [2, 4], [3]]),
          extra_test=None,
          body=body,
          get_state=lambda: (),
          set_state=lambda _: None,
          symbol_names=(),
          opts={})

    self.evaluate(test_fn())
    # Note: 123 = ((0*10 + 1)*10+2)*10+3 (first element of each row).
    self.assertEqual(self.evaluate(v.read_value()), 123)

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
    with self.assertRaisesRegex(ValueError, '"s" may not be None'):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, '"s" must be defined'):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(ValueError, '"s" is None at the end'):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(TypeError, '"s".* dtype float32 after'):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    with self.assertRaisesRegex(ValueError, r'"s".* shape \(1,\) after'):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))


@test_util.run_all_in_graph_and_eager_modes
class WhileLoopTest(test.TestCase):

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
    self.assertEqual(self.evaluate((i, s)), (5, 1234))

  def test_tensor_with_side_effecting_condition(self):
    v = variables.Variable(0)

    # tf.function required for the automatic control dependencies.
    @def_function.function
    def test_fn():
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
      return i

    self.evaluate(v.initializer)
    self.assertEqual(self.evaluate(test_fn()), (5,))
    self.assertEqual(self.evaluate(v), (12345,))

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
    self.assertEqual(self.evaluate((i, state.field)), (5, 1234))

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
    self.assertEqual(self.evaluate(s), 1234)

  def test_python_while_infinite(self):
    if not __debug__:
      self.skipTest('Feature disabled in optimized mode.')
    with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
      with self.assertRaisesRegexp(ValueError, 'iteration limit'):
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
      with self.assertRaisesRegexp(ValueError, 'iteration limit'):
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
    with self.assertRaisesRegex(ValueError, '"s" may not be None'):
      self._basic_loop(None, lambda i, s: s)
    with self.assertRaisesRegex(ValueError, '"s" must be defined'):
      self._basic_loop(variable_operators.Undefined(''), lambda i, s: s)

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(ValueError, '"s" is None at the end'):
      self._basic_loop(0, lambda i, s: None)

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(TypeError, '"s".* dtype float32 after'):
      self._basic_loop(0, lambda i, s: 1.0)

  def test_tensor_shape_change(self):
    with self.assertRaisesRegex(ValueError, r'"s".* shape \(1,\) after'):
      self._basic_loop(0, lambda i, s: np.array([1], dtype=np.int32))


@test_util.run_all_in_graph_and_eager_modes
class IfStmtTest(test.TestCase):

  def test_tensor(self):

    def test_fn(cond):
      return control_flow.if_stmt(
          cond=cond,
          body=lambda: constant_op.constant(1),
          orelse=lambda: constant_op.constant(-1),
          get_state=lambda: (),
          set_state=lambda _: None,
          basic_symbol_names=('_',),
          composite_symbol_names=())

    self.assertEqual(1, self.evaluate(test_fn(constant_op.constant(True))))
    self.assertEqual(-1, self.evaluate(test_fn(constant_op.constant(False))))

  def test_tensor_multiple_returns(self):

    def test_fn(cond):
      return control_flow.if_stmt(
          cond=cond,
          body=lambda: (constant_op.constant(1), constant_op.constant(2)),
          orelse=lambda: (constant_op.constant(-1), constant_op.constant(-2)),
          get_state=lambda: (),
          set_state=lambda _: None,
          basic_symbol_names=('_',),
          composite_symbol_names=())

    self.assertEqual((1, 2), self.evaluate(test_fn(constant_op.constant(True))))
    self.assertEqual((-1, -2),
                     self.evaluate(test_fn(constant_op.constant(False))))

  def test_python(self):

    def test_fn(cond):
      return control_flow.if_stmt(
          cond=cond,
          body=lambda: 1,
          orelse=lambda: -1,
          get_state=lambda: (),
          set_state=lambda _: None,
          basic_symbol_names=('_',),
          composite_symbol_names=())

    self.assertEqual(1, test_fn(True))
    self.assertEqual(-1, test_fn(False))

  def test_python_multiple_returns(self):

    def test_fn(cond):
      return control_flow.if_stmt(
          cond=cond,
          body=lambda: (1, 2),
          orelse=lambda: (-1, -2),
          get_state=lambda: (),
          set_state=lambda _: None,
          basic_symbol_names=('_',),
          composite_symbol_names=())

    self.assertEqual((1, 2), test_fn(True))
    self.assertEqual((-1, -2), test_fn(False))

  def _basic_cond(self, true_value, false_value):
    # Eager cond had different semantics, we don't test those here.
    with func_graph.FuncGraph('tmp').as_default():
      return control_flow.if_stmt(
          cond=constant_op.constant(True),
          body=true_value,
          orelse=false_value,
          get_state=lambda: (),
          set_state=lambda _: None,
          basic_symbol_names=('s',),
          composite_symbol_names=())

  def test_tensor_none_output(self):
    with self.assertRaisesRegex(
        ValueError, '"s" is None at the end of the TRUE branch'):
      self._basic_cond(lambda: None, lambda: 1)
    with self.assertRaisesRegex(
        ValueError, '"s" is None at the end of the FALSE branch'):
      self._basic_cond(lambda: 1, lambda: None)

  def test_tensor_undefined_output(self):
    with self.assertRaisesRegex(
        ValueError, "must also be initialized in the if.*'s'"):
      self._basic_cond(lambda: variable_operators.Undefined('s'), lambda: 1)
    with self.assertRaisesRegex(
        ValueError, "must also be initialized in the else.*'s'"):
      self._basic_cond(lambda: 1, lambda: variable_operators.Undefined('s'))

  def test_tensor_dtype_change(self):
    with self.assertRaisesRegex(TypeError, '"s" has dtype int32.*but.*float32'):
      self._basic_cond(lambda: 1, lambda: 1.0)


if __name__ == '__main__':
  test.main()
