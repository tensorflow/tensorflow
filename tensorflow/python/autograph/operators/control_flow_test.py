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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

import six

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ForLoopTest(test.TestCase):

  def test_tensor(self):
    with ops.Graph().as_default():
      s = control_flow.for_stmt(
          constant_op.constant([1, 2, 3, 4]),
          extra_test=lambda s: True,
          body=lambda i, s: (s * 10 + i,),
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=(0,))
      self.assertEqual(self.evaluate(s), (1234,))

  def test_tensor_with_extra_test_only_python_state(self):
    class MutableObject(object):
      field_1 = constant_op.constant(0, dtype=dtypes.int32)
      field_2 = constant_op.constant(1, dtype=dtypes.int32)
    state = MutableObject()

    def get_state():
      return (state.field_1, state.field_2)

    def set_state(new_state):
      state.field_1, state.field_2 = new_state

    def body(i):
      state.field_1 += i
      state.field_2 *= i
      return ()

    control_flow.for_stmt(
        iter_=constant_op.constant([1, 2, 3, 4]),
        body=body,
        extra_test=lambda: state.field_1 < 6,
        get_state=get_state,
        set_state=set_state,
        init_vars=())
    self.assertEqual(self.evaluate(state.field_1), 6)
    self.assertEqual(self.evaluate(state.field_2), 6)

  def test_python(self):
    s = control_flow.for_stmt(
        range(5),
        extra_test=lambda s: True,
        body=lambda i, s: (s * 10 + i,),
        get_state=None,
        set_state=None,
        init_vars=(0,))
    self.assertEqual(s, (1234,))

  def test_tf_dataset(self):
    with ops.Graph().as_default():
      s = control_flow.for_stmt(
          dataset_ops.Dataset.range(5),
          extra_test=None,
          body=lambda i, s: (s * 10 + i,),
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=(constant_op.constant(0, dtype=dtypes.int64),))
      self.assertEqual(self.evaluate(s), (1234,))

  def test_dataset_with_extra_test(self):
    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda s: s < 3,
        body=lambda i, s: (s + i,),
        get_state=lambda: (),
        set_state=lambda _: None,
        init_vars=(constant_op.constant(0, dtype=dtypes.int64),))
    self.assertEqual(self.evaluate(s), (3,))

  def test_dataset_with_extra_test_and_state(self):
    state = [constant_op.constant(0, dtype=dtypes.int64)]

    def get_state():
      return (state[0],)

    def set_state(new_state):
      state[0], = new_state

    def body(i, s):
      state[0] += i
      return (s + i,)

    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda s: s < 3,
        body=body,
        get_state=get_state,
        set_state=set_state,
        init_vars=(constant_op.constant(0, dtype=dtypes.int64),))
    self.assertEqual(self.evaluate(s), (3,))
    self.assertEqual(self.evaluate(state[0]), (3,))

  def test_dataset_with_extra_test_no_extra_iterations(self):

    def guarded_body(i, s):
      with ops.control_dependencies((control_flow_ops.Assert(i < 3, (i,)),)):
        return s + i,

    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=lambda s: s < 3,
        body=guarded_body,
        get_state=lambda: (),
        set_state=lambda _: None,
        init_vars=(constant_op.constant(0, dtype=dtypes.int64),))
    self.assertEqual(self.evaluate(s), (3,))

  @test_util.run_v2_only
  def test_tf_dataset_no_loop_vars(self):
    v = variables.Variable(0, dtype=dtypes.int64)
    self.evaluate(v.initializer)

    def stateless_with_side_effects(i):
      v.assign(v.read_value() * 10 + i)

    # function is important here, because ops test for its presence.
    @def_function.function(autograph=False)
    def test_fn():
      control_flow.for_stmt(
          dataset_ops.Dataset.range(5),
          extra_test=None,
          body=stateless_with_side_effects,
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=())

    test_fn()
    self.assertEqual(self.evaluate(v.read_value()), 1234)

  def test_tf_iterator(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn():
      itr = iter(dataset_ops.Dataset.range(5))
      return control_flow.for_stmt(
          itr,
          extra_test=None,
          body=lambda i, s: (s * 10 + i,),
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=(constant_op.constant(0, dtype=dtypes.int64),))
    s, = test_fn()
    self.assertAllEqual(s, 1234)

  @test_util.run_v2_only
  def test_tf_iterator_no_loop_vars(self):
    v = variables.Variable(0, dtype=dtypes.int64)

    def stateless_with_side_effects(i):
      v.assign(v.read_value() * 10 + i)

    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn():
      control_flow.for_stmt(
          iter(dataset_ops.Dataset.range(5)),
          extra_test=None,
          body=stateless_with_side_effects,
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=())

    test_fn()
    self.assertEqual(self.evaluate(v.read_value()), 1234)


class WhileLoopTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_tensor(self):
    n = constant_op.constant(5)
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        get_state=lambda: (),
        set_state=lambda _: None,
        init_vars=(0, 0))
    self.assertEqual((5, 10), self.evaluate(results))

  def test_tensor_with_tf_side_effects_in_cond(self):

    n = constant_op.constant(5, dtype=dtypes.int64)
    v = variables.Variable(0, dtype=dtypes.int64)

    def get_and_increment(v):
      v.assign(v.read_value() + 1)
      return v.read_value()

    # function is important here, because ops test for its presence.
    @def_function.function(autograph=False)
    def test_fn():
      return control_flow.while_stmt(
          test=lambda i: get_and_increment(v) < n,
          body=lambda i: (i + 1,),
          get_state=lambda: (),
          set_state=lambda _: None,
          init_vars=(0,))

    results = test_fn()

    self.evaluate(v.initializer)
    self.assertEqual(self.evaluate(results), (4,))
    self.assertEqual(self.evaluate(v), (5,))

  def test_tensor_with_python_state(self):
    n = constant_op.constant(5)

    class MutableObject(object):
      field = constant_op.constant(0, dtype=dtypes.int32)
    state = MutableObject()

    def get_state():
      return (state.field,)

    def set_state(new_state):
      state.field, = new_state

    def body(i, s):
      state.field += i
      return (i + 1, s + i)

    s = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=body,
        get_state=get_state,
        set_state=set_state,
        init_vars=(0, 0))
    self.assertEqual(self.evaluate(s), (5, 10))
    self.assertEqual(self.evaluate(state.field), 10)

  @test_util.run_deprecated_v1
  def test_python_with_tensor_state(self):
    n = 5
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        get_state=lambda: (),
        set_state=lambda _: None,
        init_vars=(0, constant_op.constant(0)))
    result_i, result_s = results
    self.assertEqual(5, result_i)
    self.assertEqual(10, self.evaluate(result_s))

  def test_python(self):
    n = 5
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        get_state=None,
        set_state=None,
        init_vars=(0, 0))
    self.assertEqual((5, 10), results)

  def test_python_infinite_loop(self):
    if __debug__:
      with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 100):
        with self.assertRaisesRegexp(ValueError, 'iteration limit'):
          control_flow.while_stmt(
              test=lambda _: True,
              body=lambda i: (i + 1,),
              get_state=None,
              set_state=None,
              init_vars=(0,))

  def test_python_long_loop_unroll_warning(self):
    if __debug__:
      with test.mock.patch.object(
          control_flow, 'INEFFICIENT_UNROLL_MIN_ITERATIONS', 10):
        with ops.Graph().as_default():
          out_capturer = six.StringIO()
          with test.mock.patch.object(sys, 'stdout', out_capturer):
            ag_logging.echo_log_to_stdout = True
            sys.stdout = out_capturer
            control_flow.while_stmt(
                test=lambda i, _: i < 100,
                body=lambda i, _: (i + 1, gen_math_ops.add(i, 1),),
                get_state=None,
                set_state=None,
                init_vars=(0, None))
          self.assertTrue(re.match(
              r'.*ops.*loop.*large.*iterations.*Add.*',
              out_capturer.getvalue()))


class IfStmtTest(test.TestCase):

  def single_return_if_stmt(self, cond):
    return control_flow.if_stmt(
        cond=cond,
        body=lambda: 1,
        orelse=lambda: -1,
        get_state=lambda: (),
        set_state=lambda _: None)

  def multi_return_if_stmt(self, cond):
    return control_flow.if_stmt(
        cond=cond,
        body=lambda: (1, 2),
        orelse=lambda: (-1, -2),
        get_state=lambda: (),
        set_state=lambda _: None)

  @test_util.run_deprecated_v1
  def test_tensor(self):
    with self.cached_session():
      t = self.single_return_if_stmt(constant_op.constant(True))
      self.assertEqual(1, self.evaluate(t))
      t = self.single_return_if_stmt(constant_op.constant(False))
      self.assertEqual(-1, self.evaluate(t))

  def test_python(self):
    self.assertEqual(1, self.single_return_if_stmt(True))
    self.assertEqual(-1, self.single_return_if_stmt(False))

  @test_util.run_deprecated_v1
  def test_tensor_multiple_returns(self):
    with self.cached_session():
      t = self.multi_return_if_stmt(constant_op.constant(True))
      self.assertAllEqual([1, 2], self.evaluate(t))
      t = self.multi_return_if_stmt(constant_op.constant(False))
      self.assertAllEqual([-1, -2], self.evaluate(t))

  def test_python_multiple_returns(self):
    self.assertEqual((1, 2), self.multi_return_if_stmt(True))
    self.assertEqual((-1, -2), self.multi_return_if_stmt(False))


if __name__ == '__main__':
  test.main()
