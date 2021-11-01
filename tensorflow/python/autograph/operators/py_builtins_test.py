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
"""Tests for py_builtins module."""

import sys

import six

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TestBase(object):

  def overridden_method(self, x):
    return x + 20


@test_util.run_all_in_graph_and_eager_modes
class PyBuiltinsTest(test.TestCase):

  def test_abs(self):
    self.assertEqual(py_builtins.abs_(-1), 1)
    with self.cached_session() as sess:
      t = py_builtins.abs_(constant_op.constant(-1))
      self.assertEqual(self.evaluate(t), 1)
      t = py_builtins.abs_(constant_op.constant([-1, 2, -3]))
      self.assertAllEqual(self.evaluate(t), [1, 2, 3])

  def test_abs_dataset(self):
    dataset = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 1)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 2)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 3)

  def test_abs_dataset_zipped(self):
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([1, -2, 3])
    dataset = dataset_ops.DatasetV2.zip((dataset_1, dataset_2))
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (1, 1))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (2, 2))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (3, 3))

  def test_abs_dataset_mixed(self):
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([-1, 2, 3])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([1, -2, 3])
    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([-1, -2, -3])
    dataset_4 = dataset_ops.DatasetV2.zip((dataset_1, dataset_2))
    dataset = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    dataset = py_builtins.abs_(dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      for i in range(1, 4):
        actual = self.evaluate(iterator.get_next())
        self.assertAllEqual(actual[0], i)
        self.assertAllEqual(actual[1], (i, i))

  def test_float(self):
    self.assertEqual(py_builtins.float_(10), 10.0)
    self.assertEqual(py_builtins.float_('10.0'), 10.0)
    with self.cached_session() as sess:
      t = py_builtins.float_(constant_op.constant(1, dtype=dtypes.int64))
      self.assertEqual(self.evaluate(t), 1.0)
      st = py_builtins.float_(constant_op.constant('1.0'))
      self.assertEqual(self.evaluate(st), 1.0)

  def test_int(self):
    self.assertEqual(py_builtins.int_(10.0), 10)
    self.assertEqual(py_builtins.int_('11', 2), 3)
    with self.cached_session() as sess:
      t = py_builtins.int_(constant_op.constant(1, dtype=dtypes.float64))
      self.assertEqual(self.evaluate(t), 1)
      st = py_builtins.int_(constant_op.constant('1'))
      self.assertEqual(self.evaluate(st), 1)
      st = py_builtins.int_(constant_op.constant('1'), 10)
      self.assertEqual(self.evaluate(st), 1)

  def test_int_unsupported_base(self):
    t = constant_op.constant(1, dtype=dtypes.float64)
    with self.assertRaises(NotImplementedError):
      py_builtins.int_(t, 2)

  def test_len(self):
    self.assertEqual(py_builtins.len_([1, 2, 3]), 3)
    with self.cached_session() as sess:
      t = py_builtins.len_(constant_op.constant([[1], [2], [3]]))
      self.assertEqual(t, 3)
      ta = py_builtins.len_(tensor_array_ops.TensorArray(dtypes.int32, size=5))
      self.assertEqual(self.evaluate(ta), 5)
      tl = py_builtins.len_(data_structures.tf_tensor_list_new([3, 4, 5]))
      self.assertEqual(self.evaluate(tl), 3)

  def test_len_dataset(self):
    dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
    self.assertEqual(self.evaluate(py_builtins.len_(dataset)), 3)

    # graph mode
    @def_function.function(autograph=False)
    def test_fn():
      dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
      return py_builtins.len_(dataset)

    self.assertEqual(self.evaluate(test_fn()), 3)

  def test_len_dataset_infinite(self):
    dataset = dataset_ops.DatasetV2.range(5).repeat().batch(2)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      _ = self.evaluate(py_builtins.len_(dataset))

    # graph mode
    @def_function.function
    def test_fn():
      dataset = dataset_ops.DatasetV2.range(5).repeat().batch(2)
      return py_builtins.len_(dataset)

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(test_fn())

  def test_len_dataset_unknown(self):
    dataset = dataset_ops.DatasetV2.range(5).filter(lambda _: True).batch(2)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      _ = self.evaluate(py_builtins.len_(dataset))

    # graph mode
    @def_function.function(autograph=False)
    def test_fn():
      dataset = dataset_ops.DatasetV2.range(5).filter(lambda _: True).batch(2)
      return py_builtins.len_(dataset)

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(test_fn())

  def test_len_scalar(self):
    with self.assertRaises(ValueError):
      py_builtins.len_(constant_op.constant(1))

  @test_util.run_deprecated_v1
  def test_len_dynamic_shape(self):
    with self.cached_session() as sess:
      p = array_ops.placeholder(dtype=dtypes.int32, shape=None)
      t = py_builtins.len_(p)
      self.assertEqual(sess.run(t, {p: [1, 2, 3]}), 3)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        t = py_builtins.len_(p)
        sess.run(t, {p: 1})

  @test_util.run_deprecated_v1
  def test_print_tensors(self):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      with self.cached_session() as sess:
        sess.run(py_builtins.print_(constant_op.constant('test message'), 1))
        self.assertEqual(out_capturer.getvalue(), 'test message 1\n')
    finally:
      sys.stdout = sys.__stdout__

  @test_util.run_deprecated_v1
  def test_print_complex(self):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      with self.cached_session() as sess:
        sess.run(
            py_builtins.print_(constant_op.constant('test message'), [1, 2]))
        self.assertEqual(out_capturer.getvalue(), 'test message [1, 2]\n')
    finally:
      sys.stdout = sys.__stdout__

  def test_range(self):
    self.assertListEqual(list(py_builtins.range_(3)), [0, 1, 2])
    self.assertListEqual(list(py_builtins.range_(1, 3)), [1, 2])
    self.assertListEqual(list(py_builtins.range_(2, 0, -1)), [2, 1])

  def test_range_tensor(self):
    with self.cached_session() as sess:
      r = py_builtins.range_(constant_op.constant(3))
      self.assertAllEqual(self.evaluate(r), [0, 1, 2])
      r = py_builtins.range_(1, constant_op.constant(3))
      self.assertAllEqual(self.evaluate(r), [1, 2])
      r = py_builtins.range_(2, 0, constant_op.constant(-1))
      self.assertAllEqual(self.evaluate(r), [2, 1])

  def test_range_tensor_empty_range(self):
    with self.session() as sess:
      r = py_builtins.range_(constant_op.constant(-3))
      self.assertAllEqual(self.evaluate(r), [])
      r = py_builtins.range_(5, constant_op.constant(2))
      self.assertAllEqual(self.evaluate(r), [])

  def test_enumerate(self):
    self.assertListEqual(
        list(py_builtins.enumerate_([3, 2, 1])), [(0, 3), (1, 2), (2, 1)])
    self.assertListEqual(
        list(py_builtins.enumerate_([3, 2, 1], 5)), [(5, 3), (6, 2), (7, 1)])
    self.assertListEqual(list(py_builtins.enumerate_([-8], -3)), [(-3, -8)])

  def test_enumerate_dataset(self):
    dataset = dataset_ops.DatasetV2.from_tensor_slices(['a', 'c'])
    start = constant_op.constant(20, dtype=dtypes.int64)
    dataset = py_builtins.enumerate_(dataset, start)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (20, b'a'))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (21, b'c'))

  def test_zip(self):
    self.assertListEqual(
        list(py_builtins.zip_([3, 2, 1], [1, 2, 3])), [(3, 1), (2, 2), (1, 3)])
    self.assertListEqual(
        list(py_builtins.zip_([4, 5, 6], [-1, -2])), [(4, -1), (5, -2)])

  def test_zip_dataset(self):
    ds1 = dataset_ops.DatasetV2.from_tensor_slices([-11, -12, 4])
    ds2 = dataset_ops.DatasetV2.from_tensor_slices([-21, -22, 5])
    ds3 = py_builtins.zip_(ds1, ds2)
    iterator = dataset_ops.make_one_shot_iterator(ds3)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), (-11, -21))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (-12, -22))
      self.assertAllEqual(self.evaluate(iterator.get_next()), (4, 5))

  def test_map(self):

    def increment(x):
      return x + 1

    add_list = lambda x, y: x + y
    self.assertListEqual(
        list(py_builtins.map_(increment, [4, 5, 6])), [5, 6, 7])
    self.assertListEqual(
        list(py_builtins.map_(add_list, [3, 2, 1], [-1, -2, -3])), [2, 0, -2])

  def test_map_dataset(self):

    def increment(x):
      return x + 1

    ds1 = dataset_ops.DatasetV2.from_tensor_slices([4, 5, 6])
    ds2 = py_builtins.map_(increment, ds1)
    iterator = dataset_ops.make_one_shot_iterator(ds2)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 5)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 6)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 7)

  def test_map_multiple_datasets(self):
    add_list = lambda x, y: x + y
    ds1 = dataset_ops.DatasetV2.from_tensor_slices([-11, -12, 4])
    ds2 = dataset_ops.DatasetV2.from_tensor_slices([-21, -22, 5])
    ds3 = py_builtins.map_(add_list, ds1, ds2)
    iterator = dataset_ops.make_one_shot_iterator(ds3)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), -32)
      self.assertAllEqual(self.evaluate(iterator.get_next()), -34)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 9)

  def test_next_normal(self):
    iterator = iter([1, 2, 3])
    self.assertEqual(py_builtins.next_(iterator), 1)
    self.assertEqual(py_builtins.next_(iterator), 2)
    self.assertEqual(py_builtins.next_(iterator), 3)
    with self.assertRaises(StopIteration):
      py_builtins.next_(iterator)
    self.assertEqual(py_builtins.next_(iterator, 4), 4)

  def test_next_tf_iterator(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn(go_out_of_range, with_default):
      iterator = iter(dataset_ops.Dataset.range(3))
      retval = (
          py_builtins.next_(iterator),
          py_builtins.next_(iterator),
          py_builtins.next_(iterator),
      )
      if go_out_of_range:
        if with_default:
          retval += (
              py_builtins.next_(iterator,
                                constant_op.constant(-3, dtype=dtypes.int64)),
              py_builtins.next_(iterator,
                                constant_op.constant(-4, dtype=dtypes.int64)),
          )
        else:
          py_builtins.next_(iterator)
      return retval

    self.assertAllEqual(
        self.evaluate(test_fn(go_out_of_range=False, with_default=None)),
        (0, 1, 2))
    self.assertAllEqual(
        self.evaluate(test_fn(go_out_of_range=True, with_default=True)),
        (0, 1, 2, -3, -4))
    with self.assertRaises(errors_impl.OutOfRangeError):
      self.evaluate(test_fn(go_out_of_range=True, with_default=False))

  def test_next_tf_iterator_error_checking(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn():
      iterator = iter(dataset_ops.Dataset.range(1))
      py_builtins.next_(iterator)
      py_builtins.next_(iterator, constant_op.constant(-3))

    # Dataset.range defaults to int64,
    with self.assertRaisesRegex(TypeError, 'default.*int64'):
      self.evaluate(test_fn())

  def test_next_tf_iterator_error_checking_structures(self):
    # graph-mode iterators are only supported inside tf.function.
    @def_function.function(autograph=False)
    def test_fn(default_val):
      ds = dataset_ops.Dataset.range(1)
      ds = ds.map(lambda i: {'a': i + 1, 'b': i + 10})
      iterator = iter(ds)
      py_builtins.next_(iterator)
      py_builtins.next_(iterator, default_val)

    default = {
        'a': constant_op.constant(3, dtype=dtypes.int64),
    }
    with self.assertRaisesRegex(TypeError, 'same element structure'):
      test_fn(default)
    default = {
        'a': constant_op.constant(3.0),
        'b': [constant_op.constant(30), constant_op.constant(300)]
    }
    with self.assertRaisesRegex(TypeError, 'same element structure'):
      test_fn(default)
    default = {
        'a': constant_op.constant(3.0),
        'b': constant_op.constant(30, dtype=dtypes.int64),
    }
    with self.assertRaisesRegex(TypeError, 'float32'):
      test_fn(default)

  def _basic_function_scope(self):
    return function_wrappers.FunctionScope(
        'test_function_name',
        'test_scope',  # Note: this must match the name in the `with` statement.
        converter.ConversionOptions())

  def test_eval_in_original_context(self):

    def test_fn():
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:
        return py_builtins.eval_in_original_context(eval, ('l',), test_scope)

    self.assertEqual(test_fn(), 1)

  def test_eval_in_original_context_inner_function(self):

    def test_fn():
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          l = 2  # pylint:disable=unused-variable
          return py_builtins.eval_in_original_context(eval, ('l',), test_scope)

        return inner_fn()

    self.assertEqual(test_fn(), 2)

  def test_locals_in_original_context(self):

    def test_fn():
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:
        return py_builtins.locals_in_original_context(test_scope)

    locs = test_fn()

    self.assertEqual(locs['l'], 1)

  def test_locals_in_original_context_inner_function(self):

    def test_fn():
      l = 1  # pylint:disable=unused-variable
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          l = 2  # pylint:disable=unused-variable
          return py_builtins.locals_in_original_context(test_scope)

        return inner_fn()

    locs = test_fn()

    self.assertEqual(locs['l'], 2)

  def test_globals_in_original_context(self):

    def test_fn():
      with self._basic_function_scope() as test_scope:
        return py_builtins.globals_in_original_context(test_scope)

    globs = test_fn()

    self.assertIs(globs['TestBase'], TestBase)

  def test_globals_in_original_context_inner_function(self):

    def test_fn():
      with self._basic_function_scope() as test_scope:

        def inner_fn():
          # Note: a user function without a top-level function scope should
          # never be found in user code; it's only possible in generated code.
          return py_builtins.globals_in_original_context(test_scope)

        return inner_fn()

    globs = test_fn()

    self.assertIs(globs['TestBase'], TestBase)

  def test_super_in_original_context_unary_call(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self):
        with test_case_self._basic_function_scope() as test_scope:
          test_base_unbound = py_builtins.super_in_original_context(
              super, (TestSubclass,), test_scope)
          test_base = test_base_unbound.__get__(self, TestSubclass)
          return test_base.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_binary_call(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self):
        with test_case_self._basic_function_scope() as test_scope:
          test_base = py_builtins.super_in_original_context(
              super, (TestSubclass, self), test_scope)
          return test_base.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_niladic_call(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self):
        with test_case_self._basic_function_scope() as test_scope:
          b = py_builtins.super_in_original_context(super, (), test_scope)
          return b.overridden_method(1)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(), 21)

  def test_super_in_original_context_caller_with_locals(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        y = 7
        with test_case_self._basic_function_scope() as test_scope:
          z = 7
          return py_builtins.super_in_original_context(
              super, (), test_scope).overridden_method(x + y - z)

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_function(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          def inner_fn():
            return py_builtins.super_in_original_context(
                super, (), test_scope).overridden_method(x)

          return inner_fn()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_super_in_original_context_inner_lambda(self):
    test_case_self = self

    class TestSubclass(TestBase):

      def overridden_method(self, x):
        test_case_self.fail('This should never be called.')

      def test_method(self, x):
        with test_case_self._basic_function_scope() as test_scope:
          # Oddly, it's sufficient to use `self` in an inner function
          # to gain access to __class__ in this scope.
          # TODO(mdan): Is this true across implementations?
          # Note: normally, it's illegal to use super() in inner functions (it
          # throws an error), but the generated code may create them.
          l = lambda: py_builtins.super_in_original_context(  # pylint:disable=g-long-lambda
              super, (), test_scope).overridden_method(x)
          return l()

    tc = TestSubclass()
    self.assertEqual(tc.test_method(1), 21)

  def test_filter(self):
    self.assertListEqual(
        list(py_builtins.filter_(lambda x: x == 'b', ['a', 'b', 'c'])), ['b'])
    self.assertListEqual(
        list(py_builtins.filter_(lambda x: x < 3, [3, 2, 1])), [2, 1])

  def test_filter_dataset(self):
    dataset = dataset_ops.DatasetV2.from_tensor_slices([3, 2, 1])
    dataset = py_builtins.filter_(lambda x: x < 3, dataset)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(iterator.get_next()), 2)
      self.assertAllEqual(self.evaluate(iterator.get_next()), 1)

  def test_any(self):
    self.assertEqual(py_builtins.any_([False, True, False]), True)
    self.assertEqual(py_builtins.any_([False, False, False]), False)

  def test_any_dataset(self):
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([False, False, False])
    self.assertEqual(self.evaluate(py_builtins.any_(dataset_1)), True)
    self.assertEqual(self.evaluate(py_builtins.any_(dataset_2)), False)

    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2])
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_3)

    dataset_4 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_zipped = dataset_ops.DatasetV2.zip((dataset_4, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_zipped)

    dataset_mixed = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.any_(dataset_mixed)

  def test_all(self):
    self.assertEqual(py_builtins.all_([False, True, False]), False)
    self.assertEqual(py_builtins.all_([True, True, True]), True)

  def test_all_dataset(self):
    dataset_1 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_2 = dataset_ops.DatasetV2.from_tensor_slices([True, True, True])
    self.assertEqual(self.evaluate(py_builtins.all_(dataset_1)), False)
    self.assertEqual(self.evaluate(py_builtins.all_(dataset_2)), True)

    dataset_3 = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2])
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_3)

    dataset_4 = dataset_ops.DatasetV2.from_tensor_slices([False, True, False])
    dataset_zipped = dataset_ops.DatasetV2.zip((dataset_4, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_zipped)

    dataset_mixed = dataset_ops.DatasetV2.zip((dataset_3, dataset_4))
    with self.assertRaises(ValueError):
      py_builtins.all_(dataset_mixed)

  def test_sorted(self):
    self.assertListEqual(py_builtins.sorted_([2, 3, 1]), [1, 2, 3])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], key=lambda x: -x), [3, 2, 1])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], reverse=True), [3, 2, 1])
    self.assertListEqual(
        py_builtins.sorted_([2, 3, 1], key=lambda x: -x, reverse=True),
        [1, 2, 3])
    self.assertAllEqual(
        py_builtins.sorted_([[4, 3], [2, 1]], key=lambda x: sum(x)),
        [[2, 1], [4, 3]])

  def test_sorted_tensor(self):
    iterable_1 = constant_op.constant([2, 3, 1])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1))), [1, 2, 3])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1, key=lambda x: -x))),
        [3, 2, 1])
    self.assertListEqual(
        list(self.evaluate(py_builtins.sorted_(iterable_1, reverse=True))),
        [3, 2, 1])
    self.assertListEqual(
        list(
            self.evaluate(
                py_builtins.sorted_(iterable_1, key=lambda x: -x,
                                    reverse=True))), [1, 2, 3])

    iterable_2 = constant_op.constant([[4, 3], [2, 1]])
    with self.assertRaises(ValueError):
      py_builtins.sorted_(iterable_2)
    with self.assertRaises(ValueError):
      py_builtins.sorted_(iterable_2, key=lambda x: -x)
    self.assertAllEqual(
        list(
            self.evaluate(
                py_builtins.sorted_(
                    iterable_2, key=lambda x: math_ops.reduce_sum(x)))),
        [[2, 1], [4, 3]])


if __name__ == '__main__':
  test.main()
