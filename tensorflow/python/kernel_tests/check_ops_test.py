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
"""Tests for tensorflow.ops.check_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class AssertV2Asserts(test.TestCase):

  def test_passes_when_it_should(self):
    # This is a v2 test and need to run eagerly
    with context.eager_mode():
      c1 = constant_op.constant(-1, name="minus_one", dtype=dtypes.int32)
      c2 = constant_op.constant(2, name="two", dtype=dtypes.int32)
      c3 = constant_op.constant([3., 3.], name="three", dtype=dtypes.float32)
      c4 = constant_op.constant([3., 3.5], name="three_and_a_half",
                                dtype=dtypes.float32)
      scalar = c1
      non_scalar = c3
      integer = c1
      non_integer = c3
      positive = c2
      negative = c1
      cases = [
          (check_ops.assert_equal_v2, (c1, c1), (c1, c2)),
          (check_ops.assert_less_v2, (c1, c2), (c1, c1)),
          (check_ops.assert_near_v2, (c3, c3), (c3, c4)),
          (check_ops.assert_greater_v2, (c2, c1), (c1, c1)),
          (check_ops.assert_negative_v2, (negative,), (positive,)),
          (check_ops.assert_positive_v2, (positive,), (negative,)),
          (check_ops.assert_less_equal_v2, (c1, c1), (c2, c1)),
          (check_ops.assert_none_equal_v2, (c1, c2), (c3, c4)),
          (check_ops.assert_non_negative_v2, (positive,), (negative,)),
          (check_ops.assert_non_positive_v2, (negative,), (positive,)),
          (check_ops.assert_greater_equal_v2, (c1, c1), (c1, c2)),
          (check_ops.assert_type_v2, (c1, dtypes.int32), (c1, dtypes.float32),
           TypeError),
          (check_ops.assert_integer_v2, (integer,), (non_integer,),
           TypeError),
          (check_ops.assert_scalar_v2, (scalar,), (non_scalar,),
           ValueError),
          (check_ops.assert_rank_v2, (c1, 0), (c3, 2), ValueError),
          (check_ops.assert_rank_in_v2, (c1, [0, 1]), (c1, [1, 2]),
           ValueError),
          (check_ops.assert_rank_at_least_v2, (non_scalar, 1), (scalar, 1),
           ValueError),
      ]

      for case in cases:
        fn = case[0]
        passing_args = case[1]
        failing_args = case[2]
        error = errors.InvalidArgumentError if len(case) < 4 else case[3]

        print("Testing %s passing properly." % fn)

        fn(*passing_args)

        print("Testing %s failing properly." % fn)

        @def_function.function
        def failing_fn():
          fn(*failing_args, message="fail")  # pylint: disable=cell-var-from-loop

        with self.assertRaisesRegexp(error, "fail"):
          failing_fn()

        del failing_fn


class AssertProperIterableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_single_tensor_raises(self):
    tensor = constant_op.constant(1)
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(tensor)

  @test_util.run_in_graph_and_eager_modes
  def test_single_sparse_tensor_raises(self):
    ten = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(ten)

  @test_util.run_in_graph_and_eager_modes
  def test_single_ndarray_raises(self):
    array = np.array([1, 2, 3])
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(array)

  @test_util.run_in_graph_and_eager_modes
  def test_single_string_raises(self):
    mystr = "hello"
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(mystr)

  @test_util.run_in_graph_and_eager_modes
  def test_non_iterable_object_raises(self):
    non_iterable = 1234
    with self.assertRaisesRegexp(TypeError, "to be iterable"):
      check_ops.assert_proper_iterable(non_iterable)

  @test_util.run_in_graph_and_eager_modes
  def test_list_does_not_raise(self):
    list_of_stuff = [
        constant_op.constant([11, 22]), constant_op.constant([1, 2])
    ]
    check_ops.assert_proper_iterable(list_of_stuff)

  @test_util.run_in_graph_and_eager_modes
  def test_generator_does_not_raise(self):
    generator_of_stuff = (constant_op.constant([11, 22]), constant_op.constant(
        [1, 2]))
    check_ops.assert_proper_iterable(generator_of_stuff)


class AssertEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies([check_ops.assert_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_scalar_comparison(self):
    const_true = constant_op.constant(True, name="true")
    const_false = constant_op.constant(False, name="false")
    with self.assertRaisesRegexp(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(const_true, const_false, message="fail")

  def test_returns_none_with_eager(self):
    with context.eager_mode():
      small = constant_op.constant([1, 2], name="small")
      x = check_ops.assert_equal(small, small)
      assert x is None

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    # Static check
    static_small = constant_op.constant([1, 2], name="small")
    static_big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesRegexp(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(static_big, static_small, message="fail")

  @test_util.run_deprecated_v1
  def test_raises_when_greater_dynamic(self):
    with self.cached_session():
      small = array_ops.placeholder(dtypes.int32, name="small")
      big = array_ops.placeholder(dtypes.int32, name="big")
      with ops.control_dependencies(
          [check_ops.assert_equal(big, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*big.*small"):
        out.eval(feed_dict={small: [1, 2], big: [3, 4]})

  def test_error_message_eager(self):
    expected_error_msg_full = r"""big does not equal small
Condition x == y did not hold.
Indices of first 3 different values:
\[\[0 0\]
 \[1 1\]
 \[2 0\]\]
Corresponding x values:
\[2 3 6\]
Corresponding y values:
\[20 30 60\]
First 6 elements of x:
\[2 2 3 3 6 6\]
First 6 elements of y:
\[20  2  3 30 60  6\]
"""
    expected_error_msg_default = r"""big does not equal small
Condition x == y did not hold.
Indices of first 3 different values:
\[\[0 0\]
 \[1 1\]
 \[2 0\]\]
Corresponding x values:
\[2 3 6\]
Corresponding y values:
\[20 30 60\]
First 3 elements of x:
\[2 2 3\]
First 3 elements of y:
\[20  2  3\]
"""
    expected_error_msg_short = r"""big does not equal small
Condition x == y did not hold.
Indices of first 2 different values:
\[\[0 0\]
 \[1 1\]\]
Corresponding x values:
\[2 3\]
Corresponding y values:
\[20 30\]
First 2 elements of x:
\[2 2\]
First 2 elements of y:
\[20  2\]
"""
    with context.eager_mode():
      big = constant_op.constant([[2, 2], [3, 3], [6, 6]])
      small = constant_op.constant([[20, 2], [3, 30], [60, 6]])
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_full):
        check_ops.assert_equal(big, small, message="big does not equal small",
                               summarize=10)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_default):
        check_ops.assert_equal(big, small, message="big does not equal small")
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_short):
        check_ops.assert_equal(big, small, message="big does not equal small",
                               summarize=2)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    # Static check
    static_small = constant_op.constant([3, 1], name="small")
    static_big = constant_op.constant([4, 2], name="big")
    with self.assertRaisesRegexp(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(static_big, static_small, message="fail")

  @test_util.run_deprecated_v1
  def test_raises_when_less_dynamic(self):
    with self.cached_session():
      small = array_ops.placeholder(dtypes.int32, name="small")
      big = array_ops.placeholder(dtypes.int32, name="big")
      with ops.control_dependencies([check_ops.assert_equal(small, big)]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("small.*big"):
        out.eval(feed_dict={small: [3, 1], big: [4, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal_and_broadcastable_shapes(self):
    small = constant_op.constant([[1, 2], [1, 2]], name="small")
    small_2 = constant_op.constant([1, 2], name="small_2")
    with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_equal_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="small")
    small_2 = constant_op.constant([1, 1], name="small_2")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[3\] vs. \[2\]|"
         r"Dimensions must be equal, but are 3 and 2")):
      with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_not_equal_and_broadcastable_shapes(self):
    cond = constant_op.constant([True, False], name="small")
    with self.assertRaisesRegexp(errors.InvalidArgumentError, "fail"):
      check_ops.assert_equal(cond, False, message="fail")

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)


class AssertNoneEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_not_equal(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([10, 20], name="small")
    with ops.control_dependencies(
        [check_ops.assert_none_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    small = constant_op.constant([3, 1], name="small")
    with self.assertRaisesOpError("x != y did not hold"):
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_not_equal_and_broadcastable_shapes(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3], name="big")
    with ops.control_dependencies(
        [check_ops.assert_none_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_not_equal_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([10, 10], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (ValueError, errors.InvalidArgumentError),
        (r"Incompatible shapes: \[3\] vs. \[2\]|"
         r"Dimensions must be equal, but are 3 and 2")):
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_none_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_returns_none_with_eager(self):
    with context.eager_mode():
      t1 = constant_op.constant([1, 2])
      t2 = constant_op.constant([3, 4])
      x = check_ops.assert_none_equal(t1, t2)
      assert x is None

  def test_error_message_eager(self):
    # Note that the following three strings are regexes
    expected_error_msg_full = r"""0.0, 1.0, 2.0, 3.0, 4.0, 5.0"""
    expected_error_msg_default = r"""0.0, 1.0, 2.0, \.\.\."""
    expected_error_msg_short = r"""0.0, 1.0, \.\.\."""
    with context.eager_mode():
      t = constant_op.constant(
          np.array(range(6)), shape=[2, 3], dtype=np.float32)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_full):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=10)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_full):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=-1)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_default):
        check_ops.assert_none_equal(t, t, message="This is the error message.")
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg_short):
        check_ops.assert_none_equal(
            t, t, message="This is the error message.", summarize=2)


class AssertAllCloseTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1., name="y")
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_32_bit_due_to_default_rtol(self):
    eps = np.finfo(np.float32).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1. + 2 * eps, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_32_bit_due_to_default_atol(self):
    eps = np.finfo(np.float32).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(0., name="x")
    y = constant_op.constant(0. + 2 * eps, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, rtol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_64_bit_due_to_default_rtol(self):
    eps = np.finfo(np.float64).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(1., name="x", dtype=np.float64)
    y = constant_op.constant(1. + 2 * eps, name="y", dtype=np.float64)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_64_bit_due_to_default_atol(self):
    eps = np.finfo(np.float64).eps
    # Default rtol/atol is 10*eps
    x = constant_op.constant(0., name="x", dtype=np.float64)
    y = constant_op.constant(0. + 2 * eps, name="y", dtype=np.float64)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, rtol=0., message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_due_to_custom_rtol(self):
    x = constant_op.constant(1., name="x")
    y = constant_op.constant(1.1, name="y")
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0., rtol=0.5,
                               message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_close_enough_due_to_custom_atol(self):
    x = constant_op.constant(0., name="x")
    y = constant_op.constant(0.1, name="y", dtype=np.float32)
    with ops.control_dependencies(
        [check_ops.assert_near(x, y, atol=0.5, rtol=0.,
                               message="failure message")]):
      out = array_ops.identity(x)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_near(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_atol_violated(self):
    x = constant_op.constant(10., name="x")
    y = constant_op.constant(10.2, name="y")
    with self.assertRaisesOpError("x and y not equal to tolerance"):
      with ops.control_dependencies(
          [check_ops.assert_near(x, y, atol=0.1,
                                 message="failure message")]):
        out = array_ops.identity(x)
        self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_default_rtol_violated(self):
    x = constant_op.constant(0.1, name="x")
    y = constant_op.constant(0.0, name="y")
    with self.assertRaisesOpError("x and y not equal to tolerance"):
      with ops.control_dependencies(
          [check_ops.assert_near(x, y, message="failure message")]):
        out = array_ops.identity(x)
        self.evaluate(out)

  def test_returns_none_with_eager(self):
    with context.eager_mode():
      t1 = constant_op.constant([1., 2.])
      t2 = constant_op.constant([1., 2.])
      x = check_ops.assert_near(t1, t2)
      assert x is None


class AssertLessTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with self.assertRaisesOpError("failure message.*\n*.* x < y did not hold"):
      with ops.control_dependencies(
          [check_ops.assert_less(
              small, small, message="failure message")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError("x < y did not hold"):
      with ops.control_dependencies([check_ops.assert_less(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less(self):
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([4, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([3, 2], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (ValueError, errors.InvalidArgumentError),
        (r"Incompatible shapes: \[3\] vs. \[2\]|"
         "Dimensions must be equal, but are 3 and 2")):
      with ops.control_dependencies([check_ops.assert_less(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_less(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_returns_none_with_eager(self):
    with context.eager_mode():
      t1 = constant_op.constant([1, 2])
      t2 = constant_op.constant([3, 4])
      x = check_ops.assert_less(t1, t2)
      assert x is None


class AssertLessEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies(
        [check_ops.assert_less_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(
              big, small, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 1], name="big")
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([1, 1, 1], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_less_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)


class AssertGreaterTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_greater(
              small, small, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError("x > y did not hold"):
      with ops.control_dependencies([check_ops.assert_greater(small, big)]):
        out = array_ops.identity(big)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater(self):
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([4, 2], name="big")
    with ops.control_dependencies([check_ops.assert_greater(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies([check_ops.assert_greater(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_greater_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([3, 2], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies([check_ops.assert_greater(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies([check_ops.assert_greater(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)


class AssertGreaterEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_less(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(
              small, big, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_equal(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 2], name="big")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_greater_equal_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 1], name="big")
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(big, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="big")
    big = constant_op.constant([3, 1], name="small")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegexp(
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)


class AssertNegativeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_negative(self):
    frank = constant_op.constant([-1, -2], name="frank")
    with ops.control_dependencies([check_ops.assert_negative(frank)]):
      out = array_ops.identity(frank)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_positive(self):
    doug = constant_op.constant([1, 2], name="doug")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_negative(
              doug, message="fail")]):
        out = array_ops.identity(doug)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_zero(self):
    claire = constant_op.constant([0], name="claire")
    with self.assertRaisesOpError("x < 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_negative(claire)]):
        out = array_ops.identity(claire)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # A tensor is negative when it satisfies:
    #   For every element x_i in x, x_i < 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_negative(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)


class AssertPositiveTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_negative(self):
    freddie = constant_op.constant([-1, -2], name="freddie")
    with self.assertRaisesOpError("fail"):
      with ops.control_dependencies(
          [check_ops.assert_positive(
              freddie, message="fail")]):
        out = array_ops.identity(freddie)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_positive(self):
    remmy = constant_op.constant([1, 2], name="remmy")
    with ops.control_dependencies([check_ops.assert_positive(remmy)]):
      out = array_ops.identity(remmy)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_zero(self):
    meechum = constant_op.constant([0], name="meechum")
    with self.assertRaisesOpError("x > 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_positive(meechum)]):
        out = array_ops.identity(meechum)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # A tensor is positive when it satisfies:
    #   For every element x_i in x, x_i > 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_positive(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)


class EnsureShapeTest(test.TestCase):

  # Static shape inference
  @test_util.run_deprecated_v1
  def testStaticShape(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    ensure_shape_op = check_ops.ensure_shape(placeholder, (3, 3, 3))
    self.assertEqual(ensure_shape_op.get_shape(), (3, 3, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_MergesShapes(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    ensure_shape_op = check_ops.ensure_shape(placeholder, (5, 4, None))
    self.assertEqual(ensure_shape_op.get_shape(), (5, 4, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenRankIncompatible(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenDimIncompatible(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 2, 4))

  @test_util.run_deprecated_v1
  def testStaticShape_CanSetUnknownShape(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    ensure_shape_op = check_ops.ensure_shape(derived, None)
    self.assertEqual(ensure_shape_op.get_shape(), None)

  # Dynamic shape check
  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape_RaisesError(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = math_ops.divide(placeholder, 3, name="MyDivide")
    derived = check_ops.ensure_shape(derived, (3, 3, 3))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"Shape of tensor MyDivide \[2,1\] is not compatible with "
          r"expected shape \[3,3,3\]."):
        sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape_RaisesErrorDimUnknown(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (None, None, 3))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"Shape of tensor [A-Za-z_]* \[2,1\] is not compatible with "
          r"expected shape \[\?,\?,3\]."):
        sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (2, 1))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testEnsuresDynamicShape_WithUnknownDims(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    derived = check_ops.ensure_shape(derived, (None, None))
    feed_val = [[1], [2]]
    with self.cached_session() as sess:
      sess.run(derived, feed_dict={placeholder: feed_val})

  @test_util.run_deprecated_v1
  def testGradient(self):
    placeholder = array_ops.placeholder(dtypes.float32)
    derived = check_ops.ensure_shape(placeholder, (None, None))
    gradient = gradients.gradients(derived, placeholder)

    feed_val = [[4.0], [-1.0]]
    with self.cached_session() as sess:
      gradient_values, = sess.run(gradient, feed_dict={placeholder: feed_val})

    expected = [[1.0], [1.0]]
    self.assertAllEqual(gradient_values, expected)


class EnsureShapeBenchmark(test.Benchmark):

  def _grappler_all_off_config(self):
    config = config_pb2.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.disable_model_pruning = 1
    config.graph_options.rewrite_options.constant_folding = off
    config.graph_options.rewrite_options.layout_optimizer = off
    config.graph_options.rewrite_options.arithmetic_optimization = off
    config.graph_options.rewrite_options.dependency_optimization = off
    return config

  def _run(self, op, feed_dict=None, num_iters=5000, name=None, **kwargs):
    config = self._grappler_all_off_config()
    with session.Session(config=config) as sess:
      deltas = []
      # Warm up the session
      for _ in range(5):
        sess.run(op, feed_dict=feed_dict)
      for _ in range(num_iters):
        start = time.time()
        sess.run(op, feed_dict=feed_dict)
        end = time.time()
        deltas.append(end - start)
      mean_time = np.median(deltas)
      mean_us = mean_time * 1e6
      # mean_us = (end - start) * 1e6 / num_iters
      self.report_benchmark(
          name=name,
          wall_time=mean_us,
          extras=kwargs,
      )

  def benchmark_const_op(self):
    # In this case, we expect that the overhead of a `session.run` call
    # far outweighs the time taken to execute the op...
    shape = (3, 3, 100)
    input_op = random_ops.random_normal(shape)
    self._run(array_ops.identity(input_op), name="SingleConstOp")

  def benchmark_single_ensure_op(self):
    # In this case, we expect that the overhead of a `session.run` call
    # far outweighs the time taken to execute the op...
    shape = (3, 3, 100)
    input_op = random_ops.random_normal(shape)
    ensure_shape_op = check_ops.ensure_shape(input_op, shape)
    self._run(ensure_shape_op, name="SingleEnsureShapeOp")

  def _apply_n_times(self, op, target, n=1000):
    for _ in range(n):
      target = op(target)
    return target

  def benchmark_n_ops(self):
    shape = (1000,)
    input_op = random_ops.random_normal(shape)
    n_ops = self._apply_n_times(array_ops.identity, input_op)
    self._run(n_ops, name="NIdentityOps_1000")

  def benchmark_n_ensure_ops(self):
    shape = (1000,)
    input_op = random_ops.random_normal(shape)
    n_ensure_ops = self._apply_n_times(
        lambda x: check_ops.ensure_shape(array_ops.identity(x), shape),
        input_op)
    self._run(n_ensure_ops, name="NEnsureShapeAndIdentityOps_1000")


class AssertRankTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 1
    with self.assertRaisesRegexp(ValueError,
                                 "fail.*must have rank 1"):
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_large_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 0
    with self.assertRaisesRegexp(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_large_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 1
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 2
    with self.assertRaisesRegexp(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_scalar_static(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegexp(ValueError, "Rank must be a scalar"):
      check_ops.assert_rank(tensor, np.array([], dtype=np.int32))

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_scalar_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.int32, name="rank_tensor")
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_integer_static(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegexp(TypeError,
                                 "must be of type <dtype: 'int32'>"):
      check_ops.assert_rank(tensor, .5)

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_integer_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankInTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_mismatch_static_rank(self):
    tensor_rank0 = constant_op.constant(42, name="my_tensor")
    with self.assertRaisesRegexp(
        ValueError, "fail.*must have rank.*in.*1.*2"):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
        self.evaluate(array_ops.identity(tensor_rank0))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_mismatch_dynamic_rank(self):
    with self.cached_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    tensor_rank0 = constant_op.constant(42, name="my_tensor")
    for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
        self.evaluate(array_ops.identity(tensor_rank0))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    with self.cached_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    tensor_rank1 = constant_op.constant([42, 43], name="my_tensor")
    for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
        self.evaluate(array_ops.identity(tensor_rank1))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    with self.cached_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_mismatches_static_rank(self):
    tensor_rank1 = constant_op.constant((42, 43), name="my_tensor")
    with self.assertRaisesRegexp(ValueError, "rank"):
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
        self.evaluate(array_ops.identity(tensor_rank1))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_mismatches_dynamic_rank(self):
    with self.cached_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_scalar_static(self):
    tensor = constant_op.constant((42, 43), name="my_tensor")
    desired_ranks = (
        np.array(1, dtype=np.int32),
        np.array((2, 1), dtype=np.int32))
    with self.assertRaisesRegexp(ValueError, "Rank must be a scalar"):
      check_ops.assert_rank_in(tensor, desired_ranks)

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_scalar_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          (42, 43), dtype=dtypes.float32, name="my_tensor")
      desired_ranks = (
          array_ops.placeholder(dtypes.int32, name="rank0_tensor"),
          array_ops.placeholder(dtypes.int32, name="rank1_tensor"))
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            (check_ops.assert_rank_in(tensor, desired_ranks),)):
          array_ops.identity(tensor).eval(feed_dict={
              desired_ranks[0]: 1,
              desired_ranks[1]: [2, 1],
          })

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_integer_static(self):
    tensor = constant_op.constant((42, 43), name="my_tensor")
    with self.assertRaisesRegexp(TypeError,
                                 "must be of type <dtype: 'int32'>"):
      check_ops.assert_rank_in(tensor, (1, .5,))

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_integer_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          (42, 43), dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        with ops.control_dependencies(
            [check_ops.assert_rank_in(tensor, (1, rank_tensor))]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankAtLeastTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 1
    with self.assertRaisesRegexp(ValueError, "rank at least 1"):
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_ten_doesnt_raise_raise_if_rank_too_large_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_ten_doesnt_raise_if_rank_too_large_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 1
    with ops.control_dependencies(
        [check_ops.assert_rank_at_least(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 2
    with self.assertRaisesRegexp(ValueError, "rank at least 2"):
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})


class AssertNonNegativeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_negative(self):
    zoe = constant_op.constant([-1, -2], name="zoe")
    with self.assertRaisesOpError("x >= 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_non_negative(zoe)]):
        out = array_ops.identity(zoe)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_zero_and_positive(self):
    lucas = constant_op.constant([0, 2], name="lucas")
    with ops.control_dependencies([check_ops.assert_non_negative(lucas)]):
      out = array_ops.identity(lucas)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-negative when it satisfies:
    #   For every element x_i in x, x_i >= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_non_negative(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)


class AssertNonPositiveTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_zero_and_negative(self):
    tom = constant_op.constant([0, -2], name="tom")
    with ops.control_dependencies([check_ops.assert_non_positive(tom)]):
      out = array_ops.identity(tom)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_positive(self):
    rachel = constant_op.constant([0, 2], name="rachel")
    with self.assertRaisesOpError("x <= 0 did not hold"):
      with ops.control_dependencies([check_ops.assert_non_positive(rachel)]):
        out = array_ops.identity(rachel)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-positive when it satisfies:
    #   For every element x_i in x, x_i <= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    empty = constant_op.constant([], name="empty")
    with ops.control_dependencies([check_ops.assert_non_positive(empty)]):
      out = array_ops.identity(empty)
    self.evaluate(out)


class AssertIntegerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_integer(self):
    integers = constant_op.constant([1, 2], name="integers")
    with ops.control_dependencies([check_ops.assert_integer(integers)]):
      out = array_ops.identity(integers)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_float(self):
    floats = constant_op.constant([1.0, 2.0], name="floats")
    with self.assertRaisesRegexp(TypeError, "Expected.*integer"):
      check_ops.assert_integer(floats)


class AssertTypeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_correct_type(self):
    integers = constant_op.constant([1, 2], dtype=dtypes.int64)
    with ops.control_dependencies([
        check_ops.assert_type(integers, dtypes.int64)]):
      out = array_ops.identity(integers)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_wrong_type(self):
    floats = constant_op.constant([1.0, 2.0], dtype=dtypes.float16)
    with self.assertRaisesRegexp(TypeError, "must be of type.*float32"):
      check_ops.assert_type(floats, dtypes.float32)


class IsStrictlyIncreasingTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_constant_tensor_is_not_strictly_increasing(self):
    self.assertFalse(self.evaluate(check_ops.is_strictly_increasing([1, 1, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_decreasing_tensor_is_not_strictly_increasing(self):
    self.assertFalse(self.evaluate(
        check_ops.is_strictly_increasing([1, 0, -1])))

  @test_util.run_in_graph_and_eager_modes
  def test_2d_decreasing_tensor_is_not_strictly_increasing(self):
    self.assertFalse(
        self.evaluate(check_ops.is_strictly_increasing([[1, 3], [2, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_tensor_is_increasing(self):
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([1, 2, 3])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_two_tensor(self):
    self.assertTrue(
        self.evaluate(check_ops.is_strictly_increasing([[-1, 2], [3, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_tensor_with_one_element_is_strictly_increasing(self):
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([1])))

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_is_strictly_increasing(self):
    self.assertTrue(self.evaluate(check_ops.is_strictly_increasing([])))


class IsNonDecreasingTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_constant_tensor_is_non_decreasing(self):
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1, 1, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_decreasing_tensor_is_not_non_decreasing(self):
    self.assertFalse(self.evaluate(check_ops.is_non_decreasing([3, 2, 1])))

  @test_util.run_in_graph_and_eager_modes
  def test_2d_decreasing_tensor_is_not_non_decreasing(self):
    self.assertFalse(self.evaluate(
        check_ops.is_non_decreasing([[1, 3], [2, 4]])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_one_tensor_is_non_decreasing(self):
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1, 2, 3])))

  @test_util.run_in_graph_and_eager_modes
  def test_increasing_rank_two_tensor(self):
    self.assertTrue(self.evaluate(
        check_ops.is_non_decreasing([[-1, 2], [3, 3]])))

  @test_util.run_in_graph_and_eager_modes
  def test_tensor_with_one_element_is_non_decreasing(self):
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([1])))

  @test_util.run_in_graph_and_eager_modes
  def test_empty_tensor_is_non_decreasing(self):
    self.assertTrue(self.evaluate(check_ops.is_non_decreasing([])))


class FloatDTypeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_assert_same_float_dtype(self):
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(None, None))
    self.assertIs(dtypes.float32, check_ops.assert_same_float_dtype([], None))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([], dtypes.float32))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(None, dtypes.float32))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([None, None], None))
    self.assertIs(
        dtypes.float32,
        check_ops.assert_same_float_dtype([None, None], dtypes.float32))

    const_float = constant_op.constant(3.0, dtype=dtypes.float32)
    self.assertIs(
        dtypes.float32,
        check_ops.assert_same_float_dtype([const_float], dtypes.float32))
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_float], dtypes.int32)

    sparse_float = sparse_tensor.SparseTensor(
        constant_op.constant([[111], [232]], dtypes.int64),
        constant_op.constant([23.4, -43.2], dtypes.float32),
        constant_op.constant([500], dtypes.int64))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype([sparse_float],
                                                    dtypes.float32))
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float], dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_float, None, sparse_float], dtypes.float64)

    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(
                      [const_float, sparse_float]))
    self.assertIs(dtypes.float32,
                  check_ops.assert_same_float_dtype(
                      [const_float, sparse_float], dtypes.float32))

    const_int = constant_op.constant(3, dtype=dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int])
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int], dtypes.int32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [sparse_float, const_int], dtypes.float32)
    self.assertRaises(ValueError, check_ops.assert_same_float_dtype,
                      [const_int])


class AssertScalarTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_assert_scalar(self):
    check_ops.assert_scalar(constant_op.constant(3))
    check_ops.assert_scalar(constant_op.constant("foo"))
    check_ops.assert_scalar(3)
    check_ops.assert_scalar("foo")
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      check_ops.assert_scalar(constant_op.constant([3, 4]))


if __name__ == "__main__":
  test.main()
