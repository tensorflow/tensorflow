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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test


class AssertProperIterableTest(test.TestCase):

  def test_single_tensor_raises(self):
    tensor = constant_op.constant(1)
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(tensor)

  def test_single_sparse_tensor_raises(self):
    ten = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(ten)

  def test_single_ndarray_raises(self):
    array = np.array([1, 2, 3])
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(array)

  def test_single_string_raises(self):
    mystr = "hello"
    with self.assertRaisesRegexp(TypeError, "proper"):
      check_ops.assert_proper_iterable(mystr)

  def test_non_iterable_object_raises(self):
    non_iterable = 1234
    with self.assertRaisesRegexp(TypeError, "to be iterable"):
      check_ops.assert_proper_iterable(non_iterable)

  def test_list_does_not_raise(self):
    list_of_stuff = [
        constant_op.constant([11, 22]), constant_op.constant([1, 2])
    ]
    check_ops.assert_proper_iterable(list_of_stuff)

  def test_generator_does_not_raise(self):
    generator_of_stuff = (constant_op.constant([11, 22]), constant_op.constant(
        [1, 2]))
    check_ops.assert_proper_iterable(generator_of_stuff)


class AssertEqualTest(test.TestCase):

  def test_doesnt_raise_when_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      with ops.control_dependencies([check_ops.assert_equal(small, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_greater(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 4], name="big")
      with ops.control_dependencies(
          [check_ops.assert_equal(
              big, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*big.*small"):
        out.eval()

  def test_raises_when_less(self):
    with self.test_session():
      small = constant_op.constant([3, 1], name="small")
      big = constant_op.constant([4, 2], name="big")
      with ops.control_dependencies([check_ops.assert_equal(small, big)]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("small.*big"):
        out.eval()

  def test_doesnt_raise_when_equal_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      small_2 = constant_op.constant([1, 2], name="small_2")
      with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_equal_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="small")
      small_2 = constant_op.constant([1, 1], name="small_2")
      with self.assertRaisesRegexp(ValueError, "must be"):
        with ops.control_dependencies([check_ops.assert_equal(small, small_2)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies([check_ops.assert_equal(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertNoneEqualTest(test.TestCase):

  def test_doesnt_raise_when_not_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([10, 20], name="small")
      with ops.control_dependencies(
          [check_ops.assert_none_equal(big, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_equal(self):
    with self.test_session():
      small = constant_op.constant([3, 1], name="small")
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, small)]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("x != y did not hold"):
        out.eval()

  def test_doesnt_raise_when_not_equal_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3], name="big")
      with ops.control_dependencies(
          [check_ops.assert_none_equal(small, big)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_not_equal_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="small")
      big = constant_op.constant([10, 10], name="big")
      with self.assertRaisesRegexp(ValueError, "must be"):
        with ops.control_dependencies(
            [check_ops.assert_none_equal(small, big)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies(
          [check_ops.assert_none_equal(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertLessTest(test.TestCase):

  def test_raises_when_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      with ops.control_dependencies(
          [check_ops.assert_less(
              small, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*small.*small"):
        out.eval()

  def test_raises_when_greater(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 4], name="big")
      with ops.control_dependencies([check_ops.assert_less(big, small)]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("big.*small"):
        out.eval()

  def test_doesnt_raise_when_less(self):
    with self.test_session():
      small = constant_op.constant([3, 1], name="small")
      big = constant_op.constant([4, 2], name="big")
      with ops.control_dependencies([check_ops.assert_less(small, big)]):
        out = array_ops.identity(small)
      out.eval()

  def test_doesnt_raise_when_less_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1], name="small")
      big = constant_op.constant([3, 2], name="big")
      with ops.control_dependencies([check_ops.assert_less(small, big)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_less_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="small")
      big = constant_op.constant([3, 2], name="big")
      with self.assertRaisesRegexp(ValueError, "must be"):
        with ops.control_dependencies([check_ops.assert_less(small, big)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies([check_ops.assert_less(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertLessEqualTest(test.TestCase):

  def test_doesnt_raise_when_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      with ops.control_dependencies(
          [check_ops.assert_less_equal(small, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_greater(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 4], name="big")
      with ops.control_dependencies(
          [check_ops.assert_less_equal(
              big, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*big.*small"):
        out.eval()

  def test_doesnt_raise_when_less_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 2], name="big")
      with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
        out = array_ops.identity(small)
      out.eval()

  def test_doesnt_raise_when_less_equal_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1], name="small")
      big = constant_op.constant([3, 1], name="big")
      with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="small")
      big = constant_op.constant([3, 1], name="big")
      with self.assertRaisesRegexp(ValueError, "must be"):
        with ops.control_dependencies(
            [check_ops.assert_less_equal(small, big)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies(
          [check_ops.assert_less_equal(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertGreaterTest(test.TestCase):

  def test_raises_when_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      with ops.control_dependencies(
          [check_ops.assert_greater(
              small, small, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*small.*small"):
        out.eval()

  def test_raises_when_less(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 4], name="big")
      with ops.control_dependencies([check_ops.assert_greater(small, big)]):
        out = array_ops.identity(big)
      with self.assertRaisesOpError("small.*big"):
        out.eval()

  def test_doesnt_raise_when_greater(self):
    with self.test_session():
      small = constant_op.constant([3, 1], name="small")
      big = constant_op.constant([4, 2], name="big")
      with ops.control_dependencies([check_ops.assert_greater(big, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_doesnt_raise_when_greater_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1], name="small")
      big = constant_op.constant([3, 2], name="big")
      with ops.control_dependencies([check_ops.assert_greater(big, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_greater_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="small")
      big = constant_op.constant([3, 2], name="big")
      with self.assertRaisesRegexp(ValueError, "must be"):
        with ops.control_dependencies([check_ops.assert_greater(big, small)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies([check_ops.assert_greater(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertGreaterEqualTest(test.TestCase):

  def test_doesnt_raise_when_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(small, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_less(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 4], name="big")
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(
              small, big, message="fail")]):
        out = array_ops.identity(small)
      with self.assertRaisesOpError("fail.*small.*big"):
        out.eval()

  def test_doesnt_raise_when_greater_equal(self):
    with self.test_session():
      small = constant_op.constant([1, 2], name="small")
      big = constant_op.constant([3, 2], name="big")
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(big, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_doesnt_raise_when_greater_equal_and_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1], name="small")
      big = constant_op.constant([3, 1], name="big")
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(big, small)]):
        out = array_ops.identity(small)
      out.eval()

  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = constant_op.constant([1, 1, 1], name="big")
      big = constant_op.constant([3, 1], name="small")
      with self.assertRaisesRegexp(ValueError, "Dimensions must be equal"):
        with ops.control_dependencies(
            [check_ops.assert_greater_equal(big, small)]):
          out = array_ops.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = constant_op.constant([])
      curly = constant_op.constant([])
      with ops.control_dependencies(
          [check_ops.assert_greater_equal(larry, curly)]):
        out = array_ops.identity(larry)
      out.eval()


class AssertNegativeTest(test.TestCase):

  def test_doesnt_raise_when_negative(self):
    with self.test_session():
      frank = constant_op.constant([-1, -2], name="frank")
      with ops.control_dependencies([check_ops.assert_negative(frank)]):
        out = array_ops.identity(frank)
      out.eval()

  def test_raises_when_positive(self):
    with self.test_session():
      doug = constant_op.constant([1, 2], name="doug")
      with ops.control_dependencies(
          [check_ops.assert_negative(
              doug, message="fail")]):
        out = array_ops.identity(doug)
      with self.assertRaisesOpError("fail.*doug"):
        out.eval()

  def test_raises_when_zero(self):
    with self.test_session():
      claire = constant_op.constant([0], name="claire")
      with ops.control_dependencies([check_ops.assert_negative(claire)]):
        out = array_ops.identity(claire)
      with self.assertRaisesOpError("claire"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is negative when it satisfies:
    #   For every element x_i in x, x_i < 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = constant_op.constant([], name="empty")
      with ops.control_dependencies([check_ops.assert_negative(empty)]):
        out = array_ops.identity(empty)
      out.eval()


class AssertPositiveTest(test.TestCase):

  def test_raises_when_negative(self):
    with self.test_session():
      freddie = constant_op.constant([-1, -2], name="freddie")
      with ops.control_dependencies(
          [check_ops.assert_positive(
              freddie, message="fail")]):
        out = array_ops.identity(freddie)
      with self.assertRaisesOpError("fail.*freddie"):
        out.eval()

  def test_doesnt_raise_when_positive(self):
    with self.test_session():
      remmy = constant_op.constant([1, 2], name="remmy")
      with ops.control_dependencies([check_ops.assert_positive(remmy)]):
        out = array_ops.identity(remmy)
      out.eval()

  def test_raises_when_zero(self):
    with self.test_session():
      meechum = constant_op.constant([0], name="meechum")
      with ops.control_dependencies([check_ops.assert_positive(meechum)]):
        out = array_ops.identity(meechum)
      with self.assertRaisesOpError("meechum"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is positive when it satisfies:
    #   For every element x_i in x, x_i > 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = constant_op.constant([], name="empty")
      with ops.control_dependencies([check_ops.assert_positive(empty)]):
        out = array_ops.identity(empty)
      out.eval()


class AssertRankTest(test.TestCase):

  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant(1, name="my_tensor")
      desired_rank = 1
      with self.assertRaisesRegexp(ValueError,
                                   "fail.*my_tensor.*must have rank 1"):
        with ops.control_dependencies(
            [check_ops.assert_rank(
                tensor, desired_rank, message="fail")]):
          array_ops.identity(tensor).eval()

  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant(1, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval()

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_one_tensor_raises_if_rank_too_large_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 0
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, desired_rank)]):
          array_ops.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_large_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval()

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 2
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, desired_rank)]):
          array_ops.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_raises_if_rank_is_not_scalar_static(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      with self.assertRaisesRegexp(ValueError, "Rank must be a scalar"):
        check_ops.assert_rank(tensor, np.array([], dtype=np.int32))

  def test_raises_if_rank_is_not_scalar_dynamic(self):
    with self.test_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.int32, name="rank_tensor")
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: [1, 2]})

  def test_raises_if_rank_is_not_integer_static(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        check_ops.assert_rank(tensor, .5)

  def test_raises_if_rank_is_not_integer_dynamic(self):
    with self.test_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankInTest(test.TestCase):

  def test_rank_zero_tensor_raises_if_rank_mismatch_static_rank(self):
    with self.test_session():
      tensor_rank0 = constant_op.constant(42, name="my_tensor")
      with self.assertRaisesRegexp(
          ValueError, "fail.*my_tensor.*must have rank.*in.*1.*2"):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
          array_ops.identity(tensor_rank0).eval()

  def test_rank_zero_tensor_raises_if_rank_mismatch_dynamic_rank(self):
    with self.test_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank0, (1, 2), message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    with self.test_session():
      tensor_rank0 = constant_op.constant(42, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
          array_ops.identity(tensor_rank0).eval()

  def test_rank_zero_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    with self.test_session():
      tensor_rank0 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank0, desired_ranks)]):
          array_ops.identity(tensor_rank0).eval(feed_dict={tensor_rank0: 42.0})

  def test_rank_one_tensor_doesnt_raise_if_rank_matches_static_rank(self):
    with self.test_session():
      tensor_rank1 = constant_op.constant([42, 43], name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
          array_ops.identity(tensor_rank1).eval()

  def test_rank_one_tensor_doesnt_raise_if_rank_matches_dynamic_rank(self):
    with self.test_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      for desired_ranks in ((0, 1, 2), (1, 0, 2), (1, 2, 0)):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank1, desired_ranks)]):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  def test_rank_one_tensor_raises_if_rank_mismatches_static_rank(self):
    with self.test_session():
      tensor_rank1 = constant_op.constant((42, 43), name="my_tensor")
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with ops.control_dependencies([
            check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
          array_ops.identity(tensor_rank1).eval()

  def test_rank_one_tensor_raises_if_rank_mismatches_dynamic_rank(self):
    with self.test_session():
      tensor_rank1 = array_ops.placeholder(dtypes.float32, name="my_tensor")
      with ops.control_dependencies([
          check_ops.assert_rank_in(tensor_rank1, (0, 2))]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor_rank1).eval(feed_dict={
              tensor_rank1: (42.0, 43.0)
          })

  def test_raises_if_rank_is_not_scalar_static(self):
    with self.test_session():
      tensor = constant_op.constant((42, 43), name="my_tensor")
      desired_ranks = (
          np.array(1, dtype=np.int32),
          np.array((2, 1), dtype=np.int32))
      with self.assertRaisesRegexp(ValueError, "Rank must be a scalar"):
        check_ops.assert_rank_in(tensor, desired_ranks)

  def test_raises_if_rank_is_not_scalar_dynamic(self):
    with self.test_session():
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

  def test_raises_if_rank_is_not_integer_static(self):
    with self.test_session():
      tensor = constant_op.constant((42, 43), name="my_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        check_ops.assert_rank_in(tensor, (1, .5,))

  def test_raises_if_rank_is_not_integer_dynamic(self):
    with self.test_session():
      tensor = constant_op.constant(
          (42, 43), dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegexp(TypeError,
                                   "must be of type <dtype: 'int32'>"):
        with ops.control_dependencies(
            [check_ops.assert_rank_in(tensor, (1, rank_tensor))]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})


class AssertRankAtLeastTest(test.TestCase):

  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant(1, name="my_tensor")
      desired_rank = 1
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank at least 1"):
        with ops.control_dependencies(
            [check_ops.assert_rank_at_least(tensor, desired_rank)]):
          array_ops.identity(tensor).eval()

  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant(1, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval()

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_one_ten_doesnt_raise_raise_if_rank_too_large_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval()

  def test_rank_one_ten_doesnt_raise_if_rank_too_large_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval()

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = constant_op.constant([1, 2], name="my_tensor")
      desired_rank = 2
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with ops.control_dependencies(
            [check_ops.assert_rank_at_least(tensor, desired_rank)]):
          array_ops.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank_at_least(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})


class AssertNonNegativeTest(test.TestCase):

  def test_raises_when_negative(self):
    with self.test_session():
      zoe = constant_op.constant([-1, -2], name="zoe")
      with ops.control_dependencies([check_ops.assert_non_negative(zoe)]):
        out = array_ops.identity(zoe)
      with self.assertRaisesOpError("zoe"):
        out.eval()

  def test_doesnt_raise_when_zero_and_positive(self):
    with self.test_session():
      lucas = constant_op.constant([0, 2], name="lucas")
      with ops.control_dependencies([check_ops.assert_non_negative(lucas)]):
        out = array_ops.identity(lucas)
      out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-negative when it satisfies:
    #   For every element x_i in x, x_i >= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = constant_op.constant([], name="empty")
      with ops.control_dependencies([check_ops.assert_non_negative(empty)]):
        out = array_ops.identity(empty)
      out.eval()


class AssertNonPositiveTest(test.TestCase):

  def test_doesnt_raise_when_zero_and_negative(self):
    with self.test_session():
      tom = constant_op.constant([0, -2], name="tom")
      with ops.control_dependencies([check_ops.assert_non_positive(tom)]):
        out = array_ops.identity(tom)
      out.eval()

  def test_raises_when_positive(self):
    with self.test_session():
      rachel = constant_op.constant([0, 2], name="rachel")
      with ops.control_dependencies([check_ops.assert_non_positive(rachel)]):
        out = array_ops.identity(rachel)
      with self.assertRaisesOpError("rachel"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-positive when it satisfies:
    #   For every element x_i in x, x_i <= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = constant_op.constant([], name="empty")
      with ops.control_dependencies([check_ops.assert_non_positive(empty)]):
        out = array_ops.identity(empty)
      out.eval()


class AssertIntegerTest(test.TestCase):

  def test_doesnt_raise_when_integer(self):
    with self.test_session():
      integers = constant_op.constant([1, 2], name="integers")
      with ops.control_dependencies([check_ops.assert_integer(integers)]):
        out = array_ops.identity(integers)
      out.eval()

  def test_raises_when_float(self):
    with self.test_session():
      floats = constant_op.constant([1.0, 2.0], name="floats")
      with self.assertRaisesRegexp(TypeError, "Expected.*integer"):
        check_ops.assert_integer(floats)


class IsStrictlyIncreasingTest(test.TestCase):

  def test_constant_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(check_ops.is_strictly_increasing([1, 1, 1]).eval())

  def test_decreasing_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(check_ops.is_strictly_increasing([1, 0, -1]).eval())

  def test_2d_decreasing_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(
          check_ops.is_strictly_increasing([[1, 3], [2, 4]]).eval())

  def test_increasing_tensor_is_increasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_strictly_increasing([1, 2, 3]).eval())

  def test_increasing_rank_two_tensor(self):
    with self.test_session():
      self.assertTrue(
          check_ops.is_strictly_increasing([[-1, 2], [3, 4]]).eval())

  def test_tensor_with_one_element_is_strictly_increasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_strictly_increasing([1]).eval())

  def test_empty_tensor_is_strictly_increasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_strictly_increasing([]).eval())


class IsNonDecreasingTest(test.TestCase):

  def test_constant_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_non_decreasing([1, 1, 1]).eval())

  def test_decreasing_tensor_is_not_non_decreasing(self):
    with self.test_session():
      self.assertFalse(check_ops.is_non_decreasing([3, 2, 1]).eval())

  def test_2d_decreasing_tensor_is_not_non_decreasing(self):
    with self.test_session():
      self.assertFalse(check_ops.is_non_decreasing([[1, 3], [2, 4]]).eval())

  def test_increasing_rank_one_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_non_decreasing([1, 2, 3]).eval())

  def test_increasing_rank_two_tensor(self):
    with self.test_session():
      self.assertTrue(check_ops.is_non_decreasing([[-1, 2], [3, 3]]).eval())

  def test_tensor_with_one_element_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_non_decreasing([1]).eval())

  def test_empty_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(check_ops.is_non_decreasing([]).eval())


if __name__ == "__main__":
  test.main()
