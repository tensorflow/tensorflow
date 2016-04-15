# Copyright 2016 Google Inc. All Rights Reserved.
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

import tensorflow as tf


class AssertLessTest(tf.test.TestCase):

  def test_raises_when_equal(self):
    with self.test_session():
      small = tf.constant([1, 2], name="small")
      with tf.control_dependencies([tf.assert_less(small, small)]):
        out = tf.identity(small)
      with self.assertRaisesOpError("small.*small"):
        out.eval()

  def test_raises_when_greater(self):
    with self.test_session():
      small = tf.constant([1, 2], name="small")
      big = tf.constant([3, 4], name="big")
      with tf.control_dependencies([tf.assert_less(big, small)]):
        out = tf.identity(small)
      with self.assertRaisesOpError("big.*small"):
        out.eval()

  def test_doesnt_raise_when_less(self):
    with self.test_session():
      small = tf.constant([3, 1], name="small")
      big = tf.constant([4, 2], name="big")
      with tf.control_dependencies([tf.assert_less(small, big)]):
        out = tf.identity(small)
      out.eval()

  def test_doesnt_raise_when_less_and_broadcastable_shapes(self):
    with self.test_session():
      small = tf.constant([1], name="small")
      big = tf.constant([3, 2], name="big")
      with tf.control_dependencies([tf.assert_less(small, big)]):
        out = tf.identity(small)
      out.eval()

  def test_raises_when_less_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = tf.constant([1, 1, 1], name="small")
      big = tf.constant([3, 2], name="big")
      with self.assertRaisesRegexp(ValueError, "broadcast"):
        with tf.control_dependencies([tf.assert_less(small, big)]):
          out = tf.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = tf.constant([])
      curly = tf.constant([])
      with tf.control_dependencies([tf.assert_less(larry, curly)]):
        out = tf.identity(larry)
      out.eval()


class AssertLessEqualTest(tf.test.TestCase):

  def test_doesnt_raise_when_equal(self):
    with self.test_session():
      small = tf.constant([1, 2], name="small")
      with tf.control_dependencies([tf.assert_less_equal(small, small)]):
        out = tf.identity(small)
      out.eval()

  def test_raises_when_greater(self):
    with self.test_session():
      small = tf.constant([1, 2], name="small")
      big = tf.constant([3, 4], name="big")
      with tf.control_dependencies([tf.assert_less_equal(big, small)]):
        out = tf.identity(small)
      with self.assertRaisesOpError("big.*small"):
        out.eval()

  def test_doesnt_raise_when_less_equal(self):
    with self.test_session():
      small = tf.constant([1, 2], name="small")
      big = tf.constant([3, 2], name="big")
      with tf.control_dependencies([tf.assert_less_equal(small, big)]):
        out = tf.identity(small)
      out.eval()

  def test_doesnt_raise_when_less_equal_and_broadcastable_shapes(self):
    with self.test_session():
      small = tf.constant([1], name="small")
      big = tf.constant([3, 1], name="big")
      with tf.control_dependencies([tf.assert_less_equal(small, big)]):
        out = tf.identity(small)
      out.eval()

  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    with self.test_session():
      small = tf.constant([1, 1, 1], name="small")
      big = tf.constant([3, 1], name="big")
      with self.assertRaisesRegexp(ValueError, "broadcast"):
        with tf.control_dependencies([tf.assert_less_equal(small, big)]):
          out = tf.identity(small)
        out.eval()

  def test_doesnt_raise_when_both_empty(self):
    with self.test_session():
      larry = tf.constant([])
      curly = tf.constant([])
      with tf.control_dependencies([tf.assert_less_equal(larry, curly)]):
        out = tf.identity(larry)
      out.eval()


class AssertNegativeTest(tf.test.TestCase):

  def test_doesnt_raise_when_negative(self):
    with self.test_session():
      frank = tf.constant([-1, -2], name="frank")
      with tf.control_dependencies([tf.assert_negative(frank)]):
        out = tf.identity(frank)
      out.eval()

  def test_raises_when_positive(self):
    with self.test_session():
      doug = tf.constant([1, 2], name="doug")
      with tf.control_dependencies([tf.assert_negative(doug)]):
        out = tf.identity(doug)
      with self.assertRaisesOpError("doug"):
        out.eval()

  def test_raises_when_zero(self):
    with self.test_session():
      claire = tf.constant([0], name="claire")
      with tf.control_dependencies([tf.assert_negative(claire)]):
        out = tf.identity(claire)
      with self.assertRaisesOpError("claire"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is negative when it satisfies:
    #   For every element x_i in x, x_i < 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = tf.constant([], name="empty")
      with tf.control_dependencies([tf.assert_negative(empty)]):
        out = tf.identity(empty)
      out.eval()


class AssertPositiveTest(tf.test.TestCase):

  def test_raises_when_negative(self):
    with self.test_session():
      freddie = tf.constant([-1, -2], name="freddie")
      with tf.control_dependencies([tf.assert_positive(freddie)]):
        out = tf.identity(freddie)
      with self.assertRaisesOpError("freddie"):
        out.eval()

  def test_doesnt_raise_when_positive(self):
    with self.test_session():
      remmy = tf.constant([1, 2], name="remmy")
      with tf.control_dependencies([tf.assert_positive(remmy)]):
        out = tf.identity(remmy)
      out.eval()

  def test_raises_when_zero(self):
    with self.test_session():
      meechum = tf.constant([0], name="meechum")
      with tf.control_dependencies([tf.assert_positive(meechum)]):
        out = tf.identity(meechum)
      with self.assertRaisesOpError("meechum"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is positive when it satisfies:
    #   For every element x_i in x, x_i > 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = tf.constant([], name="empty")
      with tf.control_dependencies([tf.assert_positive(empty)]):
        out = tf.identity(empty)
      out.eval()


class AssertRankTest(tf.test.TestCase):

  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = tf.constant(1, name="my_tensor")
      desired_rank = 1
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
          tf.identity(tensor).eval()

  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          tf.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = tf.constant(1, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        tf.identity(tensor).eval()

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        tf.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_one_tensor_raises_if_rank_too_large_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 0
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
          tf.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_large_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        tf.identity(tensor).eval()

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 2
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
          tf.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 2
      with tf.control_dependencies([tf.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})


class AssertRankAtLeastTest(tf.test.TestCase):

  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = tf.constant(1, name="my_tensor")
      desired_rank = 1
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                              desired_rank)]):
          tf.identity(tensor).eval()

  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          tf.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = tf.constant(1, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval()

  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval(feed_dict={tensor: 0})

  def test_rank_one_ten_doesnt_raise_raise_if_rank_too_large_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval()

  def test_rank_one_ten_doesnt_raise_if_rank_too_large_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 0
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval()

  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 1
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    with self.test_session():
      tensor = tf.constant([1, 2], name="my_tensor")
      desired_rank = 2
      with self.assertRaisesRegexp(ValueError, "my_tensor.*rank"):
        with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                              desired_rank)]):
          tf.identity(tensor).eval()

  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.test_session():
      tensor = tf.placeholder(tf.float32, name="my_tensor")
      desired_rank = 2
      with tf.control_dependencies([tf.assert_rank_at_least(tensor,
                                                            desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          tf.identity(tensor).eval(feed_dict={tensor: [1, 2]})


class AssertNonNegativeTest(tf.test.TestCase):

  def test_raises_when_negative(self):
    with self.test_session():
      zoe = tf.constant([-1, -2], name="zoe")
      with tf.control_dependencies([tf.assert_non_negative(zoe)]):
        out = tf.identity(zoe)
      with self.assertRaisesOpError("zoe"):
        out.eval()

  def test_doesnt_raise_when_zero_and_positive(self):
    with self.test_session():
      lucas = tf.constant([0, 2], name="lucas")
      with tf.control_dependencies([tf.assert_non_negative(lucas)]):
        out = tf.identity(lucas)
      out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-negative when it satisfies:
    #   For every element x_i in x, x_i >= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = tf.constant([], name="empty")
      with tf.control_dependencies([tf.assert_non_negative(empty)]):
        out = tf.identity(empty)
      out.eval()


class AssertNonPositiveTest(tf.test.TestCase):

  def test_doesnt_raise_when_zero_and_negative(self):
    with self.test_session():
      tom = tf.constant([0, -2], name="tom")
      with tf.control_dependencies([tf.assert_non_positive(tom)]):
        out = tf.identity(tom)
      out.eval()

  def test_raises_when_positive(self):
    with self.test_session():
      rachel = tf.constant([0, 2], name="rachel")
      with tf.control_dependencies([tf.assert_non_positive(rachel)]):
        out = tf.identity(rachel)
      with self.assertRaisesOpError("rachel"):
        out.eval()

  def test_empty_tensor_doesnt_raise(self):
    # A tensor is non-positive when it satisfies:
    #   For every element x_i in x, x_i <= 0
    # and an empty tensor has no elements, so this is trivially satisfied.
    # This is standard set theory.
    with self.test_session():
      empty = tf.constant([], name="empty")
      with tf.control_dependencies([tf.assert_non_positive(empty)]):
        out = tf.identity(empty)
      out.eval()


class IsStrictlyIncreasingTest(tf.test.TestCase):

  def test_constant_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(tf.is_strictly_increasing([1, 1, 1]).eval())

  def test_decreasing_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(tf.is_strictly_increasing([1, 0, -1]).eval())

  def test_2d_decreasing_tensor_is_not_strictly_increasing(self):
    with self.test_session():
      self.assertFalse(tf.is_strictly_increasing([[1, 3], [2, 4]]).eval())

  def test_increasing_tensor_is_increasing(self):
    with self.test_session():
      self.assertTrue(tf.is_strictly_increasing([1, 2, 3]).eval())

  def test_increasing_rank_two_tensor(self):
    with self.test_session():
      self.assertTrue(tf.is_strictly_increasing([[-1, 2], [3, 4]]).eval())

  def test_tensor_with_one_element_is_strictly_increasing(self):
    with self.test_session():
      self.assertTrue(tf.is_strictly_increasing([1]).eval())

  def test_empty_tensor_is_strictly_increasing(self):
    with self.test_session():
      self.assertTrue(tf.is_strictly_increasing([]).eval())


class IsNonDecreasingTest(tf.test.TestCase):

  def test_constant_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(tf.is_non_decreasing([1, 1, 1]).eval())

  def test_decreasing_tensor_is_not_non_decreasing(self):
    with self.test_session():
      self.assertFalse(tf.is_non_decreasing([3, 2, 1]).eval())

  def test_2d_decreasing_tensor_is_not_non_decreasing(self):
    with self.test_session():
      self.assertFalse(tf.is_non_decreasing([[1, 3], [2, 4]]).eval())

  def test_increasing_rank_one_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(tf.is_non_decreasing([1, 2, 3]).eval())

  def test_increasing_rank_two_tensor(self):
    with self.test_session():
      self.assertTrue(tf.is_non_decreasing([[-1, 2], [3, 3]]).eval())

  def test_tensor_with_one_element_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(tf.is_non_decreasing([1]).eval())

  def test_empty_tensor_is_non_decreasing(self):
    with self.test_session():
      self.assertTrue(tf.is_non_decreasing([]).eval())


if __name__ == "__main__":
  tf.test.main()
