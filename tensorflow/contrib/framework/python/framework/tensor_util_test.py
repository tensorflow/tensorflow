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
"""tensor_util tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FloatDTypeTest(tf.test.TestCase):

  def test_assert_same_float_dtype(self):
    self.assertIs(
        tf.float32, tf.contrib.framework.assert_same_float_dtype(None, None))
    self.assertIs(
        tf.float32, tf.contrib.framework.assert_same_float_dtype([], None))
    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype([], tf.float32))
    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype(None, tf.float32))
    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype([None, None], None))
    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype([None, None], tf.float32))

    const_float = tf.constant(3.0, dtype=tf.float32)
    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype([const_float], tf.float32))
    self.assertRaises(
        ValueError,
        tf.contrib.framework.assert_same_float_dtype, [const_float], tf.int32)

    sparse_float = tf.SparseTensor(
        tf.constant([[111], [232]], tf.int64),
        tf.constant([23.4, -43.2], tf.float32),
        tf.constant([500], tf.int64))
    self.assertIs(tf.float32, tf.contrib.framework.assert_same_float_dtype(
        [sparse_float], tf.float32))
    self.assertRaises(
        ValueError,
        tf.contrib.framework.assert_same_float_dtype, [sparse_float], tf.int32)
    self.assertRaises(
        ValueError, tf.contrib.framework.assert_same_float_dtype,
        [const_float, None, sparse_float], tf.float64)

    self.assertIs(
        tf.float32,
        tf.contrib.framework.assert_same_float_dtype(
            [const_float, sparse_float]))
    self.assertIs(tf.float32, tf.contrib.framework.assert_same_float_dtype(
        [const_float, sparse_float], tf.float32))

    const_int = tf.constant(3, dtype=tf.int32)
    self.assertRaises(ValueError, tf.contrib.framework.assert_same_float_dtype,
                      [sparse_float, const_int])
    self.assertRaises(ValueError, tf.contrib.framework.assert_same_float_dtype,
                      [sparse_float, const_int], tf.int32)
    self.assertRaises(ValueError, tf.contrib.framework.assert_same_float_dtype,
                      [sparse_float, const_int], tf.float32)
    self.assertRaises(
        ValueError, tf.contrib.framework.assert_same_float_dtype, [const_int])


class AssertScalarIntTest(tf.test.TestCase):

  def test_assert_scalar_int(self):
    tf.contrib.framework.assert_scalar_int(tf.constant(3, dtype=tf.int32))
    tf.contrib.framework.assert_scalar_int(tf.constant(3, dtype=tf.int64))
    with self.assertRaisesRegexp(ValueError, "Unexpected type"):
      tf.contrib.framework.assert_scalar_int(tf.constant(3, dtype=tf.float32))
    with self.assertRaisesRegexp(ValueError, "Unexpected shape"):
      tf.contrib.framework.assert_scalar_int(
          tf.constant([3, 4], dtype=tf.int32))


class LocalVariabletest(tf.test.TestCase):

  def test_local_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], tf.local_variables())
      value0 = 42
      tf.contrib.framework.local_variable(value0)
      value1 = 43
      tf.contrib.framework.local_variable(value1)
      variables = tf.local_variables()
      self.assertEquals(2, len(variables))
      self.assertRaises(tf.OpError, sess.run, variables)
      tf.initialize_variables(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))


class ReduceSumNTest(tf.test.TestCase):

  def test_reduce_sum_n(self):
    with self.test_session():
      a = tf.constant(1)
      b = tf.constant([2])
      c = tf.constant([[3, 4], [5, 6]])
      self.assertEqual(21, tf.contrib.framework.reduce_sum_n([a, b, c]).eval())


if __name__ == "__main__":
  tf.test.main()
