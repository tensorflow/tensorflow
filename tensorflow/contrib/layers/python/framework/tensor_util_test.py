# Copyright 2015 Google Inc. All Rights Reserved.
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
"""DType tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.python.framework


class FloatDTypeTest(tf.test.TestCase):

  def test_assert_same_float_dtype(self):
    self.assertIs(
        tf.float32, tf.contrib.layers.assert_same_float_dtype(None, None))
    self.assertIs(
        tf.float32, tf.contrib.layers.assert_same_float_dtype([], None))
    self.assertIs(
        tf.float32, tf.contrib.layers.assert_same_float_dtype([], tf.float32))
    self.assertIs(
        tf.float32,
        tf.contrib.layers.assert_same_float_dtype(None, tf.float32))
    self.assertIs(
        tf.float32,
        tf.contrib.layers.assert_same_float_dtype([None, None], None))
    self.assertIs(
        tf.float32,
        tf.contrib.layers.assert_same_float_dtype([None, None], tf.float32))

    const_float = tf.constant(3.0, dtype=tf.float32)
    self.assertIs(
        tf.float32,
        tf.contrib.layers.assert_same_float_dtype([const_float], tf.float32))
    self.assertRaises(
        ValueError,
        tf.contrib.layers.assert_same_float_dtype, [const_float], tf.int32)

    sparse_float = tf.SparseTensor(
        tf.constant([[111], [232]], tf.int64),
        tf.constant([23.4, -43.2], tf.float32),
        tf.constant([500], tf.int64))
    self.assertIs(tf.float32, tf.contrib.layers.assert_same_float_dtype(
        [sparse_float], tf.float32))
    self.assertRaises(
        ValueError,
        tf.contrib.layers.assert_same_float_dtype, [sparse_float], tf.int32)
    self.assertRaises(
        ValueError, tf.contrib.layers.assert_same_float_dtype,
        [const_float, None, sparse_float], tf.float64)

    self.assertIs(
        tf.float32,
        tf.contrib.layers.assert_same_float_dtype([const_float, sparse_float]))
    self.assertIs(tf.float32, tf.contrib.layers.assert_same_float_dtype(
        [const_float, sparse_float], tf.float32))

    const_int = tf.constant(3, dtype=tf.int32)
    self.assertRaises(ValueError, tf.contrib.layers.assert_same_float_dtype,
                      [sparse_float, const_int])
    self.assertRaises(ValueError, tf.contrib.layers.assert_same_float_dtype,
                      [sparse_float, const_int], tf.int32)
    self.assertRaises(ValueError, tf.contrib.layers.assert_same_float_dtype,
                      [sparse_float, const_int], tf.float32)
    self.assertRaises(
        ValueError, tf.contrib.layers.assert_same_float_dtype, [const_int])


if __name__ == "__main__":
  tf.test.main()
