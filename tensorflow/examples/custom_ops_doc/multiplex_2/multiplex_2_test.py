# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multiplex_2."""

import numpy as np
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.python.framework import errors_impl
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.with_eager_op_as_function
class MultiplexOpRank1Test(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_float(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    b = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0])
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    # expected result is [1.0, 20.0, 3.0, 40.0, 5.0]
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples>MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int64 tensor '
        r'\[Op:Examples>MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b' of 'Examples>MultiplexDense' Op has type int64 that "
        r"does not match type float32 of argument 'a'.)"):
      self.evaluate(multiplex_2_op.multiplex(cond, a, b))

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_size(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)  # longer than b
    b = tf.constant([10, 20], dtype=tf.int64)  # shorter than a
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(?s)(a and b must have the same shape. '
        r'a shape: \[5\] b shape: \[2\].* '
        r'\[Op:Examples>MultiplexDense\]'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Dimension 0 in both shapes must be equal, but are 5 and 2\. '
        r'Shapes are \[5\] and \[2\]\.)'):
      self.evaluate(multiplex_2_op.multiplex(cond, a, b))

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_2d(self):
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
    b = tf.constant([[10, 20, 30], [40, 50, 60]], dtype=tf.int64)
    cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [[1, 20], [3, 40]]
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_shape(self):
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)  # shape (2,3)
    b = tf.constant([[10, 20], [30, 40], [50, 60]],
                    dtype=tf.int64)  # shape (3,2)
    cond = tf.constant([[True, False, True], [False, True, False]], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(a and b must have the same shape.'
        r' a shape: \[2,3\] b shape: \[3,2\]'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Dimension 0 in both shapes must be equal, '
        r'but are 2 and 3\. Shapes are \[2,3\] and \[3,2\])\.'):
      self.evaluate(multiplex_2_op.multiplex(cond, a, b))


if __name__ == '__main__':
  tf.test.main()
