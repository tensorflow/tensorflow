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
"""tensor_util tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
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


class AssertScalarTest(tf.test.TestCase):

  def test_assert_scalar(self):
    tf.contrib.framework.assert_scalar(tf.constant(3))
    tf.contrib.framework.assert_scalar(tf.constant("foo"))
    tf.contrib.framework.assert_scalar(3)
    tf.contrib.framework.assert_scalar("foo")
    with self.assertRaisesRegexp(ValueError, "Unexpected shape"):
      tf.contrib.framework.assert_scalar(tf.constant([3, 4]))

  def test_assert_scalar_int(self):
    tf.contrib.framework.assert_scalar_int(tf.constant(3, dtype=tf.int32))
    tf.contrib.framework.assert_scalar_int(tf.constant(3, dtype=tf.int64))
    tf.contrib.framework.assert_scalar_int(3)
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


class WithShapeTest(tf.test.TestCase):

  def _assert_with_shape(
      self, tensor, expected_value, expected_shape, unexpected_shapes):
    for unexpected_shape in unexpected_shapes:
      self.assertRaises(
          ValueError, tf.contrib.framework.with_shape, unexpected_shape, tensor)
      pattern = (
          r"\[Wrong shape for %s \[expected\] \[actual\].\] \[%s\] \[%s\]" %
          (tensor.name,
           " ".join([str(dim) for dim in unexpected_shape]),
           " ".join([str(dim) for dim in expected_shape])))
      self.assertRaisesRegexp(
          tf.OpError,
          re.compile(pattern),
          tf.contrib.framework.with_shape(
              tf.constant(unexpected_shape), tensor).eval)
      expected_placeholder = tf.placeholder(tf.float32)
      self.assertRaisesRegexp(
          tf.OpError,
          re.compile(pattern),
          tf.contrib.framework.with_same_shape(
              expected_placeholder, tensor).eval, {
                  expected_placeholder: np.ones(unexpected_shape)
              })

    self.assertIs(tensor, tf.contrib.framework.with_shape(
        expected_shape, tensor))
    self.assertIs(tensor, tf.contrib.framework.with_same_shape(
        tf.constant(1, shape=expected_shape), tensor))
    tensor_with_shape = tf.contrib.framework.with_shape(
        tf.constant(expected_shape), tensor)
    np.testing.assert_array_equal(expected_value, tensor_with_shape.eval())
    tensor_with_same_shape = tf.contrib.framework.with_same_shape(
        expected_placeholder, tensor)
    np.testing.assert_array_equal(expected_value, tensor_with_same_shape.eval({
        expected_placeholder: np.ones(expected_shape)
    }))

  def test_with_shape_invalid_expected_shape(self):
    with self.test_session():
      self.assertRaisesRegexp(
          ValueError, "Invalid rank", tf.contrib.framework.with_shape,
          [[1], [2]], tf.constant(1.0))

  def test_with_shape_invalid_type(self):
    with self.test_session():
      self.assertRaisesRegexp(
          ValueError, "Invalid dtype", tf.contrib.framework.with_shape,
          [1.1], tf.constant([1.0]))
      self.assertRaisesRegexp(
          ValueError, "Invalid dtype", tf.contrib.framework.with_shape,
          np.array([1.1]), tf.constant(1.0))
      self.assertRaisesRegexp(
          ValueError, "Invalid dtype", tf.contrib.framework.with_shape,
          tf.constant(np.array([1.1])), tf.constant(1.0))

  def test_with_shape_0(self):
    with self.test_session():
      value = 42
      shape = [0]
      unexpected_shapes = [[1], [2], [1, 1]]
      self._assert_with_shape(
          tf.constant(value, shape=shape), value, shape, unexpected_shapes)

  def test_with_shape_1(self):
    with self.test_session():
      value = [42]
      shape = [1]
      unexpected_shapes = [[0], [2], [1, 1]]
      self._assert_with_shape(
          tf.constant(value, shape=shape), value, shape, unexpected_shapes)

  def test_with_shape_2(self):
    with self.test_session():
      value = [42, 43]
      shape = [2]
      unexpected_shapes = [[0], [1], [2, 1]]
      self._assert_with_shape(
          tf.constant(value, shape=shape), value, shape, unexpected_shapes)

  def test_with_shape_2x2(self):
    with self.test_session():
      value = [[42, 43], [44, 45]]
      shape = [2, 2]
      unexpected_shapes = [[0], [1], [2, 1]]
      self._assert_with_shape(
          tf.constant(value, shape=shape), value, shape, unexpected_shapes)

  def test_with_shape_none(self):
    with self.test_session():
      tensor_no_shape = tf.placeholder(tf.float32)

      compatible_shape = [2, 2]
      with_present_2x2 = tf.contrib.framework.with_shape(
          compatible_shape, tensor_no_shape)
      self.assertEquals(compatible_shape, with_present_2x2.get_shape().dims)
      with_future_2x2 = tf.contrib.framework.with_shape(
          tf.constant(compatible_shape), tensor_no_shape)

      array_2x2 = [[42.0, 43.0], [44.0, 45.0]]
      for tensor_2x2 in [with_present_2x2, with_future_2x2]:
        np.testing.assert_array_equal(
            array_2x2, tensor_2x2.eval({tensor_no_shape: array_2x2}))
        self.assertRaisesRegexp(
            tf.OpError, "Wrong shape", tensor_2x2.eval,
            {tensor_no_shape: [42.0, 43.0]})
        self.assertRaisesRegexp(
            tf.OpError, "Wrong shape", tensor_2x2.eval,
            {tensor_no_shape: [42.0]})

  def test_with_shape_partial(self):
    with self.test_session():
      tensor_partial_shape = tf.placeholder(tf.float32)
      tensor_partial_shape.set_shape([None, 2])

      for incompatible_shape in [[0], [1]]:
        self.assertRaisesRegexp(
            ValueError, r"Shapes \(\?, 2\) and \([01],\) are not compatible",
            tf.contrib.framework.with_shape,
            incompatible_shape, tensor_partial_shape)
      for incompatible_shape in [[1, 2, 1]]:
        self.assertRaisesRegexp(
            ValueError, "Dimensions must be equal",
            tf.contrib.framework.with_shape,
            incompatible_shape, tensor_partial_shape)
      for incompatible_shape in [[2, 1]]:
        self.assertRaisesRegexp(
            ValueError, r"Shapes \(\?, 2\) and \(2, 1\) are not compatible",
            tf.contrib.framework.with_shape,
            incompatible_shape, tensor_partial_shape)

      compatible_shape = [2, 2]
      with_present_2x2 = tf.contrib.framework.with_shape(
          compatible_shape, tensor_partial_shape)
      self.assertEquals(compatible_shape, with_present_2x2.get_shape().dims)
      with_future_2x2 = tf.contrib.framework.with_shape(
          tf.constant(compatible_shape), tensor_partial_shape)

      array_2x2 = [[42.0, 43.0], [44.0, 45.0]]
      for tensor_2x2 in [with_present_2x2, with_future_2x2]:
        np.testing.assert_array_equal(
            array_2x2, tensor_2x2.eval({tensor_partial_shape: array_2x2}))
        self.assertRaises(
            ValueError, tensor_2x2.eval, {tensor_partial_shape: [42.0, 43.0]})
        self.assertRaises(
            ValueError, tensor_2x2.eval, {tensor_partial_shape: [42.0]})


class RemoveSqueezableDimensionsTest(tf.test.TestCase):

  def testRemoveSqueezableDimensions(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticLabel_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=True,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraPredictionDim_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=True,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_static(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticPrediction_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=True,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=True,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  # TODO(ptucker): Replace this with parameterized test.
  def _testRemoveSqueezableDimensions(
      self,
      predictions_have_static_shape,
      predictions_have_extra_dim,
      labels_have_static_shape,
      labels_have_extra_dim):
    assert not (predictions_have_extra_dim and labels_have_extra_dim)
    predictions_value = (0, 1, 1, 0, 0, 1, 0)
    labels_value = (0, 0, 1, 1, 0, 0, 0)

    input_predictions_value = (
        [[p] for p in predictions_value] if predictions_have_extra_dim else
        predictions_value)
    input_labels_value = (
        [[l] for l in labels_value] if labels_have_extra_dim else labels_value)

    with tf.Graph().as_default() as g:
      feed_dict = {}
      if predictions_have_static_shape:
        predictions = tf.constant(input_predictions_value, dtype=tf.int32)
      else:
        predictions = tf.placeholder(dtype=tf.int32, name="predictions")
        feed_dict[predictions] = input_predictions_value
      if labels_have_static_shape:
        labels = tf.constant(input_labels_value, dtype=tf.int32)
      else:
        labels = tf.placeholder(dtype=tf.int32, name="labels")
        feed_dict[labels] = input_labels_value

      squeezed_predictions, squeezed_labels = (
          tf.contrib.framework.remove_squeezable_dimensions(
              predictions, labels))
      with self.test_session(g):
        tf.local_variables_initializer().run()
        self.assertAllClose(
            predictions_value, squeezed_predictions.eval(feed_dict=feed_dict))
        self.assertAllClose(
            labels_value, squeezed_labels.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  tf.test.main()
