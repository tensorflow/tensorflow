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

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.framework.python.ops import variables as variables_lib2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class LocalVariabletest(test.TestCase):

  def test_local_variable(self):
    with self.test_session() as sess:
      self.assertEquals([], variables_lib.local_variables())
      value0 = 42
      variables_lib2.local_variable(value0)
      value1 = 43
      variables_lib2.local_variable(value1)
      variables = variables_lib.local_variables()
      self.assertEquals(2, len(variables))
      self.assertRaises(errors_impl.OpError, sess.run, variables)
      variables_lib.initialize_variables(variables).run()
      self.assertAllEqual(set([value0, value1]), set(sess.run(variables)))


class ReduceSumNTest(test.TestCase):

  def test_reduce_sum_n(self):
    with self.test_session():
      a = constant_op.constant(1)
      b = constant_op.constant([2])
      c = constant_op.constant([[3, 4], [5, 6]])
      self.assertEqual(21, tensor_util.reduce_sum_n([a, b, c]).eval())


class AssertScalarIntTest(test.TestCase):

  def test_assert_scalar_int(self):
    tensor_util.assert_scalar_int(constant_op.constant(3, dtype=dtypes.int32))
    tensor_util.assert_scalar_int(constant_op.constant(3, dtype=dtypes.int64))
    tensor_util.assert_scalar_int(3)
    with self.assertRaisesRegexp(ValueError, "Expected integer"):
      tensor_util.assert_scalar_int(
          constant_op.constant(
              3, dtype=dtypes.float32))
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      tensor_util.assert_scalar_int(
          constant_op.constant(
              [3, 4], dtype=dtypes.int32))


@test_util.with_c_api
class WithShapeTest(test.TestCase):

  def _assert_with_shape(self, tensor, expected_value, expected_shape,
                         unexpected_shapes):
    for unexpected_shape in unexpected_shapes:
      self.assertRaises(ValueError, tensor_util.with_shape, unexpected_shape,
                        tensor)
      pattern = (
          r"\[Wrong shape for %s \[expected\] \[actual\].\] \[%s\] \[%s\]" %
          (tensor.name, " ".join([str(dim) for dim in unexpected_shape]),
           " ".join([str(dim) for dim in expected_shape])))
      self.assertRaisesRegexp(errors_impl.OpError,
                              re.compile(pattern),
                              tensor_util.with_shape(
                                  constant_op.constant(unexpected_shape),
                                  tensor).eval)
      expected_placeholder = array_ops.placeholder(dtypes.float32)
      self.assertRaisesRegexp(errors_impl.OpError,
                              re.compile(pattern),
                              tensor_util.with_same_shape(expected_placeholder,
                                                          tensor).eval,
                              {expected_placeholder: np.ones(unexpected_shape)})

    self.assertIs(tensor, tensor_util.with_shape(expected_shape, tensor))
    self.assertIs(
        tensor,
        tensor_util.with_same_shape(
            constant_op.constant(
                1, shape=expected_shape), tensor))
    tensor_with_shape = tensor_util.with_shape(
        constant_op.constant(expected_shape), tensor)
    np.testing.assert_array_equal(expected_value, tensor_with_shape.eval())
    tensor_with_same_shape = tensor_util.with_same_shape(expected_placeholder,
                                                         tensor)
    np.testing.assert_array_equal(expected_value,
                                  tensor_with_same_shape.eval({
                                      expected_placeholder:
                                          np.ones(expected_shape)
                                  }))

  def test_with_shape_invalid_expected_shape(self):
    with self.test_session():
      self.assertRaisesRegexp(ValueError, "Invalid rank",
                              tensor_util.with_shape, [[1], [2]],
                              constant_op.constant(1.0))

  def test_with_shape_invalid_type(self):
    with self.test_session():
      self.assertRaisesRegexp(ValueError, "Invalid dtype",
                              tensor_util.with_shape, [1.1],
                              constant_op.constant([1.0]))
      self.assertRaisesRegexp(ValueError, "Invalid dtype",
                              tensor_util.with_shape,
                              np.array([1.1]), constant_op.constant(1.0))
      self.assertRaisesRegexp(ValueError, "Invalid dtype",
                              tensor_util.with_shape,
                              constant_op.constant(np.array([1.1])),
                              constant_op.constant(1.0))

  def test_with_shape_0(self):
    with self.test_session():
      value = 42
      shape = [0]
      unexpected_shapes = [[1], [2], [1, 1]]
      self._assert_with_shape(
          constant_op.constant(
              value, shape=shape),
          value,
          shape,
          unexpected_shapes)

  def test_with_shape_1(self):
    with self.test_session():
      value = [42]
      shape = [1]
      unexpected_shapes = [[0], [2], [1, 1]]
      self._assert_with_shape(
          constant_op.constant(
              value, shape=shape),
          value,
          shape,
          unexpected_shapes)

  def test_with_shape_2(self):
    with self.test_session():
      value = [42, 43]
      shape = [2]
      unexpected_shapes = [[0], [1], [2, 1]]
      self._assert_with_shape(
          constant_op.constant(
              value, shape=shape),
          value,
          shape,
          unexpected_shapes)

  def test_with_shape_2x2(self):
    with self.test_session():
      value = [[42, 43], [44, 45]]
      shape = [2, 2]
      unexpected_shapes = [[0], [1], [2, 1]]
      self._assert_with_shape(
          constant_op.constant(
              value, shape=shape),
          value,
          shape,
          unexpected_shapes)

  def test_with_shape_none(self):
    with self.test_session():
      tensor_no_shape = array_ops.placeholder(dtypes.float32)

      compatible_shape = [2, 2]
      with_present_2x2 = tensor_util.with_shape(compatible_shape,
                                                tensor_no_shape)
      self.assertEquals(compatible_shape, with_present_2x2.get_shape().dims)
      with_future_2x2 = tensor_util.with_shape(
          constant_op.constant(compatible_shape), tensor_no_shape)

      array_2x2 = [[42.0, 43.0], [44.0, 45.0]]
      for tensor_2x2 in [with_present_2x2, with_future_2x2]:
        np.testing.assert_array_equal(array_2x2,
                                      tensor_2x2.eval({
                                          tensor_no_shape: array_2x2
                                      }))
        self.assertRaisesRegexp(errors_impl.OpError, "Wrong shape",
                                tensor_2x2.eval,
                                {tensor_no_shape: [42.0, 43.0]})
        self.assertRaisesRegexp(errors_impl.OpError, "Wrong shape",
                                tensor_2x2.eval, {tensor_no_shape: [42.0]})

  def test_with_shape_partial(self):
    with self.test_session():
      tensor_partial_shape = array_ops.placeholder(dtypes.float32)
      tensor_partial_shape.set_shape([None, 2])

      for incompatible_shape in [[0], [1]]:
        if ops._USE_C_API:
          error_message = "Shapes must be equal rank, but are 2 and 1"
        else:
          error_message = r"Shapes \(\?, 2\) and \([01],\) are not compatible"
        self.assertRaisesRegexp(
            ValueError, error_message,
            tensor_util.with_shape, incompatible_shape, tensor_partial_shape)
      for incompatible_shape in [[1, 2, 1]]:
        self.assertRaisesRegexp(ValueError, "Dimensions must be equal",
                                tensor_util.with_shape, incompatible_shape,
                                tensor_partial_shape)
      for incompatible_shape in [[2, 1]]:
        if ops._USE_C_API:
          error_message = (r"Dimension 1 in both shapes must be equal, but are "
                           r"2 and 1. Shapes are \[\?,2\] and \[2,1\].")
        else:
          error_message = r"Shapes \(\?, 2\) and \(2, 1\) are not compatible"
        self.assertRaisesRegexp(
            ValueError, error_message,
            tensor_util.with_shape, incompatible_shape, tensor_partial_shape)

      compatible_shape = [2, 2]
      with_present_2x2 = tensor_util.with_shape(compatible_shape,
                                                tensor_partial_shape)
      self.assertEquals(compatible_shape, with_present_2x2.get_shape().dims)
      with_future_2x2 = tensor_util.with_shape(
          constant_op.constant(compatible_shape), tensor_partial_shape)

      array_2x2 = [[42.0, 43.0], [44.0, 45.0]]
      for tensor_2x2 in [with_present_2x2, with_future_2x2]:
        np.testing.assert_array_equal(array_2x2,
                                      tensor_2x2.eval({
                                          tensor_partial_shape: array_2x2
                                      }))
        self.assertRaises(ValueError, tensor_2x2.eval,
                          {tensor_partial_shape: [42.0, 43.0]})
        self.assertRaises(ValueError, tensor_2x2.eval,
                          {tensor_partial_shape: [42.0]})


class RemoveSqueezableDimensionsTest(test.TestCase):

  def testRemoveSqueezableDimensions(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=False,
        labels_have_static_shape=False,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=False,
        labels_have_static_shape=False,
        labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=False,
        labels_have_static_shape=True,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticLabel_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=False,
        labels_have_static_shape=True,
        labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=True,
        labels_have_static_shape=False,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraPredictionDim_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False,
        predictions_have_extra_dim=True,
        labels_have_static_shape=True,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=False,
        labels_have_static_shape=False,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=False,
        labels_have_static_shape=False,
        labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_static(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=False,
        labels_have_static_shape=True,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=False,
        labels_have_static_shape=True,
        labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticPrediction_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=True,
        labels_have_static_shape=False,
        labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True,
        predictions_have_extra_dim=True,
        labels_have_static_shape=True,
        labels_have_extra_dim=False)

  # TODO(ptucker): Replace this with parameterized test.
  def _testRemoveSqueezableDimensions(self, predictions_have_static_shape,
                                      predictions_have_extra_dim,
                                      labels_have_static_shape,
                                      labels_have_extra_dim):
    assert not (predictions_have_extra_dim and labels_have_extra_dim)
    predictions_value = (0, 1, 1, 0, 0, 1, 0)
    labels_value = (0, 0, 1, 1, 0, 0, 0)

    input_predictions_value = ([[p] for p in predictions_value] if
                               predictions_have_extra_dim else
                               predictions_value)
    input_labels_value = ([[l] for l in labels_value] if labels_have_extra_dim
                          else labels_value)

    with ops.Graph().as_default() as g:
      feed_dict = {}
      if predictions_have_static_shape:
        predictions = constant_op.constant(
            input_predictions_value, dtype=dtypes.int32)
      else:
        predictions = array_ops.placeholder(
            dtype=dtypes.int32, name="predictions")
        feed_dict[predictions] = input_predictions_value
      if labels_have_static_shape:
        labels = constant_op.constant(input_labels_value, dtype=dtypes.int32)
      else:
        labels = array_ops.placeholder(dtype=dtypes.int32, name="labels")
        feed_dict[labels] = input_labels_value

      squeezed_predictions, squeezed_labels = (
          tensor_util.remove_squeezable_dimensions(predictions, labels))
      with self.test_session(g):
        variables_lib.local_variables_initializer().run()
        self.assertAllClose(
            predictions_value, squeezed_predictions.eval(feed_dict=feed_dict))
        self.assertAllClose(
            labels_value, squeezed_labels.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  test.main()
