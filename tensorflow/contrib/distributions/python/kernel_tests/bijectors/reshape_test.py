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
"""Tests for Reshape Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors.reshape import Reshape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.platform import test


class _ReshapeBijectorTest(object):
  """Base class for testing the reshape transformation.

  Methods defined in this class call a method self.build_shapes() that
  is implemented by subclasses defined below, returning respectively
   ReshapeBijectorTestStatic: static shapes,
   ReshapeBijectorTestDynamic: shape placeholders of known ndims, and
   ReshapeBijectorTestDynamicNdims: shape placeholders of unspecified ndims,
  so that each test in this base class is automatically run over all
  three cases. The subclasses also implement assertRaisesError to test
  for either Python exceptions (in the case of static shapes) or
  TensorFlow op errors (dynamic shapes).
  """

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijector(self):
    """Do a basic sanity check of forward, inverse, jacobian."""
    expected_x = np.random.randn(4, 3, 2)
    expected_y = np.reshape(expected_x, [4, 6])

    with self.cached_session() as sess:
      shape_in, shape_out, feed_dict = self.build_shapes([3, 2], [6,])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      (x_,
       y_,
       fldj_,
       ildj_) = sess.run((
           bijector.inverse(expected_y),
           bijector.forward(expected_x),
           bijector.forward_log_det_jacobian(expected_x, event_ndims=2),
           bijector.inverse_log_det_jacobian(expected_y, event_ndims=2),
       ), feed_dict=feed_dict)
      self.assertEqual("reshape", bijector.name)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
      self.assertAllClose(0., fldj_, rtol=1e-6, atol=0)
      self.assertAllClose(0., ildj_, rtol=1e-6, atol=0)

  def testEventShapeTensor(self):
    """Test event_shape_tensor methods when even ndims may be dynamic."""

    shape_in_static = [2, 3]
    shape_out_static = [6,]
    shape_in, shape_out, feed_dict = self.build_shapes(shape_in_static,
                                                       shape_out_static)
    bijector = Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in, validate_args=True)

    # using the _tensor methods, we should always get a fully-specified
    # result since these are evaluated at graph runtime.
    with self.cached_session() as sess:
      (shape_out_,
       shape_in_) = sess.run((
           bijector.forward_event_shape_tensor(shape_in),
           bijector.inverse_event_shape_tensor(shape_out),
       ), feed_dict=feed_dict)
      self.assertAllEqual(shape_out_static, shape_out_)
      self.assertAllEqual(shape_in_static, shape_in_)

  def testScalarReshape(self):
    """Test reshaping to and from a scalar shape ()."""

    expected_x = np.random.randn(4, 3, 1)
    expected_y = np.reshape(expected_x, [4, 3])

    expected_x_scalar = np.random.randn(1,)
    expected_y_scalar = expected_x_scalar[0]

    shape_in, shape_out, feed_dict = self.build_shapes([], [1,])
    with self.cached_session() as sess:
      bijector = Reshape(
          event_shape_out=shape_in,
          event_shape_in=shape_out, validate_args=True)
      (x_,
       y_,
       x_scalar_,
       y_scalar_
      ) = sess.run((
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
          bijector.inverse(expected_y_scalar),
          bijector.forward(expected_x_scalar),
      ), feed_dict=feed_dict)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_y_scalar, y_scalar_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x_scalar, x_scalar_, rtol=1e-6, atol=0)

  def testMultipleUnspecifiedDimensionsOpError(self):

    with self.cached_session() as sess:
      shape_in, shape_out, feed_dict = self.build_shapes([2, 3], [4, -1, -1,])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)

      with self.assertRaisesError(
          "elements must have at most one `-1`."):
        sess.run(bijector.forward_event_shape_tensor(shape_in),
                 feed_dict=feed_dict)

  # pylint: disable=invalid-name
  def _testInvalidDimensionsOpError(self, expected_error_message):

    with self.cached_session() as sess:

      shape_in, shape_out, feed_dict = self.build_shapes([2, 3], [1, 2, -2,])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)

      with self.assertRaisesError(expected_error_message):
        sess.run(bijector.forward_event_shape_tensor(shape_in),
                 feed_dict=feed_dict)
  # pylint: enable=invalid-name

  def testValidButNonMatchingInputOpError(self):
    x = np.random.randn(4, 3, 2)

    with self.cached_session() as sess:
      shape_in, shape_out, feed_dict = self.build_shapes([2, 3], [1, 6, 1,])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)

      # Here we pass in a tensor (x) whose shape is compatible with
      # the output shape, so tf.reshape will throw no error, but
      # doesn't match the expected input shape.
      with self.assertRaisesError(
          "Input `event_shape` does not match `event_shape_in`."):
        sess.run(bijector.forward(x),
                 feed_dict=feed_dict)

  def testValidButNonMatchingInputPartiallySpecifiedOpError(self):
    x = np.random.randn(4, 3, 2)

    with self.cached_session() as sess:
      shape_in, shape_out, feed_dict = self.build_shapes([2, -1], [1, 6, 1,])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)

      with self.assertRaisesError(
          "Input `event_shape` does not match `event_shape_in`."):
        sess.run(bijector.forward(x),
                 feed_dict=feed_dict)

  # pylint: disable=invalid-name
  def _testInputOutputMismatchOpError(self, expected_error_message):
    x1 = np.random.randn(4, 2, 3)
    x2 = np.random.randn(4, 1, 1, 5)

    with self.cached_session() as sess:
      shape_in, shape_out, fd_mismatched = self.build_shapes([2, 3],
                                                             [1, 1, 5])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)

      with self.assertRaisesError(expected_error_message):
        sess.run(bijector.forward(x1), feed_dict=fd_mismatched)
      with self.assertRaisesError(expected_error_message):
        sess.run(bijector.inverse(x2), feed_dict=fd_mismatched)
  # pylint: enable=invalid-name

  def testOneShapePartiallySpecified(self):
    expected_x = np.random.randn(4, 6)
    expected_y = np.reshape(expected_x, [4, 2, 3])

    with self.cached_session() as sess:
      # one of input/output shapes is partially specified
      shape_in, shape_out, feed_dict = self.build_shapes([-1,], [2, 3])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      (x_,
       y_,
      ) = sess.run((
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
      ), feed_dict=feed_dict)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def testBothShapesPartiallySpecified(self):
    expected_x = np.random.randn(4, 2, 3)
    expected_y = np.reshape(expected_x, [4, 3, 2])
    with self.cached_session() as sess:
      shape_in, shape_out, feed_dict = self.build_shapes([-1, 3], [-1, 2])
      bijector = Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      (x_,
       y_,
      ) = sess.run((
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
      ), feed_dict=feed_dict)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def testDefaultVectorShape(self):
    expected_x = np.random.randn(4, 4)
    expected_y = np.reshape(expected_x, [4, 2, 2])
    with self.cached_session() as sess:
      _, shape_out, feed_dict = self.build_shapes([-1,], [-1, 2])
      bijector = Reshape(shape_out,
                         validate_args=True)
      (x_,
       y_,
      ) = sess.run((
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
      ), feed_dict=feed_dict)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def build_shapes(self, *args, **kwargs):
    raise NotImplementedError("Subclass failed to implement `build_shapes`.")


class ReshapeBijectorTestStatic(test.TestCase, _ReshapeBijectorTest):

  def build_shapes(self, shape_in, shape_out):
    shape_in_static = shape_in
    shape_out_static = shape_out
    feed_dict = {}
    return shape_in_static, shape_out_static, feed_dict

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testEventShape(self):
    shape_in_static = tensor_shape.TensorShape([2, 3])
    shape_out_static = tensor_shape.TensorShape([6,])
    bijector = Reshape(
        event_shape_out=shape_out_static,
        event_shape_in=shape_in_static, validate_args=True)

    # test that forward_ and inverse_event_shape do sensible things
    # when shapes are statically known.
    self.assertEqual(
        bijector.forward_event_shape(shape_in_static),
        shape_out_static)
    self.assertEqual(
        bijector.inverse_event_shape(shape_out_static),
        shape_in_static)

  def testBijectiveAndFinite(self):
    x = np.random.randn(4, 2, 3)
    y = np.reshape(x, [4, 1, 2, 3])
    with self.cached_session():
      bijector = Reshape(
          event_shape_in=[2, 3],
          event_shape_out=[1, 2, 3],
          validate_args=True)
      assert_bijective_and_finite(
          bijector, x, y, event_ndims=2, rtol=1e-6, atol=0)

  def testInvalidDimensionsOpError(self):
    self._testInvalidDimensionsOpError(
        "Invalid value in tensor used for shape: -2")

  def testInputOutputMismatchOpError(self):
    self._testInputOutputMismatchOpError("Cannot reshape a tensor with")


class ReshapeBijectorTestDynamic(test.TestCase, _ReshapeBijectorTest):

  def build_shapes(self, shape_in, shape_out):
    shape_in_ph = array_ops.placeholder(shape=(len(shape_in),),
                                        dtype=dtypes.int32)
    shape_out_ph = array_ops.placeholder(shape=(len(shape_out),),
                                         dtype=dtypes.int32)
    feed_dict = {shape_in_ph: shape_in, shape_out_ph: shape_out}
    return shape_in_ph, shape_out_ph, feed_dict

  def assertRaisesError(self, msg):
    return self.assertRaisesOpError(msg)

  def testInvalidDimensionsOpError(self):
    self._testInvalidDimensionsOpError(
        "elements must be either positive integers or `-1`.")

  def testInputOutputMismatchOpError(self):
    self._testInputOutputMismatchOpError("Input to reshape is a tensor with")


class ReshapeBijectorTestDynamicNdims(test.TestCase, _ReshapeBijectorTest):

  def build_shapes(self, shape_in, shape_out):
    shape_in_ph = array_ops.placeholder(shape=None, dtype=dtypes.int32)
    shape_out_ph = array_ops.placeholder(shape=None, dtype=dtypes.int32)
    feed_dict = {shape_in_ph: shape_in, shape_out_ph: shape_out}
    return shape_in_ph, shape_out_ph, feed_dict

  def assertRaisesError(self, msg):
    return self.assertRaisesOpError(msg)

  def testInvalidDimensionsOpError(self):
    self._testInvalidDimensionsOpError(
        "elements must be either positive integers or `-1`.")

  def testInputOutputMismatchOpError(self):
    self._testInputOutputMismatchOpError("Input to reshape is a tensor with")


if __name__ == "__main__":
  test.main()
