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


class ReshapeBijectorTest(test.TestCase):
  """Tests correctness of the reshape transformation."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijector(self):
    """Do a basic sanity check of forward, inverse, jacobian."""
    expected_x = np.random.randn(4, 3, 2)
    expected_y = np.reshape(expected_x, [4, 6])

    with self.test_session() as sess:
      bijector = Reshape(
          event_shape_out=[6,],
          event_shape_in=[3, 2],
          validate_args=True)
      (x_,
       y_,
       fldj_,
       ildj_) = sess.run((
           bijector.inverse(expected_y),
           bijector.forward(expected_x),
           bijector.forward_log_det_jacobian(expected_x),
           bijector.inverse_log_det_jacobian(expected_y),
       ))
      self.assertEqual("reshape", bijector.name)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
      self.assertAllClose(0., fldj_, rtol=1e-6, atol=0)
      self.assertAllClose(0., ildj_, rtol=1e-6, atol=0)

  def testEventShapeDynamicNdims(self):
    """Check forward/inverse shape methods with dynamic ndims."""

    shape_in = tensor_shape.TensorShape([6,])
    shape_in_ph = array_ops.placeholder(dtype=dtypes.int32)

    shape_out = tensor_shape.TensorShape([2, 3])
    shape_out_ph = array_ops.placeholder(dtype=dtypes.int32)

    bijector = Reshape(
        event_shape_out=shape_out_ph,
        event_shape_in=shape_in_ph, validate_args=True)

    # using the _tensor methods, we should always get a fully-specified
    # result since these are evaluated at graph runtime.
    with self.test_session() as sess:
      (shape_out_,
       shape_in_) = sess.run((
           bijector.forward_event_shape_tensor(shape_in),
           bijector.inverse_event_shape_tensor(shape_out),
       ), feed_dict={
           shape_in_ph: shape_in,
           shape_out_ph: shape_out,
       })
      self.assertAllEqual(shape_out, shape_out_)
      self.assertAllEqual(shape_in, shape_in_)

  def testEventShapeDynamic(self):
    """Check shape methods with static ndims but dynamic shape."""

    shape_in = tensor_shape.TensorShape([6,])
    shape_in_partial = tensor_shape.TensorShape([None,])
    shape_in_ph = array_ops.placeholder(
        shape=[1,], dtype=dtypes.int32)

    shape_out = tensor_shape.TensorShape([2, 3])
    shape_out_partial = tensor_shape.TensorShape([None, None])
    shape_out_ph = array_ops.placeholder(
        shape=[2,], dtype=dtypes.int32)

    bijector = Reshape(
        event_shape_out=shape_out_ph,
        event_shape_in=shape_in_ph,
        validate_args=True)

    # if event shapes are not statically available, should
    # return partially-specified TensorShapes.
    self.assertAllEqual(
        bijector.forward_event_shape(shape_in).as_list(),
        shape_out_partial.as_list())
    self.assertAllEqual(
        bijector.inverse_event_shape(shape_out).as_list(),
        shape_in_partial.as_list())

    # using the _tensor methods, we should always get a fully-specified
    # result since these are evaluated at graph runtime.
    with self.test_session() as sess:
      (shape_out_,
       shape_in_) = sess.run((
           bijector.forward_event_shape_tensor(shape_in),
           bijector.inverse_event_shape_tensor(shape_out),
       ), feed_dict={
           shape_in_ph: shape_in,
           shape_out_ph: shape_out,
       })
      self.assertAllEqual(shape_out, shape_out_)
      self.assertAllEqual(shape_in, shape_in_)

  def testEventShapeStatic(self):
    """Check shape methods when shape is statically known."""

    shape_in = tensor_shape.TensorShape([6,])
    shape_out = tensor_shape.TensorShape([2, 3])

    bijector_static = Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)

    # test that forward_ and inverse_event_shape do sensible things
    # when shapes are statically known.
    self.assertEqual(
        bijector_static.forward_event_shape(shape_in),
        shape_out)
    self.assertEqual(
        bijector_static.inverse_event_shape(shape_out),
        shape_in)

    with self.test_session() as sess:
      (shape_out_static_,
       shape_in_static_,
      ) = sess.run((
          bijector_static.forward_event_shape_tensor(shape_in),
          bijector_static.inverse_event_shape_tensor(shape_out),
      ))
      self.assertAllEqual(shape_out, shape_out_static_)
      self.assertAllEqual(shape_in, shape_in_static_)

  def testScalarReshape(self):
    """Test reshaping to and from a scalar shape ()."""

    expected_x = np.random.randn(4, 3, 1)
    expected_y = np.reshape(expected_x, [4, 3])

    expected_x_scalar = np.random.randn(1,)
    expected_y_scalar = expected_x_scalar[0]

    with self.test_session() as sess:
      bijector = Reshape(
          event_shape_out=[],
          event_shape_in=[1,], validate_args=True)

      (x_,
       y_,
       x_scalar_,
       y_scalar_
      ) = sess.run((
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
          bijector.inverse(expected_y_scalar),
          bijector.forward(expected_x_scalar),
      ))
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_y_scalar, y_scalar_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x_scalar, x_scalar_, rtol=1e-6, atol=0)

  def testRaisesOpError(self):
    x1 = np.random.randn(4, 2, 3)
    x2 = np.random.randn(4, 3, 2)
    x3 = np.random.randn(4, 5, 1, 1)

    with self.test_session() as sess:
      shape_in_ph = array_ops.placeholder(shape=[2,], dtype=dtypes.int32)
      shape_out_ph = array_ops.placeholder(shape=[3,], dtype=dtypes.int32)
      bijector = Reshape(
          event_shape_out=shape_out_ph,
          event_shape_in=shape_in_ph,
          validate_args=True)

      with self.assertRaisesOpError(
          "Input `event_shape` does not match `event_shape_in`."):
        sess.run(bijector.forward(x2),
                 feed_dict={shape_out_ph: [1, 6, 1],
                            shape_in_ph: [2, 3]})

      with self.assertRaisesOpError(
          "event_shape_out entries must be positive."):
        sess.run(bijector.forward(x1),
                 feed_dict={shape_out_ph: [-1, -1, 6],
                            shape_in_ph: [2, 3]})

      # test that *all* methods check basic assertions
      fd_mismatched = {shape_out_ph: [1, 1, 5], shape_in_ph: [2, 3]}
      with self.assertRaisesOpError(
          "Input/output `event_size`s do not match."):
        sess.run(bijector.forward(x1), feed_dict=fd_mismatched)
      with self.assertRaisesOpError(
          "Input/output `event_size`s do not match."):
        sess.run(bijector.inverse(x3), feed_dict=fd_mismatched)
      with self.assertRaisesOpError(
          "Input/output `event_size`s do not match."):
        sess.run(bijector.inverse_log_det_jacobian(x3),
                 feed_dict=fd_mismatched)
      with self.assertRaisesOpError(
          "Input/output `event_size`s do not match."):
        sess.run(bijector.forward_log_det_jacobian(x1),
                 feed_dict=fd_mismatched)

  def testBijectiveAndFinite(self):
    x = np.random.randn(4, 2, 3)
    y = np.reshape(x, [4, 1, 2, 3])
    with self.test_session():
      bijector = Reshape(
          event_shape_in=[2, 3],
          event_shape_out=[1, 2, 3],
          validate_args=True)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-6, atol=0)

if __name__ == "__main__":
  test.main()
