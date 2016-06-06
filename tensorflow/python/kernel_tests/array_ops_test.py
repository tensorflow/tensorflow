# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for array_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class BooleanMaskTest(test_util.TensorFlowTestCase):

  def CheckVersusNumpy(self, ndims_mask, arr_shape, make_mask=None):
    """Check equivalence between boolean_mask and numpy masking."""
    if make_mask is None:
      make_mask = lambda shape: np.random.randint(0, 2, size=shape).astype(bool)
    arr = np.random.rand(*arr_shape)
    mask = make_mask(arr_shape[: ndims_mask])
    masked_arr = arr[mask]
    with self.test_session():
      masked_tensor = array_ops.boolean_mask(arr, mask)
      np.testing.assert_allclose(
          masked_arr,
          masked_tensor.eval(),
          err_msg="masked_arr:\n%s\n\nmasked_tensor:\n%s" % (
              masked_arr, masked_tensor.eval()))
      masked_tensor.get_shape().assert_is_compatible_with(masked_arr.shape)
      self.assertSequenceEqual(
          masked_tensor.get_shape()[1:].as_list(),
          masked_arr.shape[1:],
          msg="shape information lost %s -> %s" % (
              masked_arr.shape, masked_tensor.get_shape()))

  def testOneDimensionalMask(self):
    # Do 1d separately because it's the only easy one to debug!
    ndims_mask = 1
    for ndims_arr in range(ndims_mask, ndims_mask + 3):
      for _ in range(3):
        arr_shape = np.random.randint(1, 5, size=ndims_arr)
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMultiDimensionalMask(self):
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          arr_shape = np.random.randint(1, 5, size=ndims_arr)
          self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testEmptyOutput(self):
    make_mask = lambda shape: np.zeros(shape, dtype=bool)
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          arr_shape = np.random.randint(1, 5, size=ndims_arr)
          self.CheckVersusNumpy(ndims_mask, arr_shape, make_mask=make_mask)

  def testWorksWithDimensionsEqualToNoneDuringGraphBuild(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session() as sess:
      ph_tensor = array_ops.placeholder(dtypes.int32, shape=None)
      ph_mask = array_ops.placeholder(dtypes.bool, shape=[None])

      arr = np.array([[1, 2], [3, 4]])
      mask = np.array([False, True])

      masked_tensor = sess.run(
          array_ops.boolean_mask(ph_tensor, ph_mask),
          feed_dict={ph_tensor: arr, ph_mask: mask})
      np.testing.assert_allclose(masked_tensor, arr[mask])

  def testMaskDimensionsSetToNoneRaises(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.int32, shape=[None, 2])
      mask = array_ops.placeholder(dtypes.bool, shape=None)
      with self.assertRaisesRegexp(ValueError, "dimensions must be specified"):
        array_ops.boolean_mask(tensor, mask)

  def testMaskHasMoreDimsThanTensorRaises(self):
    mask = [[True, True], [False, False]]
    tensor = [1, 2, 3, 4]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskIsScalarRaises(self):
    mask = True
    tensor = 1
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "mask.*scalar"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskShapeDifferentThanFirstPartOfTensorShapeRaises(self):
    mask = [True, True, True]
    tensor = [[1, 2], [3, 4]]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()


class OperatorShapeTest(test_util.TensorFlowTestCase):

  def testExpandScalar(self):
    scalar = "hello"
    scalar_expanded = array_ops.expand_dims(scalar, [0])
    self.assertEqual(scalar_expanded.get_shape(), (1,))

  def testSqueeze(self):
    scalar = "hello"
    scalar_squeezed = array_ops.squeeze(scalar, ())
    self.assertEqual(scalar_squeezed.get_shape(), ())


class ReverseTest(test_util.TensorFlowTestCase):

  def testReverse0DimAuto(self):
    x_np = 4
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, []).eval()
        self.assertAllEqual(x_tf, x_np)

  def testReverse1DimAuto(self):
    x_np = [1, 4, 9]

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, [True]).eval()
        self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])

  def testUnknownDims(self):
    data_t = tf.placeholder(tf.float32)
    dims_known_t = tf.placeholder(tf.bool, shape=[3])
    reverse_known_t = tf.reverse(data_t, dims_known_t)
    self.assertEqual(3, reverse_known_t.get_shape().ndims)

    dims_unknown_t = tf.placeholder(tf.bool)
    reverse_unknown_t = tf.reverse(data_t, dims_unknown_t)
    self.assertIs(None, reverse_unknown_t.get_shape().ndims)

    data_2d_t = tf.placeholder(tf.float32, shape=[None, None])
    dims_2d_t = tf.placeholder(tf.bool, shape=[2])
    reverse_2d_t = tf.reverse(data_2d_t, dims_2d_t)
    self.assertEqual(2, reverse_2d_t.get_shape().ndims)

    dims_3d_t = tf.placeholder(tf.bool, shape=[3])
    with self.assertRaisesRegexp(ValueError, "must have rank 3"):
      tf.reverse(data_2d_t, dims_3d_t)


if __name__ == "__main__":
  googletest.main()
