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
"""Tests for image_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.image.python.ops import image_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

_DTYPES = set(
    [dtypes.uint8, dtypes.int32, dtypes.int64,
     dtypes.float16, dtypes.float32, dtypes.float64])


class ImageOpsTest(test_util.TensorFlowTestCase):

  def test_zeros(self):
    for dtype in _DTYPES:
      with self.test_session():
        for shape in [(5, 5), (24, 24), (2, 24, 24, 3)]:
          for angle in [0, 1, np.pi / 2.0]:
            image = array_ops.zeros(shape, dtype)
            self.assertAllEqual(
                image_ops.rotate(image, angle).eval(),
                np.zeros(shape, dtype.as_numpy_dtype()))

  def test_rotate_even(self):
    for dtype in _DTYPES:
      with self.test_session():
        image = array_ops.reshape(
            math_ops.cast(math_ops.range(36), dtype), (6, 6))
        image_rep = array_ops.tile(image[None, :, :, None], [3, 1, 1, 1])
        angles = constant_op.constant([0.0, np.pi / 4.0, np.pi / 2.0],
                                      dtypes.float32)
        image_rotated = image_ops.rotate(image_rep, angles)
        self.assertAllEqual(image_rotated[:, :, :, 0].eval(),
                            [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                              [12, 13, 14, 15, 16, 17],
                              [18, 19, 20, 21, 22, 23],
                              [24, 25, 26, 27, 28, 29],
                              [30, 31, 32, 33, 34, 35]],
                             [[0, 3, 4, 11, 17, 0], [2, 3, 9, 16, 23, 23],
                              [1, 8, 15, 21, 22, 29], [6, 13, 20, 21, 27, 34],
                              [12, 18, 19, 26, 33, 33], [0, 18, 24, 31, 32, 0]],
                             [[5, 11, 17, 23, 29, 35], [4, 10, 16, 22, 28, 34],
                              [3, 9, 15, 21, 27, 33], [2, 8, 14, 20, 26, 32],
                              [1, 7, 13, 19, 25, 31], [0, 6, 12, 18, 24, 30]]])

  def test_rotate_odd(self):
    for dtype in _DTYPES:
      with self.test_session():
        image = array_ops.reshape(
            math_ops.cast(math_ops.range(25), dtype), (5, 5))
        image_rep = array_ops.tile(image[None, :, :, None], [3, 1, 1, 1])
        angles = constant_op.constant([np.pi / 4.0, 1.0, -np.pi / 2.0],
                                      dtypes.float32)
        image_rotated = image_ops.rotate(image_rep, angles)
        self.assertAllEqual(image_rotated[:, :, :, 0].eval(),
                            [[[0, 3, 8, 9, 0], [1, 7, 8, 13, 19],
                              [6, 6, 12, 18, 18], [5, 11, 16, 17, 23],
                              [0, 15, 16, 21, 0]],
                             [[0, 3, 9, 14, 0], [2, 7, 8, 13, 19],
                              [1, 6, 12, 18, 23], [5, 11, 16, 17, 22],
                              [0, 10, 15, 21, 0]],
                             [[20, 15, 10, 5, 0], [21, 16, 11, 6, 1],
                              [22, 17, 12, 7, 2], [23, 18, 13, 8, 3],
                              [24, 19, 14, 9, 4]]])

  def test_translate(self):
    for dtype in _DTYPES:
      with self.test_session():
        image = constant_op.constant(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 0],
             [0, 1, 0, 1]], dtype=dtype)
        translation = constant_op.constant([-1, -1], dtypes.float32)
        image_translated = image_ops.translate(image, translation)
        self.assertAllEqual(image_translated.eval(),
                            [[1, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 1, 0],
                             [0, 0, 0, 0]])

  def test_compose(self):
    for dtype in _DTYPES:
      with self.test_session():
        image = constant_op.constant(
            [[1, 1, 1, 0],
             [1, 0, 0, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0]], dtype=dtype)
        # Rotate counter-clockwise by pi / 2.
        rotation = image_ops.angles_to_projective_transforms(np.pi / 2, 4, 4)
        # Translate right by 1 (the transformation matrix is always inverted,
        # hence the -1).
        translation = constant_op.constant([1, 0, -1,
                                            0, 1, 0,
                                            0, 0],
                                           dtype=dtypes.float32)
        composed = image_ops.compose_transforms(rotation, translation)
        image_transformed = image_ops.transform(image, composed)
        self.assertAllEqual(image_transformed.eval(),
                            [[0, 0, 0, 0],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [0, 1, 1, 1]])

  def test_bilinear(self):
    with self.test_session():
      image = constant_op.constant(
          [[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]],
          dtypes.float32)
      # The following result matches:
      # >>> scipy.ndimage.rotate(image, 45, order=1, reshape=False)
      # which uses spline interpolation of order 1, equivalent to bilinear
      # interpolation.
      self.assertAllClose(
          image_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR").eval(),
          [[0.000, 0.000, 0.343, 0.000, 0.000],
           [0.000, 0.586, 0.914, 0.586, 0.000],
           [0.343, 0.914, 0.000, 0.914, 0.343],
           [0.000, 0.586, 0.914, 0.586, 0.000],
           [0.000, 0.000, 0.343, 0.000, 0.000]],
          atol=0.001)
      self.assertAllClose(
          image_ops.rotate(image, np.pi / 4.0, interpolation="NEAREST").eval(),
          [[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 0, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])

  def test_bilinear_uint8(self):
    with self.test_session():
      image = constant_op.constant(
          np.asarray(
              [[0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 255, 255, 255, 0.0],
               [0.0, 255, 0.0, 255, 0.0],
               [0.0, 255, 255, 255, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0]],
              np.uint8),
          dtypes.uint8)
      # == np.rint((expected image above) * 255)
      self.assertAllEqual(
          image_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR").eval(),
          [[0.0, 0.0, 87., 0.0, 0.0],
           [0.0, 149, 233, 149, 0.0],
           [87., 233, 0.0, 233, 87.],
           [0.0, 149, 233, 149, 0.0],
           [0.0, 0.0, 87., 0.0, 0.0]])

  def _test_grad(self, shape_to_test):
    with self.test_session():
      test_image_shape = shape_to_test
      test_image = np.random.randn(*test_image_shape)
      test_image_tensor = constant_op.constant(
          test_image, shape=test_image_shape)
      test_transform = image_ops.angles_to_projective_transforms(
          np.pi / 2, 4, 4)

      output_shape = test_image_shape
      output = image_ops.transform(test_image_tensor, test_transform)
      left_err = gradient_checker.compute_gradient_error(
          test_image_tensor,
          test_image_shape,
          output,
          output_shape,
          x_init_value=test_image)
      self.assertLess(left_err, 1e-10)

  def test_grad(self):
    self._test_grad([16, 16])
    self._test_grad([4, 12, 12])
    self._test_grad([3, 4, 12, 12])


class BipartiteMatchTest(test_util.TensorFlowTestCase):

  def _BipartiteMatchTest(self, distance_mat, distance_mat_shape,
                          num_valid_rows,
                          expected_row_to_col_match,
                          expected_col_to_row_match):
    distance_mat_np = np.array(distance_mat, dtype=np.float32).reshape(
        distance_mat_shape)
    expected_row_to_col_match_np = np.array(expected_row_to_col_match,
                                            dtype=np.int32)
    expected_col_to_row_match_np = np.array(expected_col_to_row_match,
                                            dtype=np.int32)

    with self.test_session():
      distance_mat_tf = constant_op.constant(distance_mat_np,
                                             shape=distance_mat_shape)
      location_to_prior, prior_to_location = image_ops.bipartite_match(
          distance_mat_tf, num_valid_rows)
      location_to_prior_np = location_to_prior.eval()
      prior_to_location_np = prior_to_location.eval()
      self.assertAllEqual(location_to_prior_np, expected_row_to_col_match_np)
      self.assertAllEqual(prior_to_location_np, expected_col_to_row_match_np)

  def testBipartiteMatch(self):
    distance_mat = [0.5, 0.8, 0.1,
                    0.3, 0.2, 0.15]
    num_valid_rows = 2
    expected_row_to_col_match = [2, 1]
    expected_col_to_row_match = [-1, 1, 0]
    self._BipartiteMatchTest(distance_mat, [2, 3], num_valid_rows,
                             expected_row_to_col_match,
                             expected_col_to_row_match)

    # The case of num_valid_rows less than num-of-rows-in-distance-mat.
    num_valid_rows = 1
    expected_row_to_col_match = [2, -1]
    expected_col_to_row_match = [-1, -1, 0]
    self._BipartiteMatchTest(distance_mat, [2, 3], num_valid_rows,
                             expected_row_to_col_match,
                             expected_col_to_row_match)

    # The case of num_valid_rows being 0.
    num_valid_rows = 0
    expected_row_to_col_match = [-1, -1]
    expected_col_to_row_match = [-1, -1, -1]
    self._BipartiteMatchTest(distance_mat, [2, 3], num_valid_rows,
                             expected_row_to_col_match,
                             expected_col_to_row_match)

    # The case of num_valid_rows less being -1.
    num_valid_rows = -1
    # The expected results are the same as num_valid_rows being 2.
    expected_row_to_col_match = [2, 1]
    expected_col_to_row_match = [-1, 1, 0]
    self._BipartiteMatchTest(distance_mat, [2, 3], num_valid_rows,
                             expected_row_to_col_match,
                             expected_col_to_row_match)


if __name__ == "__main__":
  googletest.main()
