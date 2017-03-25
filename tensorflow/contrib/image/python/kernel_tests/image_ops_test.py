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
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

_DTYPES = set(
    [dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])


class ImageOpsTest(test_util.TensorFlowTestCase):

  def test_zeros(self):
    with self.test_session():
      for dtype in _DTYPES:
        for shape in [(5, 5), (24, 24), (2, 24, 24, 3)]:
          for angle in [0, 1, np.pi / 2.0]:
            image = array_ops.zeros(shape, dtype)
            self.assertAllEqual(
                image_ops.rotate(image, angle).eval(),
                np.zeros(shape, dtype.as_numpy_dtype()))

  def test_rotate_even(self):
    with self.test_session():
      for dtype in _DTYPES:
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
    with self.test_session():
      for dtype in _DTYPES:
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

  def test_compose(self):
    with self.test_session():
      for dtype in _DTYPES:
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


if __name__ == "__main__":
  googletest.main()
