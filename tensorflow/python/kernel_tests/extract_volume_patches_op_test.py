# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for ExtractVolumePatches op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ExtractVolumePatches(test.TestCase):
  """Functional tests for ExtractVolumePatches op."""

  def _VerifyValues(self, image, ksizes, strides, padding, patches):
    """Tests input-output pairs for the ExtractVolumePatches op.

    Args:
      image: Input tensor with shape:
             [batch, in_planes, in_rows, in_cols, depth].
      ksizes: Patch size specified as: [ksize_planes, ksize_rows, ksize_cols].
      strides: Output strides, specified as:
               [stride_planes, stride_rows, stride_cols].
      padding: Padding type.
      patches: Expected output.

    Note:
      rates are not supported as of now.
    """
    ksizes = [1] + ksizes + [1]
    strides = [1] + strides + [1]

    with test_util.use_gpu():
      out_tensor = array_ops.extract_volume_patches(
          constant_op.constant(image),
          ksizes=ksizes,
          strides=strides,
          padding=padding,
          name="im2col_3d")
      self.assertAllClose(patches, self.evaluate(out_tensor))

  # pylint: disable=bad-whitespace
  def testKsize1x1x1Stride1x1x1(self):
    """Verifies that for 1x1x1 kernel the output equals the input."""
    image = np.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6]) + 1
    patches = image
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 1],
          strides=[1, 1, 1],
          padding=padding,
          patches=patches)

  def testKsize1x1x1Stride2x3x4(self):
    """Test for 1x1x1 kernel and strides."""
    image = np.arange(6 * 2 * 4 * 5 * 3).reshape([6, 2, 4, 5, 3]) + 1
    patches = image[:, ::2, ::3, ::4, :]
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 1],
          strides=[2, 3, 4],
          padding=padding,
          patches=patches)

  def testKsize1x1x2Stride2x2x3(self):
    """Test for 1x1x2 kernel and strides."""
    image = np.arange(45).reshape([1, 3, 3, 5, 1]) + 1
    patches = np.array([[[[[ 1,  2],
                           [ 4,  5]],
                          [[11, 12],
                           [14, 15]]],
                         [[[31, 32],
                           [34, 35]],
                          [[41, 42],
                           [44, 45]]]]])
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1, 2],
          strides=[2, 2, 3],
          padding=padding,
          patches=patches)

  def testKsize2x2x2Stride1x1x1Valid(self):
    """Test for 2x2x2 kernel with VALID padding."""
    image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
    patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8]]]]])
    self._VerifyValues(
        image,
        ksizes=[2, 2, 2],
        strides=[1, 1, 1],
        padding="VALID",
        patches=patches)

  def testKsize2x2x2Stride1x1x1Same(self):
    """Test for 2x2x2 kernel with SAME padding."""
    image = np.arange(8).reshape([1, 2, 2, 2, 1]) + 1
    patches = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8],
                           [2, 0, 4, 0, 6, 0, 8, 0]],
                          [[3, 4, 0, 0, 7, 8, 0, 0],
                           [4, 0, 0, 0, 8, 0, 0, 0]]],
                         [[[5, 6, 7, 8, 0, 0, 0, 0],
                           [6, 0, 8, 0, 0, 0, 0, 0]],
                          [[7, 8, 0, 0, 0, 0, 0, 0],
                           [8, 0, 0, 0, 0, 0, 0, 0]]]]])
    self._VerifyValues(
        image,
        ksizes=[2, 2, 2],
        strides=[1, 1, 1],
        padding="SAME",
        patches=patches)

if __name__ == "__main__":
  test.main()
