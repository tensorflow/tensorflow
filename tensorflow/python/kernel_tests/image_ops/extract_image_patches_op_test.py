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
"""Functional tests for ExtractImagePatches op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ExtractImagePatches(test.TestCase):
  """Functional tests for ExtractImagePatches op."""

  def _VerifyValues(self, image, ksizes, strides, rates, padding, patches):
    """Tests input-output pairs for the ExtractImagePatches op.

    Args:
      image: Input tensor with shape: [batch, in_rows, in_cols, depth].
      ksizes: Patch size specified as: [ksize_rows, ksize_cols].
      strides: Output strides, specified as [stride_rows, stride_cols].
      rates: Atrous rates, specified as [rate_rows, rate_cols].
      padding: Padding type.
      patches: Expected output.
    """
    ksizes = [1] + ksizes + [1]
    strides = [1] + strides + [1]
    rates = [1] + rates + [1]

    for dtype in [
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ]:
      out_tensor = array_ops.extract_image_patches(
          constant_op.constant(image, dtype=dtype),
          ksizes=ksizes,
          strides=strides,
          rates=rates,
          padding=padding,
          name="im2col",
      )
      self.assertAllClose(
          np.array(patches, dtype=dtype), self.evaluate(out_tensor)
      )

  def testKsize1x1Stride1x1Rate1x1(self):
    """Verifies that for 1x1 kernel the output equals the input."""
    # [2, 3, 4, 5]
    image = np.reshape(range(120), [2, 3, 4, 5])
    # [2, 3, 4, 5]
    patches = np.reshape(range(120), [2, 3, 4, 5])
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[1, 1],
          rates=[1, 1],
          padding=padding,
          patches=patches)

  def testKsize1x1Stride2x3Rate1x1(self):
    """Test for 1x1 kernel and strides."""
    # [2, 4, 5, 3]
    image = np.reshape(range(120), [2, 4, 5, 3])
    # [2, 2, 2, 3]
    patches = image[:, ::2, ::3, :]
    for padding in ["VALID", "SAME"]:
      self._VerifyValues(
          image,
          ksizes=[1, 1],
          strides=[2, 3],
          rates=[1, 1],
          padding=padding,
          patches=patches)

  def testKsize2x2Stride1x1Rate1x1Valid(self):
    """Test for 2x2 kernel with VALID padding."""
    # [1, 2, 2, 1]
    image = [[[[1], [2]], [[3], [4]]]]
    # [1, 1, 1, 4]
    patches = [[[[1, 2, 3, 4]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        patches=patches)

  def testKsize2x2Stride1x1Rate1x1Same(self):
    """Test for 2x2 kernel with SAME padding."""
    # [1, 2, 2, 1]
    image = [[[[1], [2]], [[3], [4]]]]
    # [1, 2, 2, 4]
    patches = [[[[1, 2, 3, 4], [2, 0, 4, 0]], [[3, 4, 0, 0], [4, 0, 0, 0]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="SAME",
        patches=patches)

  def testKsize2x2Stride1x1Rate2x2Valid(self):
    """Test for 2x2 kernel with 2x2 dilation."""
    # [1, 2, 2, 1]
    image = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
    # [1, 2, 2, 4]
    patches = [[[[0, 2, 8, 10], [1, 3, 9, 11]],
                [[4, 6, 12, 14], [5, 7, 13, 15]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[2, 2],
        padding="VALID",
        patches=patches)

  def testComplexDataTypes(self):
    """Test for complex data types"""
    for dtype in [np.complex64, np.complex128]:
      image = (
          np.reshape(range(120), [2, 3, 4, 5]).astype(dtype) +
          np.reshape(range(120, 240), [2, 3, 4, 5]).astype(dtype) * 1j)
      patches = (
          np.reshape(range(120), [2, 3, 4, 5]).astype(dtype) +
          np.reshape(range(120, 240), [2, 3, 4, 5]).astype(dtype) * 1j)
      for padding in ["VALID", "SAME"]:
        self._VerifyValues(
            image,
            ksizes=[1, 1],
            strides=[1, 1],
            rates=[1, 1],
            padding=padding,
            patches=patches)

  def testInvalidAttributes(self):
    """Test for passing weird things into ksizes."""
    with self.assertRaisesRegex(TypeError, "Expected list"):
      image = constant_op.constant([0.0])
      ksizes = math_ops.cast(
          constant_op.constant(dtype=dtypes.int16, value=[[1, 4], [5, 2]]),
          dtype=dtypes.qint16)
      strides = [1, 1, 1, 1]
      self.evaluate(
          array_ops.extract_image_patches(
              image, ksizes=ksizes, strides=strides, padding="SAME"))

if __name__ == "__main__":
  test.main()
