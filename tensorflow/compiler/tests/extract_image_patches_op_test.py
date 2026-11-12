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
"""Functional tests for ExtractImagePatches op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ExtractImagePatches(xla_test.XLATestCase):
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

    with self.session():
      image_placeholder = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        out_tensor = array_ops.extract_image_patches(
            image_placeholder,
            ksizes=ksizes,
            strides=strides,
            rates=rates,
            padding=padding,
            name="im2col")
      feed_dict = {image_placeholder: image}
      self.assertAllClose(patches, out_tensor.eval(feed_dict=feed_dict))

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

  def testKsize3x3Stride2x2Rate1x1Same(self):
    """Test for 3x3 kernel with SAME padding on both sides."""
    # [1, 3, 3, 1]
    image = np.arange(1, 10).reshape(1, 3, 3, 1).astype(np.float32)
    # [1, 2, 2, 9]
    patches = [[
        [[0, 0, 0, 0, 1, 2, 0, 4, 5], [0, 0, 0, 2, 3, 0, 5, 6, 0]],
        [[0, 4, 5, 0, 7, 8, 0, 0, 0], [5, 6, 0, 8, 9, 0, 0, 0, 0]],
    ]]
    self._VerifyValues(
        image,
        ksizes=[3, 3],
        strides=[2, 2],
        rates=[1, 1],
        padding="SAME",
        patches=patches,
    )

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

  def testKsize2x2Stride1x1Rate1x1ValidDepth2(self):
    """Test for 2x2 kernel with VALID padding."""
    # [1, 2, 2, 2]
    image = [[[[1, 5], [2, 6]], [[3, 7], [4, 8]]]]
    # [1, 1, 1, 8]
    patches = [[[[1, 5, 2, 6, 3, 7, 4, 8]]]]
    self._VerifyValues(
        image,
        ksizes=[2, 2],
        strides=[1, 1],
        rates=[1, 1],
        padding="VALID",
        patches=patches)

  def testSamePaddingNaNDoesNotSpuriouslyPropagate(self):
    """Test that NaN values do not spuriously propagate under SAME padding."""
    nan = float("nan")
    image = np.array(
        [[
            [[0.0], [-0.0], [nan]],
            [[1.0], [-1.0], [2.0]],
            [[3.0], [4.0], [-5.0]],
        ]],
        dtype=np.float32,
    )
    expected_patches = np.array(
        [[
            [
                [0.0, -0.0, 1.0, -1.0],
                [-0.0, nan, -1.0, 2.0],
                [nan, 0.0, 2.0, 0.0],
            ],
            [
                [1.0, -1.0, 3.0, 4.0],
                [-1.0, 2.0, 4.0, -5.0],
                [2.0, 0.0, -5.0, 0.0],
            ],
            [
                [3.0, 4.0, 0.0, 0.0],
                [4.0, -5.0, 0.0, 0.0],
                [-5.0, 0.0, 0.0, 0.0],
            ],
        ]],
        dtype=np.float32,
    )
    with self.session():
      image_placeholder = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        out_tensor = array_ops.extract_image_patches(
            image_placeholder,
            ksizes=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
            name="im2col",
        )
      feed_dict = {image_placeholder: image}
      actual_patches = out_tensor.eval(feed_dict=feed_dict)
      self.assertAllEqual(expected_patches, actual_patches)

  def testIntegerTypesSupported(self):
    """Test that integer and bool dtypes are supported under XLA."""
    for dtype in self.int_types | {np.bool_}:
      image = np.array([[[[1], [2]], [[3], [4]]]], dtype=dtype)
      expected_patches = np.array(
          [[[[1, 2, 3, 4], [2, 0, 4, 0]], [[3, 4, 0, 0], [4, 0, 0, 0]]]],
          dtype=dtype,
      )
      with self.session():
        image_placeholder = array_ops.placeholder(dtypes.as_dtype(dtype))
        with self.test_scope():
          out_tensor = array_ops.extract_image_patches(
              image_placeholder,
              ksizes=[1, 2, 2, 1],
              strides=[1, 1, 1, 1],
              rates=[1, 1, 1, 1],
              padding="SAME",
          )
        feed_dict = {image_placeholder: image}
        self.assertAllEqual(
            expected_patches, out_tensor.eval(feed_dict=feed_dict)
        )

  def testEmptySpatialOutput(self):
    """Test that zero-sized spatial output dimensions succeed under XLA."""
    with self.session():
      for in_shape, ksizes, padding, expected_shape in [
          ((1, 1, 1, 2), [1, 3, 3, 1], "VALID", (1, 0, 0, 18)),
          ((1, 0, 2, 2), [1, 2, 2, 1], "SAME", (1, 0, 1, 8)),
      ]:
        image_placeholder = array_ops.placeholder(dtypes.float32)
        with self.test_scope():
          out_tensor = array_ops.extract_image_patches(
              image_placeholder,
              ksizes=ksizes,
              strides=[1, 2, 2, 1],
              rates=[1, 1, 1, 1],
              padding=padding,
          )
        feed_dict = {image_placeholder: np.zeros(in_shape, dtype=np.float32)}
        self.assertEqual(
            out_tensor.eval(feed_dict=feed_dict).shape, expected_shape
        )

  def testInvalidKernelSize(self):
    """Test that zero kernel size is rejected by shape inference."""
    with self.session():
      image_placeholder = array_ops.placeholder(dtypes.float32)
      with self.assertRaisesRegex(
          ValueError,
          "ExtractImagePatches requires spatial ksizes to be positive",
      ):
        array_ops.extract_image_patches(
            image_placeholder,
            ksizes=[1, 0, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )


if __name__ == "__main__":
  test.main()
