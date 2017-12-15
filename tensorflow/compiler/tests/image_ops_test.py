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
"""Tests for image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.platform import test


class ResizeBilinearTest(XLATestCase):

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None):
    if expected is None:
      self.fail("expected must be specified")
    with self.test_session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=True)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def _assertBackwardOpMatchesExpected(self,
                                       grads_np,
                                       input_shape=None,
                                       dtype=None,
                                       expected=None):
    if input_shape is None:
      self.fail("input_shape must be specified")
    if expected is None:
      self.fail("expected must be specified")
    with self.test_session() as sess, self.test_scope():
      dtype = dtype or np.float32
      grads = array_ops.placeholder(np.float32)
      resized = gen_image_ops._resize_bilinear_grad(
          grads,
          np.zeros([1, input_shape[0], input_shape[1], 1], dtype=dtype),
          align_corners=True)
      out = sess.run(resized, {grads: grads_np[np.newaxis, :, :, np.newaxis]})
      self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testAlignCorners1x2To3x2(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2]], dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2]], dtype=np.float32))

  def testAlignCorners1x2To3x2Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
          input_shape=[1, 2],
          dtype=dtype,
          expected=np.array([[9, 12]], dtype=np.float32))

  def testAlignCorners2x2To1x1(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2], [3, 4]], dtype=dtype), [1, 1],
          expected=np.array([[1]], dtype=np.float32))

  def testAlignCorners2x2To1x1Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[7]], dtype=np.float32),
          input_shape=[2, 2],
          dtype=dtype,
          expected=np.array([[7, 0], [0, 0]], dtype=np.float32))

  def testAlignCorners2x2To3x3(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2], [3, 4]], dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 1.5, 2], [2, 2.5, 3], [3, 3.5, 4]], dtype=np.float32))

  def testAlignCorners2x2To3x3Grad(self):
    self._assertBackwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        input_shape=[2, 2],
        expected=np.array([[5.25, 8.25], [14.25, 17.25]], dtype=np.float32))

  def testAlignCorners3x3To2x2(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype), [2, 2],
          expected=np.array([[1, 3], [7, 9]], dtype=np.float32))

  def testAlignCorners3x3To2x2Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[7, 13], [22, 4]], dtype=np.float32),
          input_shape=[3, 3],
          dtype=dtype,
          expected=np.array(
              [[7, 0, 13], [0, 0, 0], [22, 0, 4]], dtype=np.float32))

  def testAlignCorners4x4To3x3(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(
              [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
              dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 2.5, 4], [7, 8.5, 10], [13, 14.5, 16]], dtype=np.float32))

  def testAlignCorners4x4To3x3Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
          input_shape=[4, 4],
          dtype=dtype,
          expected=np.array(
              [[1, 1, 1, 3], [2, 1.25, 1.25, 3], [2, 1.25, 1.25, 3],
               [7, 4, 4, 9]],
              dtype=np.float32))


if __name__ == "__main__":
  test.main()
