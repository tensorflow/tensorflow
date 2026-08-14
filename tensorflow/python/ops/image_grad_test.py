# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Image Op Gradients."""

from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.framework import errors
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test

ResizeNearestNeighborOpTest = test_base.ResizeNearestNeighborOpTestBase
ResizeBilinearOpTest = test_base.ResizeBilinearOpTestBase
ResizeBicubicOpTest = test_base.ResizeBicubicOpTestBase
ScaleAndTranslateOpTest = test_base.ScaleAndTranslateOpTestBase
CropAndResizeOpTest = test_base.CropAndResizeOpTestBase


class CropAndResizeGradTest(test_base.CropAndResizeOpTestBase):

  def testCropAndResizeGradBoxesWithNaNOrNonFinite(self):
    """Test crop_and_resize_grad_boxes with non-finite boxes."""
    grads = np.ones((1, 2, 2, 1), dtype=np.float32)
    image = np.ones((1, 4, 4, 1), dtype=np.float32)
    boxes_nan = np.array([[0.0, np.nan, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      image_ops.crop_and_resize_grad_boxes(
          grads, image, boxes_nan, box_ind
      )

  def testCropAndResizeGradImageWithNaNOrNonFinite(self):
    """Test crop_and_resize_grad_image with non-finite boxes."""
    grads = np.ones((1, 2, 2, 1), dtype=np.float32)
    boxes_nan = np.array([[0.0, np.nan, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    image_dense_shape = np.array([1, 4, 4, 1], dtype=np.int32)

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      image_ops.crop_and_resize_grad_image(
          grads, boxes_nan, box_ind, image_dense_shape, T=np.float32
      )

RGBToHSVOpTest = test_base.RGBToHSVOpTestBase

if __name__ == "__main__":
  test.main()

class CropAndResizeGradNonFiniteTest(test_util.TensorFlowTestCase):

  def testCropAndResizeGradWithNaNOrNonFinite(self):
    grads = np.ones((1, 2, 2, 1), dtype=np.float32)
    image = np.ones((1, 4, 4, 1), dtype=np.float32)
    boxes_nan = np.array([[0.0, np.nan, 1.0, 1.0]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          image_ops.crop_and_resize_grad_boxes(
              grads, image, boxes_nan, box_ind
          )
      )

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          image_ops.crop_and_resize_grad_image(
              grads, boxes_nan, box_ind, image_shape=[1, 4, 4, 1]
          )
      )
