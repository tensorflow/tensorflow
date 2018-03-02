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
"""Tests for Python ops defined in image_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ResizeNearestNeighborOpTest(test.TestCase):

  TYPES = [np.float32, np.float64]

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 4).reshape(in_shape).astype(nptype)

      with self.test_session(use_gpu=True) as sess:
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_nearest_neighbor(input_tensor,
                                                       out_shape[1:3])
        self.assertEqual(out_shape, list(resize_out.get_shape()))

        resize_out = sess.run(resize_out)
      self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 6).reshape(in_shape).astype(nptype)

      with self.test_session(use_gpu=True):
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_nearest_neighbor(input_tensor,
                                                       out_shape[1:3])
        err = gradient_checker.compute_gradient_error(
            input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
      self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 24).reshape(in_shape).astype(nptype)

      with self.test_session(use_gpu=True):
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_nearest_neighbor(input_tensor,
                                                       out_shape[1:3])
        err = gradient_checker.compute_gradient_error(
            input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
      self.assertLess(err, 1e-3)

  def testCompareGpuVsCpu(self):
    in_shape = [1, 4, 6, 3]
    out_shape = [1, 8, 16, 3]

    for nptype in self.TYPES:
      x = np.arange(0, np.prod(in_shape)).reshape(in_shape).astype(nptype)
      for align_corners in [True, False]:
        with self.test_session(use_gpu=False):
          input_tensor = constant_op.constant(x, shape=in_shape)
          resize_out = image_ops.resize_nearest_neighbor(
              input_tensor, out_shape[1:3], align_corners=align_corners)
          grad_cpu = gradient_checker.compute_gradient(
              input_tensor, in_shape, resize_out, out_shape, x_init_value=x)

        with self.test_session(use_gpu=True):
          input_tensor = constant_op.constant(x, shape=in_shape)
          resize_out = image_ops.resize_nearest_neighbor(
              input_tensor, out_shape[1:3], align_corners=align_corners)
          grad_gpu = gradient_checker.compute_gradient(
              input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
        self.assertAllClose(grad_cpu, grad_gpu, rtol=1e-5, atol=1e-5)


class ResizeBilinearOpTest(test.TestCase):

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 4).reshape(in_shape).astype(np.float32)

    with self.test_session() as sess:
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
      self.assertEqual(out_shape, list(resize_out.get_shape()))

      resize_out = sess.run(resize_out)
      self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    with self.test_session():
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
      err = gradient_checker.compute_gradient_error(
          input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
    self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.float32)

    with self.test_session():
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
      err = gradient_checker.compute_gradient_error(
          input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
    self.assertLess(err, 1e-3)

  def testCompareGpuVsCpu(self):
    in_shape = [2, 4, 6, 3]
    out_shape = [2, 8, 16, 3]

    size = np.prod(in_shape)
    x = 1.0 / size * np.arange(0, size).reshape(in_shape).astype(np.float32)
    for align_corners in [True, False]:
      grad = {}
      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu):
          input_tensor = constant_op.constant(x, shape=in_shape)
          resized_tensor = image_ops.resize_bilinear(
              input_tensor, out_shape[1:3], align_corners=align_corners)
          grad[use_gpu] = gradient_checker.compute_gradient(
              input_tensor, in_shape, resized_tensor, out_shape, x_init_value=x)

      self.assertAllClose(grad[False], grad[True], rtol=1e-4, atol=1e-4)

  def testTypes(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]
    x = np.arange(0, 24).reshape(in_shape)

    with self.test_session() as sess:
      for dtype in [np.float16, np.float32, np.float64]:
        input_tensor = constant_op.constant(x.astype(dtype), shape=in_shape)
        resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
        grad = sess.run(gradients_impl.gradients(resize_out, input_tensor))[0]
        self.assertAllEqual(in_shape, grad.shape)
        # Not using gradient_checker.compute_gradient as I didn't work out
        # the changes required to compensate for the lower precision of
        # float16 when computing the numeric jacobian.
        # Instead, we just test the theoretical jacobian.
        self.assertAllEqual([[[[1.], [0.], [1.], [0.], [1.], [0.]], [[0.], [
            0.
        ], [0.], [0.], [0.], [0.]], [[1.], [0.], [1.], [0.], [1.], [0.]],
                              [[0.], [0.], [0.], [0.], [0.], [0.]]]], grad)


class ResizeBicubicOpTest(test.TestCase):

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 4).reshape(in_shape).astype(np.float32)

    for align_corners in [True, False]:
      with self.test_session() as sess:
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3],
                                              align_corners=align_corners)
        self.assertEqual(out_shape, list(resize_out.get_shape()))

        resize_out = sess.run(resize_out)
        self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    for align_corners in [True, False]:
      with self.test_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3],
                                              align_corners=align_corners)
        err = gradient_checker.compute_gradient_error(
            input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
      self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.float32)

    for align_corners in [True, False]:
      with self.test_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3],
                                              align_corners=align_corners)
        err = gradient_checker.compute_gradient_error(
            input_tensor, in_shape, resize_out, out_shape, x_init_value=x)
      self.assertLess(err, 1e-3)

  def testGradOnUnsupportedType(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)

    with self.test_session():
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3])
      grad = gradients_impl.gradients(input_tensor, [resize_out])
      self.assertEqual([None], grad)


class CropAndResizeOpTest(test.TestCase):

  def testShapeIsCorrectAfterOp(self):
    batch = 2
    image_height = 3
    image_width = 4
    crop_height = 4
    crop_width = 5
    depth = 2
    num_boxes = 2

    image_shape = [batch, image_height, image_width, depth]
    crop_size = [crop_height, crop_width]
    crops_shape = [num_boxes, crop_height, crop_width, depth]

    image = np.arange(0, batch * image_height * image_width *
                      depth).reshape(image_shape).astype(np.float32)
    boxes = np.array([[0, 0, 1, 1], [.1, .2, .7, .8]], dtype=np.float32)
    box_ind = np.array([0, 1], dtype=np.int32)

    with self.test_session(use_gpu=True) as sess:
      crops = image_ops.crop_and_resize(
          constant_op.constant(
              image, shape=image_shape),
          constant_op.constant(
              boxes, shape=[num_boxes, 4]),
          constant_op.constant(
              box_ind, shape=[num_boxes]),
          constant_op.constant(
              crop_size, shape=[2]))
      self.assertEqual(crops_shape, list(crops.get_shape()))
      crops = sess.run(crops)
      self.assertEqual(crops_shape, list(crops.shape))

  def _randomUniformAvoidAnchors(self, low, high, anchors, radius, num_samples):
    """Generate samples that are far enough from a set of anchor points.

    We generate uniform samples in [low, high], then reject those that are less
    than radius away from any point in anchors. We stop after we have accepted
    num_samples samples.

    Args:
      low: The lower end of the interval.
      high: The upper end of the interval.
      anchors: A list of length num_crops with anchor points to avoid.
      radius: Distance threshold for the samples from the anchors.
      num_samples: How many samples to produce.

    Returns:
      samples: A list of length num_samples with the accepted samples.
    """
    self.assertTrue(low < high)
    self.assertTrue(radius >= 0)
    num_anchors = len(anchors)
    # Make sure that at least half of the interval is not forbidden.
    self.assertTrue(2 * radius * num_anchors < 0.5 * (high - low))
    anchors = np.reshape(anchors, num_anchors)
    samples = []
    while len(samples) < num_samples:
      sample = np.random.uniform(low, high)
      if np.all(np.fabs(sample - anchors) > radius):
        samples.append(sample)
    return samples

  def testGradRandomBoxes(self):
    """Test that the gradient is correct for randomly generated boxes.

    The mapping is piecewise differentiable with respect to the box coordinates.
    The points where the function is not differentiable are those which are
    mapped to image pixels, i.e., the normalized y coordinates in
    np.linspace(0, 1, image_height) and normalized x coordinates in
    np.linspace(0, 1, image_width). Make sure that the box coordinates are
    sufficiently far away from those rectangular grid centers that are points of
    discontinuity, so that the finite difference Jacobian is close to the
    computed one.
    """
    np.random.seed(1)  # Make it reproducible.
    delta = 1e-3
    radius = 2 * delta
    low, high = -0.5, 1.5  # Also covers the case of extrapolation.

    image_height = 4
    for image_width in range(1, 3):
      for crop_height in range(1, 3):
        for crop_width in range(2, 4):
          for depth in range(1, 3):
            for num_boxes in range(1, 3):

              batch = num_boxes
              image_shape = [batch, image_height, image_width, depth]
              crop_size = [crop_height, crop_width]
              crops_shape = [num_boxes, crop_height, crop_width, depth]
              boxes_shape = [num_boxes, 4]

              image = np.arange(0, batch * image_height * image_width *
                                depth).reshape(image_shape).astype(np.float32)
              boxes = []
              for _ in range(num_boxes):
                # pylint: disable=unbalanced-tuple-unpacking
                y1, y2 = self._randomUniformAvoidAnchors(
                    low, high, np.linspace(0, 1, image_height), radius, 2)
                x1, x2 = self._randomUniformAvoidAnchors(
                    low, high, np.linspace(0, 1, image_width), radius, 2)
                # pylint: enable=unbalanced-tuple-unpacking
                boxes.append([y1, x1, y2, x2])

              boxes = np.array(boxes, dtype=np.float32)
              box_ind = np.arange(batch, dtype=np.int32)

              with self.test_session(use_gpu=True):
                image_tensor = constant_op.constant(image, shape=image_shape)
                boxes_tensor = constant_op.constant(boxes, shape=[num_boxes, 4])
                box_ind_tensor = constant_op.constant(
                    box_ind, shape=[num_boxes])
                crops = image_ops.crop_and_resize(
                    image_tensor,
                    boxes_tensor,
                    box_ind_tensor,
                    constant_op.constant(
                        crop_size, shape=[2]))

                err = gradient_checker.compute_gradient_error(
                    [image_tensor, boxes_tensor], [image_shape, boxes_shape],
                    crops,
                    crops_shape,
                    delta=delta,
                    x_init_value=[image, boxes])

              self.assertLess(err, 2e-3)


if __name__ == "__main__":
  test.main()
