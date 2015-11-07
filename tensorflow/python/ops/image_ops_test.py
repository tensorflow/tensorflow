"""Tests for tensorflow.ops.image_ops."""
import math

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import googletest


class FlipTest(test_util.TensorFlowTestCase):

  def testIdempotentLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_left_right(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testIdempotentUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_up_down(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testIdempotentTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.transpose_image(image_ops.transpose_image(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8).reshape([3, 2, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.transpose_image(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)


class RandomFlipTest(test_util.TensorFlowTestCase):

  def testRandomLeftRight(self):
    x_np = np.array([0, 1], dtype=np.uint8).reshape([1, 2, 1])
    num_iterations = 500

    hist = [0, 0]
    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf)
      for _ in xrange(num_iterations):
        y_np = y.eval().flatten()[0]
        hist[y_np] += 1

    # Ensure that each entry is observed within 4 standard deviations.
    four_stddev = 4.0 * np.sqrt(num_iterations / 2.0)
    self.assertAllClose(hist, [num_iterations / 2.0] * 2, atol=four_stddev)

  def testRandomUpDown(self):
    x_np = np.array([0, 1], dtype=np.uint8).reshape([2, 1, 1])
    num_iterations = 500

    hist = [0, 0]
    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf)
      for _ in xrange(num_iterations):
        y_np = y.eval().flatten()[0]
        hist[y_np] += 1

    # Ensure that each entry is observed within 4 standard deviations.
    four_stddev = 4.0 * np.sqrt(num_iterations / 2.0)
    self.assertAllClose(hist, [num_iterations / 2.0] * 2, atol=four_stddev)


class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor, min_value, max_value):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        x = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.adjust_contrast(x,
                                      contrast_factor,
                                      min_value=min_value,
                                      max_value=max_value)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testDoubleContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 63, 169, 255, 29, 0, 255, 135, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np,
                       y_np,
                       contrast_factor=2.0,
                       min_value=None,
                       max_value=None)

  def testDoubleContrastFloat(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float).reshape(x_shape)

    y_data = [0, 0, 0, 62.75, 169.25, 255, 28.75, 0, 255, 134.75, 255, 0]
    y_np = np.array(y_data, dtype=np.float).reshape(x_shape)

    self._testContrast(x_np,
                       y_np,
                       contrast_factor=2.0,
                       min_value=0,
                       max_value=255)

  def testHalfContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [23, 53, 66, 50, 118, 172, 41, 54, 176, 68, 178, 60]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np,
                       y_np,
                       contrast_factor=0.5,
                       min_value=None,
                       max_value=None)

  def testBatchDoubleContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 11, 0, 255, 117, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np,
                       y_np,
                       contrast_factor=2.0,
                       min_value=None,
                       max_value=None)


class AdjustBrightnessTest(test_util.TensorFlowTestCase):

  def _testBrightness(self, x_np, y_np, delta, min_value, max_value):
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_brightness(x,
                                      delta,
                                      min_value=min_value,
                                      max_value=max_value)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testPositiveDeltaUint8(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 255, 11]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=10.0, min_value=None, max_value=None)

  def testPositiveDeltaFloat(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape)

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=10.0, min_value=None, max_value=None)

  def testNegativeDelta(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [5, 5, 5, 44, 125, 216, 27, 5, 224, 80, 245, 5]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=-10.0, min_value=5, max_value=None)


class RandomCropTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    # No random cropping is performed since the target width and height
    # are match the image dimensions.
    height = 4
    width = 5
    x_shape = [height, width, 3]
    x_np = np.arange(0, np.prod(x_shape), dtype=np.int32).reshape(x_shape)
    target_shape_np = np.array([height, width], dtype=np.int64)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      target_shape = constant_op.constant(target_shape_np, shape=[2])
      y = image_ops.random_crop(x, target_shape)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testRandomization(self):
    # Run 1x1 crop num_samples times in an image and ensure that one finds each
    # pixel 1/num_pixels of the time.
    num_samples = 1000
    height = 5
    width = 4

    num_pixels = height * width
    data = np.arange(num_pixels).reshape([height, width, 1])
    x_np = np.array(data).astype(np.int32)

    target_shape_np = np.array([1, 1], dtype=np.int64)

    y = []
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_np.shape)
      target_shape = constant_op.constant(target_shape_np, shape=[2])
      y_tf = image_ops.random_crop(x, target_shape)
      for _ in xrange(num_samples):
        y_np = y_tf.eval()
        self.assertAllEqual(y_np.shape, [1, 1, 1])
        y.extend(y_np.flatten())

    # Calculate the mean and 4 * standard deviation.
    mean = [num_samples / num_pixels] * num_pixels
    four_stddev = 4.0 * np.sqrt(mean)

    # Ensure that each entry is observed in 1/num_pixels of the samples
    # within 4 standard deviations.
    counts = np.bincount(y)
    self.assertAllClose(counts, mean, atol=four_stddev)


class PerImageWhiteningTest(test_util.TensorFlowTestCase):

  def _NumpyPerImageWhitening(self, x):
    num_pixels = np.prod(x.shape)
    x2 = np.square(x).astype(np.float32)
    mn = np.mean(x)
    vr = np.mean(x2) - (mn * mn)
    stddev = max(math.sqrt(vr), 1.0 / math.sqrt(num_pixels))

    y = x.astype(np.float32)
    y -= mn
    y /= stddev
    return y

  def testBasic(self):
    x_shape = [13, 9, 3]
    x_np = np.arange(0, np.prod(x_shape), dtype=np.int32).reshape(x_shape)
    y_np = self._NumpyPerImageWhitening(x_np)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.per_image_whitening(x)
      y_tf = y.eval()
      self.assertAllClose(y_tf, y_np, atol=1e-4)


class CropToBoundingBoxTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      target_height = x_shape[0]
      target_width = x_shape[1]
      y = image_ops.crop_to_bounding_box(x, 0, 0, target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testCropping(self):
    x_np = np.arange(0, 30, dtype=np.int32).reshape([6, 5, 1])

    offset_height = 1
    after_height = 2

    offset_width = 0
    after_width = 3

    target_height = x_np.shape[0] - offset_height - after_height
    target_width = x_np.shape[1] - offset_width - after_width

    y_np = x_np[offset_height:offset_height + target_height,
                offset_width:offset_width + target_width, :]

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.crop_to_bounding_box(x, offset_height, offset_width,
                                         target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf.flatten(), y_np.flatten())


class PadToBoundingBoxTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)

    target_height = x_shape[0]
    target_width = x_shape[1]

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.pad_to_bounding_box(x, 0, 0, target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testPadding(self):
    x_shape = [3, 4, 1]
    x_np = np.ones(x_shape, dtype=np.float32)

    offset_height = 2
    after_height = 3

    offset_width = 1
    after_width = 4

    target_height = x_shape[0] + offset_height + after_height
    target_width = x_shape[1] + offset_width + after_width

    # Note the padding are along batch, height, width and depth.
    paddings = ((offset_height, after_height),
                (offset_width, after_width),
                (0, 0))

    y_np = np.pad(x_np, paddings, 'constant')

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.pad_to_bounding_box(x, offset_height, offset_width,
                                        target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)


class ResizeImagesTest(test_util.TensorFlowTestCase):

  OPTIONS = [image_ops.ResizeMethod.BILINEAR,
             image_ops.ResizeMethod.NEAREST_NEIGHBOR,
             image_ops.ResizeMethod.BICUBIC,
             image_ops.ResizeMethod.AREA]

  def testNoOp(self):
    img_shape = [1, 6, 4, 1]
    data = [128, 128, 64, 64,
            128, 128, 64, 64,
            64, 64, 128, 128,
            64, 64, 128, 128,
            50, 50, 100, 100,
            50, 50, 100, 100]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 6
    target_width = 4

    for opt in self.OPTIONS:
      with self.test_session():
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, target_height, target_width, opt)
        resized = y.eval()
        self.assertAllClose(resized, img_np, atol=1e-5)

  def testResizeDown(self):

    data = [128, 128, 64, 64,
            128, 128, 64, 64,
            64, 64, 128, 128,
            64, 64, 128, 128,
            50, 50, 100, 100,
            50, 50, 100, 100]
    expected_data = [128, 64,
                     64, 128,
                     50, 100]
    target_height = 3
    target_width = 2

    # Test out 3-D and 4-D image shapes.
    img_shapes = [[1, 6, 4, 1], [6, 4, 1]]
    target_shapes = [[1, target_height, target_width, 1],
                     [target_height, target_width, 1]]

    for target_shape, img_shape in zip(target_shapes, img_shapes):
      img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

      for opt in self.OPTIONS:
        with self.test_session():
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(image, target_height, target_width, opt)
          expected = np.array(expected_data).reshape(target_shape)
          resized = y.eval()
          self.assertAllClose(resized, expected, atol=1e-5)

  def testResizeUp(self):
    img_shape = [1, 3, 2, 1]
    data = [128, 64,
            64, 128,
            50, 100]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        128.0, 96.0, 64.0, 64.0,
        96.0, 96.0, 96.0, 96.0,
        64.0, 96.0, 128.0, 128.0,
        57.0, 85.5, 114.0, 114.0,
        50.0, 75.0, 100.0, 100.0,
        50.0, 75.0, 100.0, 100.0]
    expected_data[image_ops.ResizeMethod.NEAREST_NEIGHBOR] = [
        128.0, 128.0, 64.0, 64.0,
        128.0, 128.0, 64.0, 64.0,
        64.0, 64.0, 128.0, 128.0,
        64.0, 64.0, 128.0, 128.0,
        50.0, 50.0, 100.0, 100.0,
        50.0, 50.0, 100.0, 100.0]
    expected_data[image_ops.ResizeMethod.AREA] = [
        128.0, 128.0, 64.0, 64.0,
        128.0, 128.0, 64.0, 64.0,
        64.0, 64.0, 128.0, 128.0,
        64.0, 64.0, 128.0, 128.0,
        50.0, 50.0, 100.0, 100.0,
        50.0, 50.0, 100.0, 100.0]

    for opt in [
        image_ops.ResizeMethod.BILINEAR,
        image_ops.ResizeMethod.NEAREST_NEIGHBOR,
        image_ops.ResizeMethod.AREA]:
      with self.test_session():
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, target_height, target_width, opt)
        resized = y.eval()
        expected = np.array(expected_data[opt]).reshape(
            [1, target_height, target_width, 1])
        self.assertAllClose(resized, expected, atol=1e-05)

  def testResizeUpBicubic(self):
    img_shape = [1, 6, 6, 1]
    data = [128, 128, 64, 64, 128, 128, 64, 64,
            64, 64, 128, 128, 64, 64, 128, 128,
            50, 50, 100, 100, 50, 50, 100, 100,
            50, 50, 100, 100, 50, 50, 100, 100,
            50, 50, 100, 100]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 8
    target_width = 8
    expected_data = [128, 135, 96, 55, 64, 114, 134, 128,
                     78, 81, 68, 52, 57, 118, 144, 136,
                     55, 49, 79, 109, 103, 89, 83, 84,
                     74, 70, 95, 122, 115, 69, 49, 55,
                     100, 105, 75, 43, 50, 89, 105, 100,
                     57, 54, 74, 96, 91, 65, 55, 58,
                     70, 69, 75, 81, 80, 72, 69, 70,
                     105, 112, 75, 36, 45, 92, 111, 105]

    with self.test_session():
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, target_height, target_width,
                                  image_ops.ResizeMethod.BICUBIC)
      resized = y.eval()
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      self.assertAllClose(resized, expected, atol=1)

  def testResizeDownArea(self):
    img_shape = [1, 6, 6, 1]
    data = [128, 64, 32, 16, 8, 4,
            4, 8, 16, 32, 64, 128,
            128, 64, 32, 16, 8, 4,
            5, 10, 15, 20, 25, 30,
            30, 25, 20, 15, 10, 5,
            5, 10, 15, 20, 25, 30]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 4
    target_width = 4
    expected_data = [73, 33, 23, 39,
                     73, 33, 23, 39,
                     14, 16, 19, 21,
                     14, 16, 19, 21]

    with self.test_session():
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, target_height, target_width,
                                  image_ops.ResizeMethod.AREA)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      resized = y.eval()
      self.assertAllClose(resized, expected, atol=1)


class ResizeImageWithCropOrPadTest(test_util.TensorFlowTestCase):

  def _ResizeImageWithCropOrPad(self, original, original_shape,
                                expected, expected_shape):
    x_np = np.array(original, dtype=np.uint8).reshape(original_shape)
    y_np = np.array(expected).reshape(expected_shape)

    target_height = expected_shape[0]
    target_width = expected_shape[1]

    with self.test_session():
      image = constant_op.constant(x_np, shape=original_shape)
      y = image_ops.resize_image_with_crop_or_pad(image,
                                                  target_height,
                                                  target_width)
      resized = y.eval()
      self.assertAllClose(resized, y_np, atol=1e-5)

  def testBasic(self):
    # Basic no-op.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   original, [2, 4, 1])

  def testPad(self):
    # Pad even along col.
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = [0, 1, 2, 3, 4, 0,
                0, 5, 6, 7, 8, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 6, 1])
    # Pad odd along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 1, 2, 3, 4, 0, 0,
                0, 5, 6, 7, 8, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 7, 1])

    # Pad even along row.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
                0, 0, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [4, 4, 1])
    # Pad odd along row.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
                0, 0, 0, 0,
                0, 0, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [5, 4, 1])

  def testCrop(self):
    # Crop even along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [2, 3,
                6, 7]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 2, 1])
    # Crop odd along col.

    original = [1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12]
    expected = [2, 3, 4,
                8, 9, 10]
    self._ResizeImageWithCropOrPad(original, [2, 6, 1],
                                   expected, [2, 3, 1])

    # Crop even along row.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8]
    expected = [3, 4,
                5, 6]
    self._ResizeImageWithCropOrPad(original, [4, 2, 1],
                                   expected, [2, 2, 1])

    # Crop odd along row.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12,
                13, 14,
                15, 16]
    expected = [3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12]
    self._ResizeImageWithCropOrPad(original, [8, 2, 1],
                                   expected, [5, 2, 1])

  def testCropAndPad(self):
    # Pad along row but crop along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0,
                2, 3,
                6, 7,
                0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [4, 2, 1])

    # Crop along row but pad along col.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8]
    expected = [0, 3, 4, 0,
                0, 5, 6, 0]
    self._ResizeImageWithCropOrPad(original, [4, 2, 1],
                                   expected, [2, 4, 1])


def _SimpleColorRamp():
  """Build a simple color ramp RGB image."""
  w, h = 256, 200
  i = np.arange(h)[:, None]
  j = np.arange(w)
  image = np.empty((h, w, 3), dtype=np.uint8)
  image[:, :, 0] = i
  image[:, :, 1] = j
  image[:, :, 2] = (i + j) >> 1
  return image


class JpegTest(test_util.TensorFlowTestCase):

  # TODO(irving): Add self.assertAverageLess or similar to test_util
  def averageError(self, image0, image1):
    self.assertEqual(image0.shape, image1.shape)
    image0 = image0.astype(int)  # Avoid overflow
    return np.abs(image0 - image1).sum() / float(np.prod(image0.shape))

  def testExisting(self):
    # Read a real jpeg and verify shape
    path = ('tensorflow/core/lib/jpeg/testdata/'
            'jpeg_merge_test1.jpg')
    with self.test_session() as sess:
      jpeg0 = io_ops.read_file(path)
      image0 = image_ops.decode_jpeg(jpeg0)
      image1 = image_ops.decode_jpeg(image_ops.encode_jpeg(image0))
      jpeg0, image0, image1 = sess.run([jpeg0, image0, image1])
      self.assertEqual(len(jpeg0), 3771)
      self.assertEqual(image0.shape, (256, 128, 3))
      self.assertLess(self.averageError(image0, image1), 0.8)

  def testSynthetic(self):
    with self.test_session() as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0)
      image2 = image_ops.decode_jpeg(image_ops.encode_jpeg(image1))
      jpeg0, image0, image1, image2 = sess.run([jpeg0, image0, image1, image2])

      # The decoded-encoded image should be similar to the input
      self.assertLess(self.averageError(image0, image1), 0.6)

      # We should be very close to a fixpoint
      self.assertLess(self.averageError(image1, image2), 0.02)

      # Smooth ramps compress well (input size is 153600)
      self.assertGreaterEqual(len(jpeg0), 5000)
      self.assertLessEqual(len(jpeg0), 6000)

  def testShape(self):
    with self.test_session() as sess:
      jpeg = constant_op.constant('nonsense')
      for channels in 0, 1, 3:
        image = image_ops.decode_jpeg(jpeg, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])


class PngTest(test_util.TensorFlowTestCase):

  def testExisting(self):
    # Read some real PNGs, converting to different channel numbers
    prefix = 'tensorflow/core/lib/png/testdata/'
    inputs = (1, 'lena_gray.png'), (4, 'lena_rgba.png')
    for channels_in, filename in inputs:
      for channels in 0, 1, 3, 4:
        with self.test_session() as sess:
          png0 = io_ops.read_file(prefix + filename)
          image0 = image_ops.decode_png(png0, channels=channels)
          png0, image0 = sess.run([png0, image0])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          if channels == channels_in:
            image1 = image_ops.decode_png(image_ops.encode_png(image0))
            self.assertAllEqual(image0, image1.eval())

  def testSynthetic(self):
    with self.test_session() as sess:
      # Encode it, then decode it
      image0 = constant_op.constant(_SimpleColorRamp())
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = sess.run([png0, image0, image1])

      # PNG is lossless
      self.assertAllEqual(image0, image1)

      # Smooth ramps compress well, but not too well
      self.assertGreaterEqual(len(png0), 400)
      self.assertLessEqual(len(png0), 750)

  def testShape(self):
    with self.test_session() as sess:
      png = constant_op.constant('nonsense')
      for channels in 0, 1, 3:
        image = image_ops.decode_png(png, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])


if __name__ == '__main__':
  googletest.main()
