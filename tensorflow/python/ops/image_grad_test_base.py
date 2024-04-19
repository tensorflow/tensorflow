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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.for_all_test_methods(test_util.disable_xla,
                                'align_corners=False not supported by XLA')
class ResizeNearestNeighborOpTestBase(test.TestCase):

  TYPES = [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 4).reshape(in_shape).astype(nptype)

      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_nearest_neighbor(input_tensor,
                                                     out_shape[1:3])
      with self.cached_session():
        self.assertEqual(out_shape, list(resize_out.get_shape()))
        resize_out = self.evaluate(resize_out)
      self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = (1, 4, 6, 1)

    for nptype in self.TYPES:
      x = np.arange(0, 6).reshape(in_shape).astype(nptype)

      def resize_nn(t, shape=out_shape):
        return image_ops.resize_nearest_neighbor(t, shape[1:3])

      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                resize_nn, [input_tensor], delta=1 / 8))
        self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = (1, 2, 3, 1)

    for nptype in self.TYPES:
      x = np.arange(0, 24).reshape(in_shape).astype(nptype)

      def resize_nn(t, shape=out_shape):
        return image_ops.resize_nearest_neighbor(t, shape[1:3])

      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                resize_nn, [input_tensor], delta=1 / 8))
        self.assertLess(err, 1e-3)

  def testCompareGpuVsCpu(self):
    in_shape = [1, 4, 6, 3]
    out_shape = (1, 8, 16, 3)

    for nptype in self.TYPES:
      x = np.arange(0, np.prod(in_shape)).reshape(in_shape).astype(nptype)
      for align_corners in [True, False]:

        def resize_nn(t, shape=out_shape, align_corners=align_corners):
          return image_ops.resize_nearest_neighbor(
              t, shape[1:3], align_corners=align_corners)

        with self.cached_session(use_gpu=False):
          input_tensor = constant_op.constant(x, shape=in_shape)
          grad_cpu = gradient_checker_v2.compute_gradient(
              resize_nn, [input_tensor], delta=1 / 8)

        with self.cached_session():
          input_tensor = constant_op.constant(x, shape=in_shape)
          grad_gpu = gradient_checker_v2.compute_gradient(
              resize_nn, [input_tensor], delta=1 / 8)

        self.assertAllClose(grad_cpu, grad_gpu, rtol=1e-5, atol=1e-5)


class ResizeBilinearOpTestBase(test.TestCase, parameterized.TestCase):

  def _itGen(self, smaller_shape, larger_shape):
    up_sample = (smaller_shape, larger_shape)
    down_sample = (larger_shape, smaller_shape)
    pass_through = (larger_shape, larger_shape)
    shape_pairs = (up_sample, down_sample, pass_through)
    # Align corners is deprecated in TF2.0, but align_corners==False is not
    # supported by XLA.
    options = [(True, False)]
    if not test_util.is_xla_enabled():
      options += [(False, True), (False, False)]
    for align_corners, half_pixel_centers in options:
      for in_shape, out_shape in shape_pairs:
        yield in_shape, out_shape, align_corners, half_pixel_centers

  def _getJacobians(self,
                    in_shape,
                    out_shape,
                    align_corners=False,
                    half_pixel_centers=False,
                    dtype=np.float32,
                    use_gpu=False,
                    force_gpu=False):
    with self.cached_session(use_gpu=use_gpu, force_gpu=force_gpu):
      # Input values should not influence gradients
      x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
      input_tensor = constant_op.constant(x, shape=in_shape)

      def func(in_tensor):
        return image_ops.resize_bilinear(
            in_tensor,
            out_shape[1:3],
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers)

      return gradient_checker_v2.compute_gradient(func, [input_tensor])

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def _testShapesParameterized(self, use_tape):

    TEST_CASES = [[1, 1], [2, 3], [5, 4]]  # pylint: disable=invalid-name

    for batch_size, channel_count in TEST_CASES:
      smaller_shape = [batch_size, 2, 3, channel_count]
      larger_shape = [batch_size, 4, 6, channel_count]
      for in_shape, out_shape, _, _ in self._itGen(smaller_shape, larger_shape):
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
          # Input values should not influence shapes
          x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(np.float32)
          input_tensor = constant_op.constant(x, shape=in_shape)
          tape.watch(input_tensor)
          resized_tensor = image_ops.resize_bilinear(input_tensor,
                                                     out_shape[1:3])
          self.assertEqual(out_shape, list(resized_tensor.get_shape()))

        grad_tensor = tape.gradient(resized_tensor, input_tensor)
        self.assertEqual(in_shape, list(grad_tensor.get_shape()))
        with self.cached_session():
          resized_values = self.evaluate(resized_tensor)
          self.assertEqual(out_shape, list(resized_values.shape))
          grad_values = self.evaluate(grad_tensor)
          self.assertEqual(in_shape, list(grad_values.shape))

  @parameterized.parameters({
      'batch_size': 1,
      'channel_count': 1
  }, {
      'batch_size': 4,
      'channel_count': 3
  }, {
      'batch_size': 3,
      'channel_count': 2
  })
  def testGradients(self, batch_size, channel_count):
    smaller_shape = [batch_size, 2, 3, channel_count]
    larger_shape = [batch_size, 5, 6, channel_count]
    for in_shape, out_shape, align_corners, half_pixel_centers in \
        self._itGen(smaller_shape, larger_shape):
      jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, align_corners,
                                            half_pixel_centers)
      threshold = 5e-3
      self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

  def testTypes(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]
    for use_gpu in [False, True]:
      for dtype in [
          np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
      ]:
        jacob_a, jacob_n = self._getJacobians(
            in_shape, out_shape, dtype=dtype, use_gpu=use_gpu)
        if dtype in (np.float16, dtypes.bfloat16.as_numpy_dtype):
          # Compare fp16/bf16 analytical gradients to fp32 numerical gradients,
          # since fp16/bf16 numerical gradients are too imprecise unless great
          # care is taken with choosing the inputs and the delta. This is
          # a weaker, but pragmatic, check (in particular, it does not test
          # the op itself, only its gradient).
          _, jacob_n = self._getJacobians(
              in_shape, out_shape, dtype=np.float32, use_gpu=use_gpu)
        threshold = 1e-3
        if dtype == np.float64:
          threshold = 1e-5
        self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradOnUnsupportedType(self, use_tape):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
      input_tensor = constant_op.constant(x, shape=in_shape)
      tape.watch(input_tensor)
      resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
      with self.cached_session():
        grad = tape.gradient(resize_out, [input_tensor])
    self.assertEqual([None], grad)

  def _gpuVsCpuCase(self, in_shape, out_shape, align_corners,
                    half_pixel_centers, dtype):
    grad = {}
    for use_gpu in [False, True]:
      grad[use_gpu] = self._getJacobians(
          in_shape,
          out_shape,
          align_corners,
          half_pixel_centers,
          dtype=dtype,
          use_gpu=use_gpu)
    threshold = 1e-4
    # Note that this is comparing both analytical and numerical Jacobians
    self.assertAllClose(grad[False], grad[True], rtol=threshold, atol=threshold)

  @parameterized.parameters({
      'batch_size': 1,
      'channel_count': 1
  }, {
      'batch_size': 2,
      'channel_count': 3
  }, {
      'batch_size': 5,
      'channel_count': 4
  })
  def testCompareGpuVsCpu(self, batch_size, channel_count):
    smaller_shape = [batch_size, 4, 6, channel_count]
    larger_shape = [batch_size, 8, 16, channel_count]
    for params in self._itGen(smaller_shape, larger_shape):
      self._gpuVsCpuCase(*params, dtype=np.float32)

  def testCompareGpuVsCpuFloat64(self):
    in_shape = [1, 5, 7, 1]
    out_shape = [1, 9, 11, 1]
    # Note that there is no 16-bit floating-point format registered for GPU
    self._gpuVsCpuCase(
        in_shape,
        out_shape,
        align_corners=True,
        half_pixel_centers=False,
        dtype=np.float64)


class ResizeBicubicOpTestBase(test.TestCase, parameterized.TestCase):
  """Tests resize bicubic ops."""

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 4).reshape(in_shape).astype(np.float32)

    for align_corners in [True, False]:
      input_tensor = constant_op.constant(x, shape=in_shape)
      resize_out = image_ops.resize_bicubic(
          input_tensor, out_shape[1:3], align_corners=align_corners)
      with self.cached_session():
        self.assertEqual(out_shape, list(resize_out.get_shape()))
        resize_out = self.evaluate(resize_out)
        self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)
    input_tensor = constant_op.constant(x, shape=in_shape)

    for align_corners in [True, False]:

      def func(input_tensor, align_corners=align_corners):
        return image_ops.resize_bicubic(
            input_tensor, out_shape[1:3], align_corners=align_corners)

      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [input_tensor]))

      self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.float32)
    input_tensor = constant_op.constant(x, shape=in_shape)

    for align_corners in [True, False]:

      def func(input_tensor, align_corners=align_corners):
        return image_ops.resize_bicubic(
            input_tensor, out_shape[1:3], align_corners=align_corners)

      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [input_tensor]))

      self.assertLess(err, 1e-3)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradOnUnsupportedType(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      in_shape = [1, 4, 6, 1]
      out_shape = [1, 2, 3, 1]

      x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
      input_tensor = constant_op.constant(x, shape=in_shape)
      tape.watch(input_tensor)

      resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3])
      with self.cached_session():
        grad = tape.gradient(resize_out, [input_tensor])
    self.assertEqual([None], grad)


class ScaleAndTranslateOpTestBase(test.TestCase):
  """Tests scale and translate op."""

  def testGrads(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    kernel_types = [
        'lanczos1', 'lanczos3', 'lanczos5', 'gaussian', 'box', 'triangle',
        'keyscubic', 'mitchellcubic'
    ]
    scales = [(1.0, 1.0), (0.37, 0.47), (2.1, 2.1)]
    translations = [(0.0, 0.0), (3.14, 1.19), (2.1, 3.1), (100.0, 200.0)]
    for scale in scales:
      for translation in translations:
        for kernel_type in kernel_types:
          for antialias in [True, False]:
            with self.cached_session():
              input_tensor = constant_op.constant(x, shape=in_shape)

              def scale_trans(input_tensor,
                              scale=scale,
                              translation=translation,
                              kernel_type=kernel_type,
                              antialias=antialias):
                # pylint: disable=cell-var-from-loop
                return image_ops.scale_and_translate(
                    input_tensor,
                    out_shape[1:3],
                    scale=constant_op.constant(scale),
                    translation=constant_op.constant(translation),
                    kernel_type=kernel_type,
                    antialias=antialias)

              err = gradient_checker_v2.max_error(
                  *gradient_checker_v2.compute_gradient(scale_trans,
                                                        [input_tensor]))

            self.assertLess(err, 1e-3)

  def testIdentityGrads(self):
    """Tests that Gradients for 1.0 scale should be ones for some kernels."""
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    kernel_types = ['lanczos1', 'lanczos3', 'lanczos5', 'triangle', 'keyscubic']
    scale = (1.0, 1.0)
    translation = (0.0, 0.0)
    antialias = True
    for kernel_type in kernel_types:
      with self.cached_session():
        input_tensor = constant_op.constant(x, shape=in_shape)
        with backprop.GradientTape() as tape:
          tape.watch(input_tensor)
          scale_and_translate_out = image_ops.scale_and_translate(
              input_tensor,
              out_shape[1:3],
              scale=constant_op.constant(scale),
              translation=constant_op.constant(translation),
              kernel_type=kernel_type,
              antialias=antialias)
        grad = tape.gradient(scale_and_translate_out, input_tensor)[0]
        grad_v = self.evaluate(grad)
        self.assertAllClose(np.ones_like(grad_v), grad_v)


class CropAndResizeOpTestBase(test.TestCase):

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

    crops = image_ops.crop_and_resize(
        constant_op.constant(image, shape=image_shape),
        constant_op.constant(boxes, shape=[num_boxes, 4]),
        constant_op.constant(box_ind, shape=[num_boxes]),
        constant_op.constant(crop_size, shape=[2]))
    with self.session():
      self.assertEqual(crops_shape, list(crops.get_shape()))
      crops = self.evaluate(crops)
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

              image_tensor = constant_op.constant(image, shape=image_shape)
              boxes_tensor = constant_op.constant(boxes, shape=[num_boxes, 4])
              box_ind_tensor = constant_op.constant(box_ind, shape=[num_boxes])

              def crop_resize(image_tensor, boxes_tensor):
                # pylint: disable=cell-var-from-loop
                return image_ops.crop_and_resize(
                    image_tensor, boxes_tensor, box_ind_tensor,
                    constant_op.constant(crop_size, shape=[2]))

              with test_util.device(use_gpu=True):
                with self.cached_session():
                  # pylint: disable=cell-var-from-loop
                  if (config.is_op_determinism_enabled() and
                      test_util.is_gpu_available()):
                    with self.assertRaises(errors_impl.UnimplementedError):
                      gradient_checker_v2.compute_gradient(
                          lambda x: crop_resize(x, boxes_tensor),
                          [image_tensor])
                    with self.assertRaises(errors_impl.UnimplementedError):
                      gradient_checker_v2.compute_gradient(
                          lambda x: crop_resize(image_tensor, x),
                          [boxes_tensor])
                  else:
                    err1 = gradient_checker_v2.max_error(
                        *gradient_checker_v2.compute_gradient(
                            lambda x: crop_resize(x, boxes_tensor),
                            [image_tensor]))
                    err2 = gradient_checker_v2.max_error(
                        *gradient_checker_v2.compute_gradient(
                            lambda x: crop_resize(image_tensor, x),
                            [boxes_tensor]))
                    err = max(err1, err2)
                    self.assertLess(err, 2e-3)


@test_util.run_all_in_graph_and_eager_modes
class RGBToHSVOpTestBase(test.TestCase):

  TYPES = [np.float32, np.float64]

  def testShapeIsCorrectAfterOp(self):
    in_shape = [2, 20, 30, 3]
    out_shape = [2, 20, 30, 3]

    for nptype in self.TYPES:
      x = np.random.randint(0, high=255, size=[2, 20, 30, 3]).astype(nptype)
      rgb_input_tensor = constant_op.constant(x, shape=in_shape)
      hsv_out = gen_image_ops.rgb_to_hsv(rgb_input_tensor)
      with self.cached_session():
        self.assertEqual(out_shape, list(hsv_out.get_shape()))
      hsv_out = self.evaluate(hsv_out)
      self.assertEqual(out_shape, list(hsv_out.shape))

  def testRGBToHSVGradSimpleCase(self):

    def f(x):
      return gen_image_ops.rgb_to_hsv(x)

    for nptype in self.TYPES:
      # Building a simple input tensor to avoid any discontinuity
      x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8,
                                                       0.9]]).astype(nptype)
      rgb_input_tensor = constant_op.constant(x, shape=x.shape)
      # Computing Analytical and Numerical gradients of f(x)
      analytical, numerical = gradient_checker_v2.compute_gradient(
          f, [rgb_input_tensor])
      self.assertAllClose(numerical, analytical, atol=1e-4)

  def testRGBToHSVGradRandomCase(self):

    def f(x):
      return gen_image_ops.rgb_to_hsv(x)

    np.random.seed(0)
    # Building a simple input tensor to avoid any discontinuity
    x = np.random.rand(1, 5, 5, 3).astype(np.float32)
    rgb_input_tensor = constant_op.constant(x, shape=x.shape)
    # Computing Analytical and Numerical gradients of f(x)
    self.assertLess(
        gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(f, [rgb_input_tensor])), 1e-4)

  def testRGBToHSVGradSpecialCaseRGreatest(self):
    # This test tests a specific subset of the input space
    # with a dummy function implemented with native TF operations.
    in_shape = [2, 10, 20, 3]

    def f(x):
      return gen_image_ops.rgb_to_hsv(x)

    def f_dummy(x):
      # This dummy function is a implementation of RGB to HSV using
      # primitive TF functions for one particular case when R>G>B.
      r = x[..., 0]
      g = x[..., 1]
      b = x[..., 2]
      # Since MAX = r and MIN = b, we get the following h,s,v values.
      v = r
      s = 1 - math_ops.div_no_nan(b, r)
      h = 60 * math_ops.div_no_nan(g - b, r - b)
      h = h / 360
      return array_ops_stack.stack([h, s, v], axis=-1)

    # Building a custom input tensor where R>G>B
    x_reds = np.ones((in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_greens = 0.5 * np.ones(
        (in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_blues = 0.2 * np.ones(
        (in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x = np.stack([x_reds, x_greens, x_blues], axis=-1)
    rgb_input_tensor = constant_op.constant(x, shape=in_shape)

    # Computing Analytical and Numerical gradients of f(x)
    analytical, numerical = gradient_checker_v2.compute_gradient(
        f, [rgb_input_tensor])
    # Computing Analytical and Numerical gradients of f_dummy(x)
    analytical_dummy, numerical_dummy = gradient_checker_v2.compute_gradient(
        f_dummy, [rgb_input_tensor])
    self.assertAllClose(numerical, analytical, atol=1e-4)
    self.assertAllClose(analytical_dummy, analytical, atol=1e-4)
    self.assertAllClose(numerical_dummy, numerical, atol=1e-4)


if __name__ == '__main__':
  test.main()
