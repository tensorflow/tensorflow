# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.array_ops.repeat."""

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ImageOpsTest(parameterized.TestCase, xla_test.XLATestCase):

  def _assertProjectiveTransformMatchesEager(
      self,
      images,
      transforms,
      output_shape,
      interpolation,
      fill_mode,
      fill_value,
  ):
    def transform_fn(images, transforms):
      return gen_image_ops.image_projective_transform_v3(
          images,
          transforms,
          output_shape,
          fill_value,
          interpolation=interpolation,
          fill_mode=fill_mode,
      )

    compiled_transform = def_function.function(transform_fn, jit_compile=True)
    expected = transform_fn(images, transforms)
    actual = compiled_transform(images, transforms)
    self.assertAllClose(expected, actual)

  def testProjectiveTransformUsesPerImageCoordinates(self):
    with ops.device("device:{}:0".format(self.device)):
      images = array_ops.constant([[[[1.0], [2.0]]], [[[10.0], [20.0]]]])
      transforms = array_ops.constant([
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      ])
      self._assertProjectiveTransformMatchesEager(
          images,
          transforms,
          array_ops.constant([1, 2]),
          "NEAREST",
          "CONSTANT",
          array_ops.constant(0.0),
      )

  def testProjectiveTransformBilinearConstantUsesCornerFill(self):
    with ops.device("device:{}:0".format(self.device)):
      images = array_ops.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])
      transforms = array_ops.constant(
          [[1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]]
      )
      self._assertProjectiveTransformMatchesEager(
          images,
          transforms,
          array_ops.constant([2, 2]),
          "BILINEAR",
          "CONSTANT",
          array_ops.constant(-1.0),
      )

  @parameterized.named_parameters(
      ("bilinear", image_ops.ResizeMethod.BILINEAR),
      ("nearest", image_ops.ResizeMethod.NEAREST_NEIGHBOR))
  def testGradImageResize(self, resize_method):
    """Tests that the gradient of image.resize is compilable."""
    with ops.device("device:{}:0".format(self.device)):
      img_width = 2048
      var = variables.Variable(array_ops.ones(1, dtype=dtypes.float32))

      def model(x):
        x = var * x
        x = image_ops.resize_images(
            x,
            size=[img_width, img_width],
            method=resize_method)
        return x

      def train(x, y):
        with backprop.GradientTape() as tape:
          output = model(x)
          loss_value = math_ops.reduce_mean((y - output)**2)
        grads = tape.gradient(loss_value, [var])
        return grads

      compiled_train = def_function.function(train, jit_compile=True)
      rng = np.random.RandomState(57575)
      x = rng.uniform(size=(1, img_width // 2, img_width // 2, 1)).astype(
          np.float32)
      y = rng.uniform(size=(1, img_width, img_width, 1)).astype(np.float32)
      self.assertAllClose(
          train(x, y), compiled_train(x, y), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("default", False, False),
      ("align_corners", True, False),
      ("half_pixel_centers", False, True),
  )
  def testResizeNearestNeighborGrad(self, align_corners, half_pixel_centers):
    """Compares the compiled gradient with the TF kernel for each dtype."""
    # (grads shape, size): mixed up- and downsampling, a 2.5 scale where
    # align_corners rounds a tie, and a non-integer half-pixel scale.
    cases = [
        ((2, 7, 5, 3), [3, 12]),
        ((2, 3, 12, 3), [7, 5]),
        ((2, 3, 3, 3), [6, 6]),
        ((2, 3, 3, 3), [10, 10]),
    ]
    # Integer-valued inputs keep every sum exact, so the results must match.
    op_types = {
        np.uint8,
        np.int8,
        np.int32,
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    }
    rng = np.random.RandomState(57575)
    for dtype in sorted((self.float_types | self.int_types) & op_types,
                        key=lambda t: np.dtype(t).name):
      for grads_shape, size in cases:
        if (half_pixel_centers and size == [10, 10] and
            dtype in (np.float64, dtypes.bfloat16.as_numpy_dtype)):
          # The XLA resize kernels compute half-pixel nearest positions in the
          # input type and the TF kernels in float32. At this 10/3 scale they
          # pick different pixels for float64 and bfloat16, in
          # ResizeNearestNeighbor too.
          continue
        with self.subTest(dtype=np.dtype(dtype).name, grads_shape=grads_shape,
                          size=size):
          grads = rng.randint(1, 9, size=grads_shape).astype(dtype)

          def resize_grad(g, size=size):
            return gen_image_ops.resize_nearest_neighbor_grad(
                g,
                size=size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )

          compiled_resize_grad = def_function.function(
              resize_grad, jit_compile=True)
          with ops.device("device:{}:0".format(self.device)):
            expected = resize_grad(grads)
            actual = compiled_resize_grad(grads)
          self.assertAllEqual(expected, actual)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
