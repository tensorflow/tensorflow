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

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ImageOpsTest(xla_test.XLATestCase):

  def testGradImageResize(self):
    """Tests that the gradient of image.resize is compilable."""
    with ops.device("device:{}:0".format(self.device)):
      img_width = 2048
      var = variables.Variable(array_ops.ones(1, dtype=dtypes.float32))

      def model(x):
        x = var * x
        x = image_ops.resize_images(
            x,
            size=[img_width, img_width],
            method=image_ops.ResizeMethod.BILINEAR)
        return x

      def train(x, y):
        with backprop.GradientTape() as tape:
          output = model(x)
          loss_value = math_ops.reduce_mean((y - output)**2)
        grads = tape.gradient(loss_value, [var])
        return grads

      compiled_train = def_function.function(train, jit_compile=True)
      x = array_ops.zeros((1, img_width // 2, img_width // 2, 1),
                          dtype=dtypes.float32)
      y = array_ops.zeros((1, img_width, img_width, 1), dtype=dtypes.float32)
      self.assertAllClose(train(x, y), compiled_train(x, y))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
