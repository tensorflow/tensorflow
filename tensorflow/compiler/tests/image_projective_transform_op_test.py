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
"""Tests for image projective transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from absl.testing import parameterized

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ImageProjectiveTransformOpTest(xla_test.XLATestCase,
                                     parameterized.TestCase):
  """Test class for ImageProjectiveTransform."""
  INTERPOLATIONS = ["BILINEAR", "NEAREST"]
  FILL_MODES = ["REFLECT", "WRAP", "CONSTANT", "NEAREST"]
  FILL_VALUES = [0, 1, 2]

  def _run_sample(self, images, transforms, interpolation,
                  fill_mode, fill_value):
    """Generate transformed images."""
    images = constant_op.constant(images, dtype=dtypes.float32)
    transforms = constant_op.constant(transforms, dtype=dtypes.float32)
    fill_value = constant_op.constant(fill_value, dtype=dtypes.float32)
    outputs = image_ops.image_projective_transform_v3(
        images=images,
        transforms=transforms,
        output_shape=(images.shape[1], images.shape[2]),
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
        fill_value=fill_value
    )
    return self.evaluate(outputs)

  @parameterized.parameters(*itertools.product(
      INTERPOLATIONS, FILL_MODES, FILL_VALUES))
  def test_1x8_transform(self, interpolation, fill_mode, fill_value):
    """Test 1x8 transform against the one without xla."""
    images_np = np.arange(768).reshape(2, 8, 16, 3).astype(np.float32)
    transforms_np = np.random.rand(1, 8).astype(np.float32) * 2 - 1.0

    with self.test_scope():
      outputs_xla = self._run_sample(images_np, transforms_np,
                                     interpolation, fill_mode, fill_value)

    outputs = self._run_sample(images_np, transforms_np,
                               interpolation, fill_mode, fill_value)

    self.assertAllCloseAccordingToType(outputs_xla, outputs)

  @parameterized.parameters(*itertools.product(
      INTERPOLATIONS, FILL_MODES, FILL_VALUES))
  def test_bx8_transform(self, interpolation, fill_mode, fill_value):
    """Test bx8 transform against the one without xla."""
    images_np = np.arange(768).reshape(2, 8, 16, 3).astype(np.float32)
    transforms_np = np.random.rand(2, 8).astype(np.float32) * 2 - 1.0

    with self.test_scope():
      outputs_xla = self._run_sample(images_np, transforms_np,
                                     interpolation, fill_mode, fill_value)

    outputs = self._run_sample(images_np, transforms_np,
                               interpolation, fill_mode, fill_value)

    self.assertAllCloseAccordingToType(outputs_xla, outputs,
                                       atol=5e-5, rtol=5e-5)


if __name__ == "__main__":
  test.main()
