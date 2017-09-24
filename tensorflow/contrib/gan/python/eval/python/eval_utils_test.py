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
"""Tests for eval_utils_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python.eval.python import eval_utils_impl as eval_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class UtilsTest(test.TestCase):

  def test_image_grid(self):
    eval_utils.image_grid(
        input_tensor=array_ops.zeros([25, 32, 32, 3]),
        grid_shape=(5, 5))

  # TODO(joelshor): Add more `image_reshaper` tests.
  def test_image_reshaper_image_list(self):
    images = eval_utils.image_reshaper(
        images=array_ops.unstack(array_ops.zeros([25, 32, 32, 3])),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])

  def test_image_reshaper_image(self):
    images = eval_utils.image_reshaper(
        images=array_ops.zeros([25, 32, 32, 3]),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])


if __name__ == '__main__':
  test.main()
