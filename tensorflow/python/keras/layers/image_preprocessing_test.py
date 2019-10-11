# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image preprocessing layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import image_preprocessing
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import test


class ResizingTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_height, expected_width):
    np.random.seed(1337)
    num_samples = 2
    orig_height = 5
    orig_width = 8
    channels = 3
    kwargs.update({'height': expected_height, 'width': expected_width})
    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          image_preprocessing.Resizing,
          kwargs=kwargs,
          input_shape=(num_samples, orig_height, orig_width, channels),
          expected_output_shape=(None, expected_height, expected_width,
                                 channels))

  @parameterized.named_parameters(
      ('down_sample_bilinear_2_by_2', {'interpolation': 'bilinear'}, 2, 2),
      ('down_sample_bilinear_3_by_2', {'interpolation': 'bilinear'}, 3, 2),
      ('down_sample_nearest_2_by_2', {'interpolation': 'nearest'}, 2, 2),
      ('down_sample_nearest_3_by_2', {'interpolation': 'nearest'}, 3, 2),
      ('down_sample_area_2_by_2', {'interpolation': 'area'}, 2, 2),
      ('down_sample_area_3_by_2', {'interpolation': 'area'}, 3, 2))
  def test_down_sampling(self, kwargs, expected_height, expected_width):
    with CustomObjectScope({'Resizing': image_preprocessing.Resizing}):
      self._run_test(kwargs, expected_height, expected_width)

  @parameterized.named_parameters(
      ('up_sample_bilinear_10_by_12', {'interpolation': 'bilinear'}, 10, 12),
      ('up_sample_bilinear_12_by_12', {'interpolation': 'bilinear'}, 12, 12),
      ('up_sample_nearest_10_by_12', {'interpolation': 'nearest'}, 10, 12),
      ('up_sample_nearest_12_by_12', {'interpolation': 'nearest'}, 12, 12),
      ('up_sample_area_10_by_12', {'interpolation': 'area'}, 10, 12),
      ('up_sample_area_12_by_12', {'interpolation': 'area'}, 12, 12))
  def test_up_sampling(self, kwargs, expected_height, expected_width):
    with CustomObjectScope({'Resizing': image_preprocessing.Resizing}):
      self._run_test(kwargs, expected_height, expected_width)

  @parameterized.named_parameters(
      ('reshape_bilinear_10_by_4', {'interpolation': 'bilinear'}, 10, 4))
  def test_reshaping(self, kwargs, expected_height, expected_width):
    with CustomObjectScope({'Resizing': image_preprocessing.Resizing}):
      self._run_test(kwargs, expected_height, expected_width)

  def test_invalid_interpolation(self):
    with self.assertRaises(NotImplementedError):
      image_preprocessing.Resizing(5, 5, 'invalid_interpolation')

  def test_config_with_custom_name(self):
    layer = image_preprocessing.Resizing(5, 5, name='image_preproc')
    config = layer.get_config()
    layer_1 = image_preprocessing.Resizing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  test.main()
