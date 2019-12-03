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

from tensorflow.python.framework import errors
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import image_preprocessing
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import image_ops_impl as image_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
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


def get_numpy_center_crop(images, expected_height, expected_width):
  orig_height = images.shape[1]
  orig_width = images.shape[2]
  height_start = int((orig_height - expected_height) / 2)
  width_start = int((orig_width - expected_width) / 2)
  height_end = height_start + expected_height
  width_end = width_start + expected_width
  return images[:, height_start:height_end, width_start:width_end, :]


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CenterCropTest(keras_parameterized.TestCase):

  def _run_test(self, expected_height, expected_width):
    np.random.seed(1337)
    num_samples = 2
    orig_height = 5
    orig_width = 8
    channels = 3
    kwargs = {'height': expected_height, 'width': expected_width}
    input_images = np.random.random(
        (num_samples, orig_height, orig_width, channels)).astype(np.float32)
    expected_output = get_numpy_center_crop(
        input_images, expected_height, expected_width)
    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          image_preprocessing.CenterCrop,
          kwargs=kwargs,
          input_shape=(num_samples, orig_height, orig_width, channels),
          input_data=input_images,
          expected_output=expected_output,
          expected_output_shape=(None, expected_height, expected_width,
                                 channels))

  @parameterized.named_parameters(
      ('center_crop_3_by_4', 3, 4),
      ('center_crop_3_by_2', 3, 2))
  def test_center_crop_aligned(self, expected_height, expected_width):
    with CustomObjectScope({'CenterCrop': image_preprocessing.CenterCrop}):
      self._run_test(expected_height, expected_width)

  @parameterized.named_parameters(
      ('center_crop_4_by_5', 4, 5),
      ('center_crop_4_by_3', 4, 3))
  def test_center_crop_mis_aligned(self, expected_height, expected_width):
    with CustomObjectScope({'CenterCrop': image_preprocessing.CenterCrop}):
      self._run_test(expected_height, expected_width)

  @parameterized.named_parameters(
      ('center_crop_4_by_6', 4, 6),
      ('center_crop_3_by_2', 3, 2))
  def test_center_crop_half_mis_aligned(self, expected_height, expected_width):
    with CustomObjectScope({'CenterCrop': image_preprocessing.CenterCrop}):
      self._run_test(expected_height, expected_width)

  @parameterized.named_parameters(
      ('center_crop_5_by_12', 5, 12),
      ('center_crop_10_by_8', 10, 8),
      ('center_crop_10_by_12', 10, 12))
  def test_invalid_center_crop(self, expected_height, expected_width):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'assertion failed'):
      self._run_test(expected_height, expected_width)

  def test_config_with_custom_name(self):
    layer = image_preprocessing.CenterCrop(5, 5, name='image_preproc')
    config = layer.get_config()
    layer_1 = image_preprocessing.CenterCrop.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class RandomCropTest(keras_parameterized.TestCase):

  def _run_test(self, expected_height, expected_width):
    np.random.seed(1337)
    num_samples = 2
    orig_height = 5
    orig_width = 8
    channels = 3
    kwargs = {'height': expected_height, 'width': expected_width}
    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          image_preprocessing.RandomCrop,
          kwargs=kwargs,
          input_shape=(num_samples, orig_height, orig_width, channels),
          expected_output_shape=(None, expected_height, expected_width,
                                 channels))

  @parameterized.named_parameters(
      ('random_crop_5_by_12', 5, 12),
      ('random_crop_10_by_8', 10, 8),
      ('random_crop_10_by_12', 10, 12))
  def test_invalid_random_crop(self, expected_height, expected_width):
    with self.assertRaises(errors.InvalidArgumentError):
      with CustomObjectScope({'RandomCrop': image_preprocessing.RandomCrop}):
        self._run_test(expected_height, expected_width)

  def test_training_with_mock(self):
    if test.is_built_with_rocm():
      # TODO(rocm):
      # re-enable this test once ROCm adds support for
      # the StatefulUniformFullInt Op (on the GPU)
      self.skipTest('Feature not supported on ROCm')
    np.random.seed(1337)
    height, width = 3, 4
    height_offset = np.random.randint(low=0, high=3)
    width_offset = np.random.randint(low=0, high=5)
    mock_offset = [0, height_offset, width_offset, 0]
    with test.mock.patch.object(
        stateless_random_ops, 'stateless_random_uniform',
        return_value=mock_offset):
      with self.cached_session(use_gpu=True):
        layer = image_preprocessing.RandomCrop(height, width)
        inp = np.random.random((12, 5, 8, 3))
        actual_output = layer(inp, training=1)
        expected_output = inp[:, height_offset:(height_offset + height),
                              width_offset:(width_offset + width), :]
        self.assertAllClose(expected_output, actual_output)

  @parameterized.named_parameters(
      ('random_crop_4_by_6', 4, 6),
      ('random_crop_3_by_2', 3, 2))
  def test_random_crop_output_shape(self, expected_height, expected_width):
    if test.is_built_with_rocm():
      # TODO(rocm):
      # re-enable this test once ROCm adds support for
      # the StatefulUniformFullInt Op (on the GPU)
      self.skipTest('Feature not supported on ROCm')
    with CustomObjectScope({'RandomCrop': image_preprocessing.RandomCrop}):
      self._run_test(expected_height, expected_width)

  def test_predicting_with_mock_longer_height(self):
    np.random.seed(1337)
    height, width = 3, 3
    inp = np.random.random((12, 10, 6, 3))
    with self.cached_session(use_gpu=True):
      layer = image_preprocessing.RandomCrop(height, width)
      actual_output = layer(inp, training=0)
      resized_inp = image_ops.resize_images_v2(
          inp, size=[5, 3])
      expected_output = resized_inp[:, 1:4, :, :]
      self.assertAllClose(expected_output, actual_output)

  def test_predicting_with_mock_longer_width(self):
    np.random.seed(1337)
    height, width = 4, 6
    inp = np.random.random((12, 8, 16, 3))
    with self.cached_session(use_gpu=True):
      layer = image_preprocessing.RandomCrop(height, width)
      actual_output = layer(inp, training=0)
      resized_inp = image_ops.resize_images_v2(
          inp, size=[4, 8])
      expected_output = resized_inp[:, :, 1:7, :]
      self.assertAllClose(expected_output, actual_output)

  def test_config_with_custom_name(self):
    layer = image_preprocessing.RandomCrop(5, 5, name='image_preproc')
    config = layer.get_config()
    layer_1 = image_preprocessing.RandomCrop.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  test.main()
