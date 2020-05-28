# Lint as: python2, python3
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
"""Tests for Calibrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range

from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class CalibratorTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def test_calibration_with_quantization(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize(input_gen,
                                                       constants.FLOAT,
                                                       constants.FLOAT, False)
    self.assertIsNotNone(quantized_model)

  def test_calibration_with_quantization_allow_float(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize(input_gen,
                                                       constants.FLOAT,
                                                       constants.FLOAT, True)
    self.assertIsNotNone(quantized_model)

  def test_calibration_with_quantization_single_op(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize_single(
        input_gen, constants.FLOAT, constants.FLOAT, True, 'conv2d_8/BiasAdd')
    self.assertIsNotNone(quantized_model)

  def test_calibration_with_quantization_multiple_inputs(self):
    # Load multi add model from test data.
    # This model has 4 inputs of size (1, 8, 8, 3).
    model_path = resource_loader.get_path_to_datafile(
        '../../testdata/multi_add.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 8, 8, 3), dtype=np.float32) for _ in range(4)]

    quantized_model = quantizer.calibrate_and_quantize(input_gen,
                                                       constants.FLOAT,
                                                       constants.FLOAT, False)
    self.assertIsNotNone(quantized_model)

  def test_invalid_model_buffer(self):
    float_model = b'\0' * 100
    with self.assertRaisesRegex(ValueError, 'Failed to parse the model'):
      _calibrator.Calibrator(float_model)

  # TODO(fengliuai): enable mlir quantizer
  def test_empty_calibrator_gen(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    def empty_input_gen():
      for i in ():
        yield i

    with self.assertRaises(RuntimeError):
      quantizer.calibrate_and_quantize(empty_input_gen, constants.FLOAT,
                                       constants.FLOAT, False)

  def test_invalid_shape_calibrator_gen(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator with incorrect shape.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 2, 2, 3), dtype=np.float32)]

    with self.assertRaisesRegex(ValueError, 'Size mismatch'):
      quantizer.calibrate_and_quantize(input_gen, constants.FLOAT,
                                       constants.FLOAT, False, False)

  def test_invalid_type_calibrator_gen(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator with incorrect type.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.int32)]

    with self.assertRaises(ValueError):
      quantizer.calibrate_and_quantize(input_gen, constants.FLOAT,
                                       constants.FLOAT, False)

  def test_calibration(self):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate(input_gen)
    self.assertIsNotNone(quantized_model)

if __name__ == '__main__':
  test.main()
