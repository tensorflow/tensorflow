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

  @parameterized.named_parameters(
      # Activation type Int8 - enable mlir quantizer
      ('UseActivationTypeInt8EnabledMlir', constants.INT8, True),
      # Activation type Int8 - disable mlir quantizer
      ('UseActivationTypeInt8DisabledMlir', constants.INT8, False),
      # Activation type Int16
      ('UseActivationTypeInt16', constants.INT16, False))
  def test_calibration_with_quantization(self, activations_type, enable_mlir):
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
                                                       constants.FLOAT, False,
                                                       activations_type,
                                                       enable_mlir)
    self.assertIsNotNone(quantized_model)

  @parameterized.named_parameters(
      # Activation type Int8 - enable mlir quantizer
      ('UseActivationTypeInt8EnabledMlir', constants.INT8, True),
      # Activation type Int8 - disable mlir quantizer
      ('UseActivationTypeInt8DisableMlir', constants.INT8, False),
      # Activation type Int16 - disable mlir quantizer
      ('UseActivationTypeInt16', constants.INT16, False))
  def test_calibration_with_quantization_allow_float(self, activations_type, enable_mlir):
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
                                                       constants.FLOAT, True,
                                                       activations_type,
                                                       enable_mlir)
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

  @parameterized.named_parameters(
      # Activation type Int8 - enable mlir quantizer
      ('UseActivationTypeInt8 - EnableMlirQuantizer', constants.INT8, True),
      # Activation type Int8 - disable mlir quantizer
      ('UseActivationTypeInt8 - DisableMlirQuantizer', constants.INT8, False),
      # Activation type Int16 - disable mlir quantizer
      ('UseActivationTypeInt16 - DisableEnableMlirQuantizer', constants.INT16, False))
  def test_calibration_with_quantization_multiple_inputs(self, activations_type, enable_mlir):
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
                                                       constants.FLOAT, False,
                                                       activations_type,
                                                       enable_mlir)
    self.assertIsNotNone(quantized_model)

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False))  # disable mlir quantizer
  def test_invalid_model_buffer(self, enable_mlir):
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

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False))  # disable mlir quantizer
  def test_invalid_shape_calibrator_gen(self, enable_mlir):
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
                                       constants.FLOAT, False,
                                       enable_mlir)

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False))  # disable mlir quantizer
  def test_invalid_type_calibrator_gen(self, enable_mlir):
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin')
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator with incorrect shape.
    def input_gen():
      for _ in range(10):
        yield np.ones(shape=(1, 5, 5, 3), dtype=np.int32)

    with self.assertRaises(ValueError):
      quantizer.calibrate_and_quantize(input_gen, constants.FLOAT,
                                       constants.FLOAT, False,
                                       constants.INT8, enable_mlir)


if __name__ == '__main__':
  test.main()
