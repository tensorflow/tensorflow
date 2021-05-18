# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for QuantizationDebugger."""

import csv
import io
import re

from unittest import mock
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.tools.optimize.debugging.python import debugger
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import tracking

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.lite.python import metrics_portable as metrics
except ImportError:
  from tensorflow.lite.python import metrics_nonportable as metrics
# pylint: enable=g-import-not-at-top


def _get_model():
  """Returns somple model with Conv2D and representative dataset gen."""
  root = tracking.AutoTrackable()
  kernel_in = np.array([-2, -1, 1, 2], dtype=np.float32).reshape((2, 2, 1, 1))

  @tf.function(
      input_signature=[tf.TensorSpec(shape=[1, 3, 3, 1], dtype=tf.float32)])
  def func(inp):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    conv = tf.nn.conv2d(inp, kernel, strides=1, padding='SAME')
    output = tf.nn.relu(conv, name='output')
    return output

  root.f = func
  to_save = root.f.get_concrete_function()
  return to_save


def _calibration_gen():
  for i in range(5):
    yield [np.arange(9).reshape((1, 3, 3, 1)).astype(np.float32) * i]


def _convert_model(func):
  """Converts TF model to TFLite float model."""
  converter = lite.TFLiteConverterV2.from_concrete_functions([func])
  return converter.convert()


def _quantize_model(func, calibration_gen, quantized_io=False, debug=True):
  """Quantizes model, in debug or normal mode."""
  converter = lite.TFLiteConverterV2.from_concrete_functions([func])
  converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.representative_dataset = calibration_gen

  # Create a TFLite model with new quantizer and numeric verify ops.
  converter.optimizations = [lite.Optimize.DEFAULT]
  converter.experimental_new_quantizer = True
  if debug:
    converter._experimental_calibrate_only = True
    calibrated = converter.convert()
    return convert.mlir_quantize(
        calibrated, enable_numeric_verify=True, fully_quantize=quantized_io)
  else:
    return converter.convert()


class QuantizationDebuggerTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tf_model = _get_model()
    cls.float_model = _convert_model(cls.tf_model)
    cls.debug_model_float = _quantize_model(
        cls.tf_model, _calibration_gen, quantized_io=False)
    cls.debug_model_int8 = _quantize_model(
        cls.tf_model, _calibration_gen, quantized_io=True)

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_quantization_debugger_layer_metrics(self, quantized_io):
    if quantized_io:
      debug_model = QuantizationDebuggerTest.debug_model_int8
    else:
      debug_model = QuantizationDebuggerTest.debug_model_float

    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))})
    quant_debugger = debugger.QuantizationDebugger(
        quant_debug_model_content=debug_model,
        debug_dataset=_calibration_gen,
        debug_options=options)
    quant_debugger.run()

    expected_metrics = {
        'num_elements': 9,
        'stddev': 0.03850026,
        'mean_error': 0.01673192,
        'max_abs_error': 0.10039272,
        'mean_squared_error': 0.0027558778,
        'l1_norm': 0.023704167,
    }
    self.assertLen(quant_debugger.layer_statistics, 1)
    actual_metrics = next(iter(quant_debugger.layer_statistics.values()))

    self.assertCountEqual(expected_metrics.keys(), actual_metrics.keys())
    for key, value in expected_metrics.items():
      self.assertAlmostEqual(value, actual_metrics[key], places=5)

    buffer = io.StringIO()
    quant_debugger.layer_statistics_dump(buffer)
    reader = csv.DictReader(buffer.getvalue().split())
    actual_values = next(iter(reader))

    expected_values = expected_metrics.copy()
    expected_values.update({
        'op_name': 'CONV_2D',
        'tensor_idx': 7 if quantized_io else 8,
        'scale': 0.15686275,
        'zero_point': -128,
        'tensor_name': r'Identity[1-9]?$'
    })
    for key, value in expected_values.items():
      if isinstance(value, str):
        self.assertIsNotNone(
            re.match(value, actual_values[key]),
            'String is different from expected string. Please fix test code if'
            " it's being affected by graph manipulation changes.")
      elif isinstance(value, list):
        self.assertAlmostEqual(
            value[0], float(actual_values[key][1:-1]), places=5)
      else:
        self.assertAlmostEqual(value, float(actual_values[key]), places=5)

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_quantization_debugger_model_metrics(self, quantized_io):
    if quantized_io:
      debug_model = QuantizationDebuggerTest.debug_model_int8
    else:
      debug_model = QuantizationDebuggerTest.debug_model_float
    options = debugger.QuantizationDebugOptions(
        model_debug_metrics={'stdev': lambda x, y: np.std(x[0] - y[0])})
    quant_debugger = debugger.QuantizationDebugger(
        quant_debug_model_content=debug_model,
        float_model_content=QuantizationDebuggerTest.float_model,
        debug_dataset=_calibration_gen,
        debug_options=options)
    quant_debugger.run()

    expected_metrics = {'stdev': 0.050998904}
    actual_metrics = quant_debugger.model_statistics

    self.assertCountEqual(expected_metrics.keys(), actual_metrics.keys())
    for key, value in expected_metrics.items():
      self.assertAlmostEqual(value, actual_metrics[key], places=5)

  @test_util.run_v2_only
  def test_quantization_debugger_wrong_input_raises_ValueError(self):

    def wrong_calibration_gen():
      for _ in range(5):
        yield [
            np.ones((1, 3, 3, 1), dtype=np.float32),
            np.ones((1, 3, 3, 1), dtype=np.float32)
        ]

    quant_debugger = debugger.QuantizationDebugger(
        quant_debug_model_content=QuantizationDebuggerTest.debug_model_float,
        debug_dataset=wrong_calibration_gen)
    with self.assertRaisesRegex(
        ValueError, r'inputs provided \(2\).+inputs to the model \(1\)'):
      quant_debugger.run()

  @test_util.run_v2_only
  def test_quantization_debugger_non_debug_model_raises_ValueError(self):
    normal_quant_model = _quantize_model(
        QuantizationDebuggerTest.tf_model, _calibration_gen, debug=False)

    with self.assertRaisesRegex(
        ValueError, 'Please check if the quantized model is in debug mode'):
      debugger.QuantizationDebugger(
          quant_debug_model_content=normal_quant_model,
          debug_dataset=_calibration_gen)

  @parameterized.named_parameters(
      ('empty quantization parameter', {
          'quantization_parameters': {}
      }, None),
      ('empty scales/zero points', {
          'quantization_parameters': {
              'scales': [],
              'zero_points': []
          }
      }, None),
      ('invalid scales/zero points', {
          'quantization_parameters': {
              'scales': [1.0],
              'zero_points': []
          }
      }, None),
      ('correct case', {
          'quantization_parameters': {
              'scales': [0.5, 1.0],
              'zero_points': [42, 7]
          }
      }, (0.5, 42)),
  )
  def test_get_quant_params(self, tensor_detail, expected_value):
    self.assertEqual(debugger._get_quant_params(tensor_detail), expected_value)

  @mock.patch.object(metrics.TFLiteMetrics,
                     'increase_counter_debugger_creation')
  def test_quantization_debugger_creation_counter(self, increase_call):
    debug_model = QuantizationDebuggerTest.debug_model_float
    debugger.QuantizationDebugger(
        quant_debug_model_content=debug_model, debug_dataset=_calibration_gen)
    increase_call.assert_called_once()


if __name__ == '__main__':
  test.main()
