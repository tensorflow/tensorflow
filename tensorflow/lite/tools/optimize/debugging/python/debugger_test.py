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
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.tools.optimize.debugging.python import debugger
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.trackable import autotrackable


def _get_model():
  """Returns somple model with Conv2D and representative dataset gen."""
  root = autotrackable.AutoTrackable()
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
  return (root, to_save)


def _calibration_gen():
  for i in range(5):
    yield [np.arange(9).reshape((1, 3, 3, 1)).astype(np.float32) * i]


def _convert_model(model, func):
  """Converts TF model to TFLite float model."""
  converter = lite.TFLiteConverterV2.from_concrete_functions([func], model)
  # TODO(b/191205988): Explicitly disable saved model lowering in conversion.
  converter.experimental_lower_to_saved_model = False
  return converter.convert()


def _quantize_converter(model, func, calibration_gen, debug=True):
  """Returns a converter appropriate for the function and debug configs."""
  converter = lite.TFLiteConverterV2.from_concrete_functions([func], model)
  converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.representative_dataset = calibration_gen

  # TODO(b/191205988): Explicitly disable saved model lowering in conversion.
  converter.experimental_lower_to_saved_model = False

  # Create a TFLite model with new quantizer and numeric verify ops.
  converter.optimizations = [lite.Optimize.DEFAULT]
  converter.experimental_new_quantizer = True
  if debug:
    converter._experimental_calibrate_only = True
  return converter


def _quantize_model(model,
                    func,
                    calibration_gen,
                    quantized_io=False,
                    debug=True):
  """Quantizes model, in debug or normal mode."""
  converter = _quantize_converter(model, func, calibration_gen, debug)
  if debug:
    calibrated = converter.convert()
    return convert.mlir_quantize(
        calibrated, enable_numeric_verify=True, fully_quantize=quantized_io)
  else:
    return converter.convert()


def _dummy_fn(*unused_args):
  return 0.0


class QuantizationDebugOptionsTest(test_util.TensorFlowTestCase,
                                   parameterized.TestCase):

  @test_util.run_v2_only
  def test_init_duplicate_keys_raises_ValueError(self):
    with self.assertRaises(ValueError):
      debugger.QuantizationDebugOptions(
          layer_debug_metrics={
              'a': _dummy_fn,
              'b': _dummy_fn
          },
          model_debug_metrics={
              'c': _dummy_fn,
              'd': _dummy_fn
          },
          layer_direct_compare_metrics={
              'a': _dummy_fn,
              'e': _dummy_fn
          })

    with self.assertRaises(ValueError):
      debugger.QuantizationDebugOptions(
          layer_debug_metrics={
              'a': _dummy_fn,
              'b': _dummy_fn
          },
          layer_direct_compare_metrics={
              'a': _dummy_fn,
              'e': _dummy_fn
          })


class QuantizationDebuggerTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tf_model_root, cls.tf_model = _get_model()
    cls.float_model = _convert_model(cls.tf_model_root, cls.tf_model)
    cls.debug_model_float = _quantize_model(
        cls.tf_model_root, cls.tf_model, _calibration_gen, quantized_io=False)
    cls.debug_model_int8 = _quantize_model(
        cls.tf_model_root, cls.tf_model, _calibration_gen, quantized_io=True)

  @parameterized.named_parameters(
      ('float_io', False, False),
      ('quantized_io', True, False),
      ('float_io_from_converter', False, True),
      ('quantized_io_from_converter', True, True),
  )
  @test_util.run_v2_only
  def test_layer_metrics(self, quantized_io, from_converter):
    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))})
    if not from_converter:
      if quantized_io:
        debug_model = QuantizationDebuggerTest.debug_model_int8
      else:
        debug_model = QuantizationDebuggerTest.debug_model_float
      quant_debugger = debugger.QuantizationDebugger(
          quant_debug_model_content=debug_model,
          debug_dataset=_calibration_gen,
          debug_options=options)
    else:
      options.fully_quantize = quantized_io
      quant_debugger = debugger.QuantizationDebugger(
          converter=_quantize_converter(self.tf_model_root, self.tf_model,
                                        _calibration_gen),
          debug_dataset=_calibration_gen,
          debug_options=options)

    quant_debugger.run()

    expected_quant_io_metrics = {
        'num_elements': 9,
        'stddev': 0.03850026,
        'mean_error': 0.01673192,
        'max_abs_error': 0.10039272,
        'mean_squared_error': 0.0027558778,
        'l1_norm': 0.023704167,
    }
    expected_float_io_metrics = {
        'num_elements': 9,
        'stddev': 0.050998904,
        'mean_error': 0.007843441,
        'max_abs_error': 0.105881885,
        'mean_squared_error': 0.004357292,
        'l1_norm': 0.035729896,
    }
    expected_metrics = (
        expected_quant_io_metrics
        if quantized_io else expected_float_io_metrics)
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
        'tensor_idx': 7,
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
  def test_model_metrics(self, quantized_io):
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

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_layer_direct_compare_metrics(self, quantized_io):

    def _corr(float_values, quant_values, scale, zero_point):
      dequant_values = (quant_values.astype(np.int32) - zero_point) * scale
      return np.corrcoef(float_values.flatten(), dequant_values.flatten())[0, 1]

    if quantized_io:
      debug_model = QuantizationDebuggerTest.debug_model_int8
    else:
      debug_model = QuantizationDebuggerTest.debug_model_float

    options = debugger.QuantizationDebugOptions(
        layer_direct_compare_metrics={'corr': _corr})
    quant_debugger = debugger.QuantizationDebugger(
        quant_debug_model_content=debug_model,
        debug_dataset=_calibration_gen,
        debug_options=options)
    quant_debugger.run()

    expected_metrics = {
        'corr': 0.99999,
    }
    self.assertLen(quant_debugger.layer_statistics, 1)
    actual_metrics = next(iter(quant_debugger.layer_statistics.values()))

    for key, value in expected_metrics.items():
      self.assertAlmostEqual(value, actual_metrics[key], places=4)

  @test_util.run_v2_only
  def test_wrong_input_raises_ValueError(self):

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
  def test_non_debug_model_raises_ValueError(self):
    normal_quant_model = _quantize_model(
        QuantizationDebuggerTest.tf_model_root,
        QuantizationDebuggerTest.tf_model,
        _calibration_gen,
        debug=False)

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

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_denylisted_ops_from_option_setter(self, quantized_io):
    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))},
        fully_quantize=quantized_io)
    quant_debugger = debugger.QuantizationDebugger(
        converter=_quantize_converter(self.tf_model_root, self.tf_model,
                                      _calibration_gen),
        debug_dataset=_calibration_gen,
        debug_options=options)

    options.denylisted_ops = ['CONV_2D']
    # TODO(b/195084873): The exception is expected to check whether selective
    # quantization was done properly, since after the selective quantization
    # the model will have no quantized layers thus have no NumericVerify ops,
    # resulted in this exception. Marked with a bug to fix this in more
    # straightforward way.
    with self.assertRaisesRegex(
        ValueError, 'Please check if the quantized model is in debug mode'):
      quant_debugger.options = options

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_denylisted_ops_from_option_constructor(self, quantized_io):
    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))},
        fully_quantize=quantized_io,
        denylisted_ops=['CONV_2D'])
    # TODO(b/195084873): Count the number of NumericVerify op.
    with self.assertRaisesRegex(
        ValueError, 'Please check if the quantized model is in debug mode'):
      _ = debugger.QuantizationDebugger(
          converter=_quantize_converter(self.tf_model_root, self.tf_model,
                                        _calibration_gen),
          debug_dataset=_calibration_gen,
          debug_options=options)

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_denylisted_nodes_from_option_setter(self, quantized_io):
    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))},
        fully_quantize=quantized_io)
    quant_debugger = debugger.QuantizationDebugger(
        converter=_quantize_converter(self.tf_model_root, self.tf_model,
                                      _calibration_gen),
        debug_dataset=_calibration_gen,
        debug_options=options)

    options.denylisted_nodes = ['Identity']
    # TODO(b/195084873): Count the number of NumericVerify op.
    with self.assertRaisesRegex(
        ValueError, 'Please check if the quantized model is in debug mode'):
      quant_debugger.options = options

  @parameterized.named_parameters(
      ('float_io', False),
      ('quantized_io', True),
  )
  @test_util.run_v2_only
  def test_denylisted_nodes_from_option_constructor(self, quantized_io):
    options = debugger.QuantizationDebugOptions(
        layer_debug_metrics={'l1_norm': lambda diffs: np.mean(np.abs(diffs))},
        fully_quantize=quantized_io,
        denylisted_nodes=['Identity'])
    # TODO(b/195084873): Count the number of NumericVerify op.
    with self.assertRaisesRegex(
        ValueError, 'Please check if the quantized model is in debug mode'):
      _ = debugger.QuantizationDebugger(
          converter=_quantize_converter(self.tf_model_root, self.tf_model,
                                        _calibration_gen),
          debug_dataset=_calibration_gen,
          debug_options=options)

  @mock.patch.object(metrics.TFLiteMetrics,
                     'increase_counter_debugger_creation')
  def test_creation_counter(self, increase_call):
    debug_model = QuantizationDebuggerTest.debug_model_float
    debugger.QuantizationDebugger(
        quant_debug_model_content=debug_model, debug_dataset=_calibration_gen)
    increase_call.assert_called_once()


if __name__ == '__main__':
  test.main()
