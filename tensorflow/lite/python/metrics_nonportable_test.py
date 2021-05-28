# Lint as: python2, python3
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
"""TensorFlow Lite Python metrics helper TFLiteMetrics check."""
import gc
import os
from unittest import mock

import numpy as np
import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.lite.python import metrics_nonportable as metrics
from tensorflow.lite.python.convert import ConverterError
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.tracking import tracking


class MetricsNonportableTest(test_util.TensorFlowTestCase):

  def test_TFLiteMetrics_creation_no_arg_success(self):
    metrics.TFLiteMetrics()

  def test_TFLiteMetrics_creation_arg_success(self):
    metrics.TFLiteMetrics('hash', '/path/to/model')

  def test_TFLiteMetrics_creation_fails_with_only_hash(self):
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(model_hash='hash')

  def test_TFLiteMetrics_creation_fail2_with_only_model_path(self):
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(model_path='/path/to/model')

  def test_debugger_creation_counter_increase_multiple_same_topic_success(self):
    try:
      stub = metrics.TFLiteMetrics()
      stub.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 1)
      stub2 = metrics.TFLiteMetrics()
      stub2.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 2)
      del stub
      gc.collect()
      stub2.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 3)
    except:
      raise Exception('No exception should be raised.')

  def test_interpreter_creation_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_interpreter_creation()
    self.assertEqual(
        metrics._counter_interpreter_creation.get_cell('python').value(), 1)

  def test_converter_attempt_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_attempt()
    self.assertEqual(metrics._counter_conversion_attempt.get_cell().value(), 1)

  def test_converter_success_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_success()
    self.assertEqual(metrics._counter_conversion_success.get_cell().value(), 1)

  def test_converter_params_set_success(self):
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name').value(), 'value')

  def test_converter_params_multiple_set_success(self):
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')
    stub.set_converter_param('name', 'value1')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name').value(), 'value1')

  def test_converter_params_multiple_label_success(self):
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name1', 'value1')
    stub.set_converter_param('name2', 'value2')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name1').value(), 'value1')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name2').value(), 'value2')


class ConverterMetricsTest(test_util.TensorFlowTestCase):
  """Testing conversion metrics."""

  def _constructGraphDef(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    return (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

  def test_conversion_from_constructor_success(self):
    frozen_graph_def = self._constructGraphDef()

    # Check metrics when conversion successed.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('input_format', '1'),
        mock.call.set_converter_param('enable_mlir_converter', 'True'),
        mock.call.set_converter_param('allow_custom_ops', 'False'),
        mock.call.set_converter_param('api_version', '1'),
    ], any_order=True)  # pyformat: disable

  def test_conversion_from_constructor_fail(self):
    frozen_graph_def = self._constructGraphDef()

    # Check metrics when conversion failed.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('wrong_tensor', [2, 16, 16, 3])],
                                     ['add'])
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    with self.assertRaises(ConverterError):
      converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.set_converter_param('output_format', '2'),
        mock.call.set_converter_param('select_user_tf_ops', 'None'),
        mock.call.set_converter_param('post_training_quantize', 'False'),
    ], any_order=True)  # pyformat: disable
    mock_metrics.increase_counter_converter_success.assert_not_called()

  def _getIntegerQuantizeModel(self):
    np.random.seed(0)

    root = tracking.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5, 5, 3], dtype=tf.float32)])
    def func(inp):
      conv = tf.nn.conv2d(
          inp, tf.ones([3, 3, 3, 16]), strides=[1, 1, 1, 1], padding='SAME')
      output = tf.nn.relu(conv, name='output')
      return output

    def calibration_gen():
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    root.f = func
    to_save = root.f.get_concrete_function()
    return (to_save, calibration_gen)

  def test_conversion_from_frozen_graph_v2(self):
    func, calibration_gen = self._getIntegerQuantizeModel()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    quantized_converter._tflite_metrics = mock_metrics
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.set_converter_param(
            'optimization_post_training_integer_quantize', 'True'),
        mock.call.set_converter_param('inference_type', 'tf.int8'),
        mock.call.set_converter_param('select_user_tf_ops', 'None'),
        mock.call.set_converter_param('activations_type', 'tf.int8'),
    ], any_order=True)  # pyformat: disable

  def test_conversion_from_keras_v2(self):
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('inference_type', 'tf.float32'),
        mock.call.set_converter_param('target_ops', 'TFLITE_BUILTINS'),
        mock.call.set_converter_param('optimization_default', 'False'),
    ], any_order=True)  # pyformat: disable

  def _createV1SavedModel(self, shape):
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor_1 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputB')
        in_tensor_2 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputA')
        variable_node = tf.Variable(1.0, name='variable_node')
        out_tensor = in_tensor_1 + in_tensor_2 * variable_node
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        sess.run(tf.compat.v1.variables_initializer([variable_node]))
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def test_conversion_from_saved_model(self):
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteSavedModelConverter(saved_model_dir, set(['serve']),
                                               ['serving_default'])
    converter.experimental_new_converter = True
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('enable_mlir_converter', 'True'),
    ], any_order=True)  # pyformat: disable

  def test_conversion_from_saved_model_v2(self):
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.experimental_new_converter = False
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('enable_mlir_converter', 'False'),
        mock.call.set_converter_param('api_version', '2'),
    ], any_order=True)  # pyformat: disable

  def disable_converter_counter_metrics(self, tflite_metrics):
    def empty_func():
      pass
    tflite_metrics.increase_counter_converter_attempt = empty_func
    tflite_metrics.increase_counter_converter_success = empty_func

  def test_export_at_conversion_done(self):
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_metrics = converter._tflite_metrics
    mock_exporter = mock.MagicMock()
    tflite_metrics._metrics_exporter = mock_exporter
    self.disable_converter_counter_metrics(tflite_metrics)
    mock_exporter.ExportMetrics.assert_not_called()
    converter.convert()
    mock_exporter.ExportMetrics.assert_called_once()
    tflite_metrics.__del__()
    mock_exporter.ExportMetrics.assert_called_once()

  def test_export_at_exit(self):
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_metrics = converter._tflite_metrics
    mock_exporter = mock.MagicMock()
    tflite_metrics._metrics_exporter = mock_exporter
    self.disable_converter_counter_metrics(tflite_metrics)
    mock_exporter.ExportMetrics.assert_not_called()
    tflite_metrics.__del__()
    mock_exporter.ExportMetrics.assert_called_once()


if __name__ == '__main__':
  test.main()
