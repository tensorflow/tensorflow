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
"""Tests for lite.py functionality related to TensorFlow 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_v2_test_util
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking


class FromConcreteFunctionTest(lite_v2_test_util.ModelTest):

  @test_util.run_v2_only
  def testTypeInvalid(self):
    root = self._getSimpleVariableModel()
    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_concrete_functions([root.f])
    self.assertIn('call get_concrete_function', str(error.exception))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  @test_util.run_v2_only
  def testFloat(self, enable_mlir):
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1., shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testScalarInput(self):
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1., shape=[])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    """Convert a single model in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1., shape=[1])
    concrete_func = root.add.get_concrete_function(input_data)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.add(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testConvertMultipleFunctions(self):
    """Convert multiple functions in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1., shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [add_func, sub_func])
    with self.assertRaises(ValueError) as error:
      _ = converter.convert()
    self.assertIn('can only convert a single ConcreteFunction',
                  str(error.exception))

  def _getCalibrationQuantizeModel(self):
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

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False))  # disable mlir quantizer
  def testPostTrainingCalibrateAndQuantize(self, mlir_quantizer):
    func, calibration_gen = self._getCalibrationQuantizeModel()

    # Convert float model.
    float_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter._experimental_new_quantizer = mlir_quantizer
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False))  # disable mlir quantizer
  def testCalibrateAndQuantizeBuiltinInt8(self, mlir_quantizer):
    func, calibration_gen = self._getCalibrationQuantizeModel()

    # Convert float model.
    float_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert model by specifying target spec (instead of optimizations), since
    # when targeting an integer only backend, quantization is mandatory.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter._experimental_new_quantizer = mlir_quantizer
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  def _getTrainingTimeQuantizedModel(self):

    class QLinear(tf.keras.layers.Layer):

      def __init__(self, units=3, **kwargs):
        super(QLinear, self).__init__(**kwargs)
        self.units = units

      def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.min_var = self.add_weight(
            'min',
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False)
        self.max_var = self.add_weight(
            'max',
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False)

      def call(self, inputs):
        x = tf.quantization.fake_quant_with_min_max_vars(
            inputs, self.min_var, self.max_var)

        w_fq = tf.quantization.fake_quant_with_min_max_vars(
            self.w, self.min_var, self.max_var)
        x = tf.matmul(x, w_fq)

        x = tf.quantization.fake_quant_with_min_max_vars(
            x, self.min_var, self.max_var)

        return x

    return tf.keras.Sequential(QLinear(3, input_shape=(2,)))

  @test_util.run_v2_only
  def testTrainingTimeQuantizeConversion(self):
    model = self._getTrainingTimeQuantizedModel()

    float_converter = lite.TFLiteConverterV2.from_keras_model(model)
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

    interpreter = Interpreter(model_content=quantized_tflite)
    self.assertEqual(np.float32, interpreter.get_input_details()[0]['dtype'])

  @test_util.run_v2_only
  def testNewQuantizer(self):
    """Test the model quantized by the new converter."""
    func, calibration_gen = self._getCalibrationQuantizeModel()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func])
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen

    # default quantizer
    quantized_converter._experimental_new_quantizer = False
    old_tflite = quantized_converter.convert()

    # new quantizer
    quantized_converter._experimental_new_quantizer = True
    new_tflite = quantized_converter.convert()

    for _ in range(5):
      input_data = tf.constant(
          np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32))
      old_value = self._evaluateTFLiteModel(old_tflite, [input_data])
      new_value = self._evaluateTFLiteModel(new_tflite, [input_data])
      np.testing.assert_almost_equal(old_value, new_value, 1)

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  @test_util.run_v2_only
  def testEmbeddings(self, enable_mlir):
    """Test model with embeddings."""
    input_data = tf.constant(
        np.array(np.random.random_sample((20)), dtype=np.int32))

    class EmbeddingModel(tf.keras.Model):

      def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.shared_weights = self.add_weight(
            'weights',
            shape=(2000, 300),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=300**(-0.5)))

      @tf.function(input_signature=[tf.TensorSpec(shape=(20), dtype=tf.int32)])
      def func(self, x):
        return tf.gather(self.shared_weights, x)

    # Building the model.
    root = EmbeddingModel()
    concrete_func = root.func.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value[0], 5)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    """Test a concrete function has debug info captured."""
    root = tracking.AutoTrackable()
    root.v1 = tf.Variable(3.)
    root.f = tf.function(lambda x: root.v1 * x)
    input_data = tf.constant(1., shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)


class FromSavedModelTest(lite_v2_test_util.ModelTest):

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

  @test_util.run_v2_only
  def testV1SimpleModel(self):
    """Test a SavedModel."""
    with tf.Graph().as_default():
      saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

      # Convert model and ensure model is not None.
      converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
      tflite_model = converter.convert()
      self.assertTrue(tflite_model)

      interpreter = Interpreter(model_content=tflite_model)
      interpreter.allocate_tensors()

      input_details = interpreter.get_input_details()
      self.assertLen(input_details, 2)
      self.assertStartsWith(input_details[0]['name'], 'inputA')
      self.assertEqual(np.float32, input_details[0]['dtype'])
      self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
      self.assertEqual((0., 0.), input_details[0]['quantization'])

      self.assertStartsWith(
          input_details[1]['name'],
          'inputB',
      )
      self.assertEqual(np.float32, input_details[1]['dtype'])
      self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
      self.assertEqual((0., 0.), input_details[1]['quantization'])

      output_details = interpreter.get_output_details()
      self.assertLen(output_details, 1)
      self.assertStartsWith(output_details[0]['name'], 'add')
      self.assertEqual(np.float32, output_details[0]['dtype'])
      self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
      self.assertEqual((0., 0.), output_details[0]['quantization'])

  @test_util.run_v2_only
  def testConstModel(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = tf.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = tf.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testVariableModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1., shape=[1])
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testSignatures(self):
    """Test values for `signature_keys` argument."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1., shape=[1])
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save)

    # Convert model with invalid `signature_keys`.
    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_saved_model(
          save_dir, signature_keys=['INVALID'])
    self.assertIn("Invalid signature key 'INVALID'", str(error.exception))

    # Convert model with empty `signature_keys`.
    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=[])
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testMultipleFunctionModel(self):
    """Convert multiple functions in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1., shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, {'add': add_func, 'sub': sub_func})

    # Ensure the converter generates.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    self.assertLen(converter._funcs, 2)

    # Try converting multiple functions.
    with self.assertRaises(ValueError) as error:
      _ = converter.convert()
    self.assertIn('This converter can only convert a single ConcreteFunction',
                  str(error.exception))

  @test_util.run_v2_only
  def testNoConcreteFunctionModel(self):
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1., shape=[1])

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir)

    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    self.assertLen(converter._funcs, 0)

    with self.assertRaises(ValueError) as error:
      _ = converter.convert()
    self.assertIn('No ConcreteFunction is specified.', str(error.exception))

  @test_util.run_v2_only
  def testKerasSequentialModel(self):
    """Test a simple sequential tf.Keras model."""
    input_data = tf.constant(1., shape=[1, 1])

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [4.]])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(model, save_dir)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    """Test a SavedModel has debug info captured."""
    input_data = tf.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = tf.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)
    options = save_options.SaveOptions(save_debug_info=True)
    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save, options)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)


class FromKerasModelTest(lite_v2_test_util.ModelTest):

  @test_util.run_v2_only
  def testSequentialModel(self):
    """Test a simple sequential tf.Keras model."""
    input_data = tf.constant(1., shape=[1, 1])

    # Create a simple Keras model.
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [4.]])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testSequentialMultiInputOutputModel(self):
    """Test a tf.Keras model with multiple inputs and outputs."""
    left_input_data = tf.constant(1., shape=[1, 3])
    right_input_data = tf.constant(1., shape=[1, 3])

    # Create a simple Keras model.
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_c_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 2))

    input_a = tf.keras.layers.Input(shape=(3,), name='input_a')
    input_b = tf.keras.layers.Input(shape=(3,), name='input_b')

    dense = tf.keras.layers.Dense(8, name='dense_1')
    interm_a = dense(input_a)
    interm_b = dense(input_b)
    merged = tf.keras.layers.concatenate([interm_a, interm_b], name='merge')

    output_c = tf.keras.layers.Dense(
        3, activation='softmax', name='dense_2')(
            merged)
    output_d = tf.keras.layers.Dense(
        2, activation='softmax', name='dense_3')(
            merged)

    model = tf.keras.models.Model(
        inputs=[input_a, input_b], outputs=[output_c, output_d])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit([input_a_np, input_b_np], [output_c_np, output_d_np], epochs=1)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()

    # Check values from converted model.
    input_data = [left_input_data, right_input_data]
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, input_data)
    for tf_result, tflite_result in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(tf_result, tflite_result, 5)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    """Test a tf.Keras model has debug info captured."""
    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)


class ControlFlowTest(lite_v2_test_util.ModelTest):

  @test_util.run_v2_only
  def testCond(self):
    input_data = {
        'x': tf.constant([1., 2.], shape=[1, 2]),
        'b': tf.constant(True)
    }

    weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

    def true_fn(x):
      return tf.matmul(x, weights)

    def false_fn(x):
      return tf.add(x, weights)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool)
    ])
    def model(x, b):
      return tf.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(**input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data['x'], input_data['b']])[0]
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testStaticRnn(self):
    input_data = tf.constant(
        np.array(np.random.random_sample((3, 10)), dtype=np.float32))

    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(10)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 10], dtype=tf.float32)])
    def model(x):
      seq = tf.split(x, 3, 0)
      return tf.compat.v1.nn.static_rnn(
          cell, seq, dtype=tf.float32, sequence_length=[1])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)[0]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual)

  @test_util.run_v2_only
  def testWhileLoop(self):
    input_data = tf.constant([1., 2., 3., 4.], shape=[2, 2])

    weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

    def condition(x):
      return tf.reduce_sum(x) < 100

    def body(x):
      return tf.add(x, weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[2, 2], dtype=tf.float32)])
    def model(x):
      return tf.while_loop(condition, body, [x])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)[0]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    input_data = tf.constant(
        np.array(np.random.random_sample((3, 10, 10)), dtype=np.float32))

    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(10)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 10, 10], dtype=tf.float32)])
    def model(x):
      return tf.compat.v1.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    for expected, actual in zip(expected_value, actual_value):
      if isinstance(expected, ops.EagerTensor):
        expected = expected.numpy()
      else:
        expected = expected.c.numpy()
      np.testing.assert_almost_equal(expected, actual)

  @parameterized.named_parameters(('LSTM', recurrent_v2.LSTM),
                                  ('SimpleRNN', recurrent.SimpleRNN),
                                  ('GRU', recurrent_v2.GRU))
  @test_util.run_v2_only
  def testKerasRNN(self, rnn_layer):
    # This relies on TFLiteConverter to rewrite unknown batch size to 1. The
    # model will fail if resizing the input to non-1 batch size.
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
    rnn_obj = rnn_layer(units=10, input_shape=(10, 10))
    model = tf.keras.models.Sequential([rnn_obj])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    np.testing.assert_almost_equal(expected_value, actual_value, decimal=5)

  @parameterized.named_parameters(('LSTM', recurrent_v2.LSTM),
                                  ('SimpleRNN', recurrent.SimpleRNN),
                                  ('GRU', recurrent_v2.GRU))
  @test_util.run_v2_only
  def testKerasRNNMultiBatches(self, rnn_layer):
    input_data = tf.constant(
        np.array(np.random.random_sample((4, 10, 10)), dtype=np.float32))
    # Specify a fixed batch size(4) for the test model.
    x = tf.keras.layers.Input(batch_shape=(4, 10, 10))
    y = rnn_layer(units=10, input_shape=(10, 10))(x)
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    np.testing.assert_almost_equal(expected_value, actual_value, decimal=5)

  @test_util.run_v2_only
  def testKerasBidirectionalRNN(self):
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Bidirectional(
            recurrent_v2.LSTM(units=10, return_sequences=True),
            input_shape=(10, 10)))
    model.add(tf.keras.layers.Bidirectional(recurrent_v2.LSTM(units=10)))
    model.add(tf.keras.layers.Dense(5))
    model.add(tf.keras.layers.Activation('softmax'))

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    np.testing.assert_almost_equal(expected_value, actual_value, decimal=5)


class GrapplerTest(lite_v2_test_util.ModelTest):

  @test_util.run_v2_only
  def testConstantFolding(self):
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    input_data = tf.constant([1., 2., 3., 4., 5., 6., 7., 8., 9.], shape=[3, 3])

    @tf.function
    def func(x):
      y_const = tf.constant([1., 2., 3.])
      y_broadcast = tf.broadcast_to(y_const, [3, 3])
      return tf.matmul(x, y_broadcast)

    root = tracking.AutoTrackable()
    root.f = func
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value[0])

    # Enable hybrid quantization, same result
    converter.experimental_new_converter = True
    converter.optimizations = [lite.Optimize.DEFAULT]
    hybrid_tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(hybrid_tflite_model, [input_data])
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value[0])


class UnknownShapes(lite_v2_test_util.ModelTest):

  @test_util.run_v2_only
  def testMatMul(self):
    input_data = tf.constant(
        np.array(np.random.random_sample((10, 4)), dtype=np.float32))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
    def model(in_tensor):
      shape = tf.shape(in_tensor)
      fill = tf.transpose(tf.fill(shape, 1.))
      return tf.matmul(fill, in_tensor)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data], input_shapes=[([-1, 4], [10, 4])])
    np.testing.assert_almost_equal(
        expected_value.numpy(), actual_value[0], decimal=6)

  def testBatchMatMul(self):
    self.skipTest('BatchMatMulV2 does not support unknown batch size.')
    input_data_1 = tf.constant(
        np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32))
    input_data_2 = tf.constant(
        np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32)
    ])
    def model(in_tensor_1, in_tensor_2):
      return tf.matmul(in_tensor_1, in_tensor_2)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data_1, input_data_2)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data_1, input_data_2],
        input_shapes=[([-1, 256, 256], [1, 256, 256])])
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value[0])


if __name__ == '__main__':
  test.main()
