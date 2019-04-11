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
import numpy as np

from tensorflow.lite.python import lite
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking


class TestModels(test_util.TensorFlowTestCase):

  def _evaluateTFLiteModel(self, tflite_model, input_data):
    """Evaluates the model on the `input_data`."""
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for input_tensor, tensor_data in zip(input_details, input_data):
      interpreter.set_tensor(input_tensor['index'], tensor_data.numpy())
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

  def _getSimpleVariableModel(self):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    return root

  def _getMultiFunctionModel(self):

    class BasicModel(tracking.AutoTrackable):

      def __init__(self):
        self.y = None
        self.z = None

      @def_function.function
      def add(self, x):
        if self.y is None:
          self.y = variables.Variable(2.)
        return x + self.y

      @def_function.function
      def sub(self, x):
        if self.z is None:
          self.z = variables.Variable(3.)
        return x - self.z

    return BasicModel()


class FromConcreteFunctionTest(TestModels):

  @test_util.run_v2_only
  def testTypeInvalid(self):
    root = self._getSimpleVariableModel()
    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_concrete_functions([root.f])
    self.assertIn('call from_concrete_function', str(error.exception))

  @test_util.run_v2_only
  def testFloat(self):
    root = self._getSimpleVariableModel()
    input_data = constant_op.constant(1., shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testScalarInput(self):
    root = self._getSimpleVariableModel()
    input_data = constant_op.constant(1., shape=[])
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
    input_data = constant_op.constant(1., shape=[1])
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
    input_data = constant_op.constant(1., shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [add_func, sub_func])
    with self.assertRaises(ValueError) as error:
      _ = converter.convert()
    self.assertIn('can only convert a single ConcreteFunction',
                  str(error.exception))


class FromSavedModelTest(TestModels):

  @test_util.run_v2_only
  def testConstModel(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
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
    input_data = constant_op.constant(1., shape=[1])
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
    input_data = constant_op.constant(1., shape=[1])
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
    input_data = constant_op.constant(1., shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, {'add': add_func, 'sub': sub_func})

    # Ensure the converter generates.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    self.assertEqual(len(converter._funcs), 2)

    # Try converting multiple functions.
    with self.assertRaises(ValueError) as error:
      _ = converter.convert()
    self.assertIn('This converter can only convert a single ConcreteFunction',
                  str(error.exception))


class FromKerasModelTest(TestModels):

  @test_util.run_v2_only
  def testSequentialModel(self):
    """Test a simple sequential tf.Keras model."""
    input_data = constant_op.constant(1., shape=[1, 1])

    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]

    model = keras.models.Sequential(
        [keras.layers.Dense(units=1, input_shape=[1])])
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
    left_input_data = constant_op.constant(1., shape=[1, 3])
    right_input_data = constant_op.constant(1., shape=[1, 3])

    # Create a simple Keras model.
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_c_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 2))

    input_a = keras.layers.Input(shape=(3,), name='input_a')
    input_b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(8, name='dense_1')
    interm_a = dense(input_a)
    interm_b = dense(input_b)
    merged = keras.layers.concatenate([interm_a, interm_b], name='merge')

    output_c = keras.layers.Dense(
        3, activation='softmax', name='dense_2')(
            merged)
    output_d = keras.layers.Dense(
        2, activation='softmax', name='dense_3')(
            merged)

    model = keras.models.Model(
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
      np.testing.assert_almost_equal(tf_result[0], tflite_result, 5)


if __name__ == '__main__':
  test.main()
