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

from tensorflow.lite.python import lite
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking


class FromConcreteFunctionTest(test_util.TensorFlowTestCase):

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

  @test_util.run_v2_only
  def testTypeInvalid(self):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)

    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_concrete_function(root.f)
    self.assertIn('call from_concrete_function', str(error.exception))

  @test_util.run_v2_only
  def testFloat(self):
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_function(concrete_func)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testConstSavedModel(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    concrete_func = saved_model.signatures['serving_default']

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_function(concrete_func)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testVariableSavedModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    concrete_func = saved_model.signatures['serving_default']

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_function(concrete_func)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    """Test a basic model with Variables."""

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

    input_data = constant_op.constant(1., shape=[1])
    root = BasicModel()
    concrete_func = root.add.get_concrete_function(input_data)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_function(concrete_func)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.add(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testKerasModel(self):
    input_data = constant_op.constant(1., shape=[1, 1])

    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]

    model = keras.models.Sequential(
        [keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    # Get the concrete function from the Keras model.
    @def_function.function
    def to_save(x):
      return model(x)

    concrete_func = to_save.get_concrete_function(
        tensor_spec.TensorSpec([None, 1], dtypes.float32))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_function(concrete_func)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = to_save(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)


if __name__ == '__main__':
  test.main()
