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

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable


class ModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Base test class for TensorFlow Lite 2.x model tests."""

  def _evaluateTFLiteModel(self, tflite_model, input_data, input_shapes=None):
    """Evaluates the model on the `input_data`.

    Args:
      tflite_model: TensorFlow Lite model.
      input_data: List of EagerTensor const ops containing the input data for
        each input tensor.
      input_shapes: List of tuples representing the `shape_signature` and the
        new shape of each input tensor that has unknown dimensions.

    Returns:
      [np.ndarray]
    """
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    if input_shapes:
      for idx, (shape_signature, final_shape) in enumerate(input_shapes):
        self.assertTrue(
            (input_details[idx]['shape_signature'] == shape_signature).all())
        index = input_details[idx]['index']
        interpreter.resize_tensor_input(index, final_shape, strict=True)
    interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()

    for input_tensor, tensor_data in zip(input_details, input_data):
      interpreter.set_tensor(input_tensor['index'], tensor_data.numpy())
    interpreter.invoke()
    return [
        interpreter.get_tensor(details['index']) for details in output_details
    ]

  def _evaluateTFLiteModelUsingSignatureDef(self, tflite_model, signature_key,
                                            inputs):
    """Evaluates the model on the `inputs`.

    Args:
      tflite_model: TensorFlow Lite model.
      signature_key: Signature key.
      inputs: Map from input tensor names in the SignatureDef to tensor value.

    Returns:
      Dictionary of outputs.
      Key is the output name in the SignatureDef 'signature_key'
      Value is the output value
    """
    interpreter = Interpreter(model_content=tflite_model)
    signature_runner = interpreter.get_signature_runner(signature_key)
    return signature_runner(**inputs)

  def _getSimpleVariableModel(self):
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    return root

  def _getSimpleModelWithVariables(self):

    class SimpleModelWithOneVariable(autotrackable.AutoTrackable):
      """Basic model with 1 variable."""

      def __init__(self):
        super(SimpleModelWithOneVariable, self).__init__()
        self.var = variables.Variable(array_ops.zeros((1, 10), name='var'))

      @def_function.function
      def assign_add(self, x):
        self.var.assign_add(x)
        return self.var

    return SimpleModelWithOneVariable()

  def _getMultiFunctionModel(self):

    class BasicModel(autotrackable.AutoTrackable):
      """Basic model with multiple functions."""

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

      @def_function.function
      def mul_add(self, x, y):
        if self.z is None:
          self.z = variables.Variable(3.)
        return x * self.z + y

    return BasicModel()

  def _getMultiFunctionModelWithSharedWeight(self):

    class BasicModelWithSharedWeight(autotrackable.AutoTrackable):
      """Model with multiple functions and a shared weight."""

      def __init__(self):
        self.weight = constant_op.constant([1.0],
                                           shape=(1, 512, 512, 1),
                                           dtype=dtypes.float32)

      @def_function.function
      def add(self, x):
        return x + self.weight

      @def_function.function
      def sub(self, x):
        return x - self.weight

      @def_function.function
      def mul(self, x):
        return x * self.weight

    return BasicModelWithSharedWeight()

  def _getMatMulModelWithSmallWeights(self):

    class MatMulModelWithSmallWeights(autotrackable.AutoTrackable):
      """MatMul model with small weights and relatively large biases."""

      def __init__(self):
        self.weight = constant_op.constant([[1e-3, -1e-3], [-2e-4, 2e-4]],
                                           shape=(2, 2),
                                           dtype=dtypes.float32)
        self.bias = constant_op.constant([1.28, 2.55],
                                         shape=(2,),
                                         dtype=dtypes.float32)

      @def_function.function
      def matmul(self, x):
        return x @ self.weight + self.bias

    return MatMulModelWithSmallWeights()

  def _getSqrtModel(self):
    """Returns a model with only one sqrt op, to test non-quantizable op."""

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(1, 10), dtype=dtypes.float32)
    ])
    def sqrt(x):
      return math_ops.sqrt(x)

    def calibration_gen():
      for _ in range(5):
        yield [np.random.uniform(0, 16, size=(1, 10)).astype(np.float32)]

    return sqrt, calibration_gen

  def _assertValidDebugInfo(self, debug_info):
    """Verify the DebugInfo is valid."""
    file_names = set()
    for file_path in debug_info.files:
      file_names.add(os.path.basename(file_path))
    # To make the test independent on how the nodes are created, we only assert
    # the name of this test file.
    self.assertIn('lite_v2_test.py', file_names)
    self.assertNotIn('lite_test.py', file_names)

  def _createV2QATLowBitKerasModel(self, shape, weight_only, num_bits, bit_min,
                                   bit_max):
    """Creates a simple QAT num_bits-Weight Keras Model."""
    input_name = 'input'
    output_name = 'scores'

    class ConvWrapper(tf.keras.layers.Wrapper):
      """A Wrapper for simulating QAT on Conv2D layers."""

      def build(self, input_shape):
        if not self.layer.built:
          self.layer.build(input_shape)
        self.quantized_weights = self.layer.kernel

      def call(self, inputs):
        self.layer.kernel = (
            tf.quantization.fake_quant_with_min_max_vars_per_channel(
                self.quantized_weights, min=[bit_min], max=[bit_max],
                num_bits=num_bits, narrow_range=True))
        if not weight_only:
          quant_inputs = tf.quantization.fake_quant_with_min_max_vars(
              inputs, min=0, max=6, num_bits=8)
          outputs = self.layer.call(quant_inputs)
          return tf.quantization.fake_quant_with_min_max_vars(
              outputs, min=0, max=6, num_bits=8)
        return self.layer.call(inputs)

    input_tensor = tf.keras.layers.Input(shape, name=input_name)
    kernel_shape = (shape[-1], 3, 3, 1)
    # Ensure constant weights contains the min and max.
    initial_weights = np.linspace(
        bit_min, bit_max, np.prod(kernel_shape)).reshape(kernel_shape)
    test_initializer = tf.constant_initializer(initial_weights)
    x = ConvWrapper(tf.keras.layers.Conv2D(
        1, (3, 3), kernel_initializer=test_initializer,
        activation='relu6'))(input_tensor)
    scores = tf.keras.layers.Flatten(name=output_name)(x)
    model = tf.keras.Model(input_tensor, scores)
    return model, input_name, output_name
