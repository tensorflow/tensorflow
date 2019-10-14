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
"""Tests for lite.py functionality related to MLIR-TFLite converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import zip

from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import tracking


class TFLiteMLIRTest(test_util.TensorFlowTestCase):
  """Base case for testing MLIR-based TFLite converter."""

  def _evaluateTFLiteModel(self, tflite_model, input_data):
    """Evaluates the model on the `input_data`."""
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for input_tensor, tensor_data in zip(input_details, input_data):
      interpreter.set_tensor(input_tensor['index'], tensor_data.numpy())
    interpreter.invoke()
    return [
        interpreter.get_tensor(details['index']) for details in output_details
    ]

  def _getSimpleVariableModel(self):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    return root


class FromSessionTest(TFLiteMLIRTest):

  def testFloat(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testString(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.string)
      out_tensor = array_ops.reshape(in_tensor, shape=[2, 2])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.string_, input_details[0]['dtype'])
    self.assertTrue(([4] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('Reshape', output_details[0]['name'])
    self.assertEqual(np.string_, output_details[0]['dtype'])
    self.assertTrue(([2, 2] == output_details[0]['shape']).all())

  def testQuantization(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    converter.experimental_enable_mlir_converter = True
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {
        'inputA': (0., 1.),
        'inputB': (0., 1.)
    }  # mean, std_dev
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((1., 0.),
                     input_details[0]['quantization'])  # scale, zero_point

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.uint8, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((1., 0.),
                     input_details[1]['quantization'])  # scale, zero_point

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertGreater(output_details[0]['quantization'][0], 0)  # scale

  def testScalarValid(self):
    # Construct a graph using a scalar (empty shape) input.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test conversion with the scalar input shape.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertEqual(len(input_details[0]['shape']), 0)

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertEqual(len(output_details[0]['shape']), 0)

    # Validate inference using the scalar inputs/outputs.
    test_input = np.array(4.0, dtype=np.float32)
    expected_output = np.array(8.0, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testPostTrainingQuantize(self):
    self.skipTest('b/124315492')
    np.random.seed(0)
    with ops.Graph().as_default():
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1],
                                                        [out_tensor])
    float_converter.experimental_enable_mlir_converter = True
    float_tflite = float_converter.convert()

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_converter.experimental_enable_mlir_converter = True
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_tflite = quantized_converter.convert()

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  @test_util.run_in_graph_and_eager_modes
  def testFunctions(self):
    """Tests tf.function in 1.X."""

    @def_function.function
    def plus_placeholder(x, placeholder):
      return x + placeholder

    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1], name='input')
      variable_node = variables.Variable(1.0, name='variable_node')
      defun_node = plus_placeholder(variable_node, placeholder)
      output_node = math_ops.multiply(defun_node, 2.0, name='output_node')

      # Initialize variables in the model.
      sess = session.Session()
      sess.run(variables.variables_initializer([variable_node]))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output_node', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])


class FromConcreteFunctionTest(TFLiteMLIRTest):

  @test_util.run_v2_only
  def testFloat(self):
    root = self._getSimpleVariableModel()
    input_data = constant_op.constant(1., shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testControlFlow(self):
    input_data = {
        'x': constant_op.constant([1., 2.], shape=[1, 2]),
        'b': constant_op.constant(True)
    }

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def true_fn(x):
      return math_ops.matmul(x, weights)

    def false_fn(x):
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[1, 2], dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
    ])
    def model(x, b):
      return control_flow_ops.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(**input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data['x'], input_data['b']])[0]
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testStaticRnn(self):
    input_data = constant_op.constant(
        np.array(np.random.random_sample((3, 10)), dtype=np.float32))

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10], dtype=dtypes.float32)
    ])
    def model(x):
      seq = array_ops.split(x, 3, 0)
      return rnn.static_rnn(
          cell, seq, dtype=dtypes.float32, sequence_length=[1])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)[0]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual)

  @test_util.run_v2_only
  def testLoop(self):
    input_data = constant_op.constant([1., 2., 3., 4.], shape=[2, 2])

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def condition(x):
      return math_ops.reduce_sum(x) < 100

    def body(x):
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[2, 2], dtype=dtypes.float32)
    ])
    def model(x):
      return control_flow_ops.while_loop(condition, body, [x])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    np.testing.assert_almost_equal(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    input_data = constant_op.constant(
        np.array(np.random.random_sample((3, 10, 10)), dtype=np.float32))

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10, 10], dtype=dtypes.float32)
    ])
    def model(x):
      return rnn.dynamic_rnn(cell, x, dtype=dtypes.float32)

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
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

  @test_util.run_v2_only
  def testKerasLSTM(self):
    input_data = constant_op.constant(
        np.array(np.random.random_sample((10, 10, 10)), dtype=np.float32))

    model = keras.models.Sequential(
        [keras.layers.LSTM(units=10, input_shape=(10, 10))])

    run_model = def_function.function(model.__call__)
    concrete_func = run_model.get_concrete_function(
        tensor_spec.TensorSpec((10, 10, 10), dtype=dtypes.float32))

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    np.testing.assert_almost_equal(expected_value, actual_value)


class FromKerasModelTest(TFLiteMLIRTest, parameterized.TestCase):

  @parameterized.named_parameters(('LSTM', recurrent_v2.LSTM),
                                  ('SimpleRNN', recurrent.SimpleRNN),
                                  ('GRU', recurrent_v2.GRU))
  @test_util.run_v2_only
  def testKerasRNN(self, rnn_layer):
    # This test case is similar to `FromConcreteFunctionTest.testKerasLSTM`
    # above, but it's more concise to pass the Keras model directly.
    # This relies on TFLiteConverter to rewrite unknown batch size to 1. The
    # model will fail if resizing the input to non-1 batch size.
    input_data = constant_op.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
    rnn_obj = rnn_layer(units=10, input_shape=(10, 10))
    model = keras.models.Sequential([rnn_obj])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)

    np.testing.assert_almost_equal(expected_value, actual_value, decimal=5)


class TestFlexMode(TFLiteMLIRTest):

  def testSession(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_enable_mlir_converter = True
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    tflite_model = converter.convert()

    # Ensures the model contains TensorFlow ops.
    # TODO(nupurgarg): Check values once there is a Python delegate interface.
    interpreter = Interpreter(model_content=tflite_model)
    with self.assertRaises(RuntimeError) as error:
      interpreter.allocate_tensors()
    self.assertIn(
        'Regular TensorFlow ops are not supported by this interpreter.',
        str(error.exception))

  @test_util.run_v2_only
  def testConcreteFunc(self):
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
    converter.experimental_enable_mlir_converter = True
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    tflite_model = converter.convert()

    # Ensures the model contains TensorFlow ops.
    # TODO(nupurgarg): Check values once there is a Python delegate interface.
    interpreter = Interpreter(model_content=tflite_model)
    with self.assertRaises(RuntimeError) as error:
      interpreter.allocate_tensors()
    self.assertIn(
        'Regular TensorFlow ops are not supported by this interpreter.',
        str(error.exception))


if __name__ == '__main__':
  test.main()
