# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lite.py."""

import io
import logging
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
from tensorflow import keras

from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import util
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import mlir_quantize
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.variables import global_variables_initializer as _global_variables_initializer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class LiteTest(test_util.TensorFlowTestCase):
  """Base class of all the tests in this module."""


class TestModels(LiteTest):

  def assertValidDebugInfo(self, debug_info):
    """Verify the DebugInfo is valid."""
    file_names = set()
    for file_path in debug_info.files:
      file_names.add(os.path.basename(file_path))
    # To make the test independent on how the nodes are created, we only assert
    # the name of this test file.
    self.assertIn('lite_test.py', file_names)
    self.assertNotIn('lite_v2_test.py', file_names)


class FromConstructor(TestModels):

  # Tests invalid constructors using a dummy value for the GraphDef.
  def testInvalidConstructor(self):
    message = (
        'If input_tensors and output_tensors are None, both '
        'input_arrays_with_shape and output_arrays|control_output_arrays must '
        'be defined.')

    # `output_arrays` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(
          None, None, [], input_arrays_with_shape=[('input', [3,
                                                              9])]).convert()
    self.assertEqual(message, str(error.exception))

    # `input_arrays_with_shape` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(None, [], None, output_arrays=['output']).convert()
    self.assertEqual(message, str(error.exception))

  # Tests valid constructors using a dummy value for the GraphDef.
  def testValidConstructor(self):
    converter = lite.TFLiteConverter(
        None,
        None,
        None,
        input_arrays_with_shape=[('input', [3, 9])],
        output_arrays=['output'])
    self.assertFalse(converter._has_valid_tensors())
    self.assertEqual(converter.get_input_arrays(), ['input'])

    with self.assertRaises(ValueError) as error:
      converter._set_batch_size(1)
    self.assertEqual(
        'The batch size cannot be set for this model. Please use '
        'input_shapes parameter.', str(error.exception))

    converter = lite.TFLiteConverter(None, ['input_tensor'], ['output_tensor'])
    self.assertTrue(converter._has_valid_tensors())

  def testRedundantArgumentsWarning(self):
    """Test if the warning message when there are redundant arguments."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      out_tensor = math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    log = io.BytesIO() if six.PY2 else io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    converter = lite.TFLiteConverter(frozen_graph_def, [in_tensor],
                                     [out_tensor],
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])

    input_warning_message = 'input_arrays_with_shape will be ignored'
    output_warning_message = 'output_arrays will be ignored'

    # Convert model and ensure model is not None.
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    self.assertIn(input_warning_message, log.getvalue())
    self.assertIn(output_warning_message, log.getvalue())
    logging.root.removeHandler(handler)

  def testShapeOverriding(self):
    """Test a shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('in_tensor', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testPartialShapeOverriding(self):
    """Test a partial shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor_a = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_a')
      in_tensor_b = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_b')
      math_ops.add(in_tensor_a, in_tensor_b, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor_a', [2, 16, 16, 3])], ['add'])
    # There is an unhandled Placeholder op.
    with self.assertRaises(ConverterError):
      converter.convert()

  def testInvalidShapeOverriding(self):
    """Test an invalid shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('wrong_tensor', [2, 16, 16, 3])],
                                     ['add'])
    with self.assertRaises(ConverterError):
      converter.convert()


class FromSessionTest(TestModels, parameterized.TestCase):

  def testFloatModel(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFloatModelQuantizedInput(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_input_type = dtypes.uint8
    converter.inference_type = dtypes.float32
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])  # float

  def testForgottenCallToAllocateTensors(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()
    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_index = interpreter.get_input_details()[0]['index']
    dummy_tensor = np.ones(shape=[1, 16, 16, 3], dtype=np.float32)
    with self.assertRaises(ValueError):
      interpreter.set_tensor(input_index, dummy_tensor)

  @parameterized.named_parameters(
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
      ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True),
      ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True))
  def testIntegerQuantizationWithUnsupportedOps(self,
                                                is_int_only,
                                                is_int16_quantize,
                                                inference_input_output_type,
                                                enable_mlir_quantizer=False):
    with ops.Graph().as_default():
      in_tensor_a = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      in_tensor_b = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      # ceil kernel does not support int8 nor int16 types neither.
      left = math_ops.ceil(in_tensor_a)
      out_tensor_b = math_ops.tanh(in_tensor_b)
      add = math_ops.add(left, out_tensor_b)
      # ceil kernel does not support int8 nor int16 types neither.
      out_tensor_a = math_ops.ceil(add)
      sess = session.Session()

    def calibration_gen():
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(3)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3)).astype(np.float32)
        ]

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_a, in_tensor_b], [out_tensor_a, out_tensor_b])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet
            .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8, lite.OpsSet.TFLITE_BUILTINS
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet
            .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS
        ]

    quantized_converter.inference_input_type = inference_input_output_type
    quantized_converter.inference_output_type = inference_input_output_type
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    expected_dtype = inference_input_output_type.as_numpy_dtype
    # Allow float32 for fallback on non-quantizable op.
    expected_ceil_dtype = (
        expected_dtype if enable_mlir_quantizer else dtypes.float32)

    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual(input_details[0]['dtype'], expected_ceil_dtype)
    self.assertEqual(input_details[1]['dtype'], expected_dtype)
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(output_details[0]['dtype'], expected_ceil_dtype)
    self.assertEqual(output_details[1]['dtype'], expected_dtype)

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelMlirDynamicRangeQuant', False, False, False),
      ('_PerTensorMlirDynamicRangeQuant', True, False, False),
      ('_PerChannelTocoDynamicRangeQuant', False, False, False, False),
      ('_PerTensorTocoDynamicRangeQuant', True, False, False, False))
  def testDisablePerChannelQuantization(self,
                                        disable_per_channel=False,
                                        enable_mlir_quantizer=False,
                                        representative_dataset=True,
                                        enable_mlir_converter=True):
    k_conv_name = 'Conv2D1' if enable_mlir_converter else 'ones'
    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 38
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel(
          k_num_filters)
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    if representative_dataset:
      quantized_converter.representative_dataset = calibration_gen
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel:
      quantized_converter._experimental_disable_per_channel = (
          disable_per_channel)
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    detail = next((d for d in interpreter.get_tensor_details()
                   if d['name'] == k_conv_name))
    quant_params = detail['quantization_parameters']
    expected_num_params = 1 if disable_per_channel else k_num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    self.assertLen(quant_params['zero_points'], expected_num_params)

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testString(self, enable_mlir_converter):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.string)
      out_tensor = array_ops.reshape(in_tensor, shape=[2, 2])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = enable_mlir_converter
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.string_, input_details[0]['dtype'])
    self.assertAllEqual([4], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('Reshape', output_details[0]['name'])
    self.assertEqual(np.string_, output_details[0]['dtype'])
    self.assertAllEqual([2, 2], output_details[0]['shape'])
    # TODO(b/122659643): Test setting/getting string data via the python
    # interpreter API after support has been added.

  def testIntermediateInputArray(self):
    """Convert a model from an intermediate input array."""
    with ops.Graph().as_default():
      in_tensor_init = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      in_tensor_final = in_tensor_init + in_tensor_init
      out_tensor = in_tensor_final + in_tensor_final
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor_final],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('add', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add_1', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSizeNoneInvalid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test None as shape when dynamic shapes are disabled. Run with TOCO in
    # order to invoke shape checking code.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = False
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual('Provide an input shape for input array \'Placeholder\'.',
                     str(error.exception))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testScalarValid(self, enable_mlir_converter):
    # Construct a graph using a scalar (empty shape) input.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test conversion with the scalar input shape.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = enable_mlir_converter
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertEmpty(input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertEmpty(input_details[0]['shape'])

    # Validate inference using the scalar inputs/outputs.
    test_input = np.array(4.0, dtype=np.float32)
    expected_output = np.array(8.0, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertEqual(expected_output, output_data)

  def testSizeInvalid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, None, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test invalid shape. None after 1st dimension. Run with TOCO in order to
    # invoke shape checking code.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = False
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'None is only supported in the 1st dimension. Tensor '
        '\'Placeholder\' has invalid shape \'[1, None, 16, 3]\'.',
        str(error.exception))

  def testSizeNone(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, None, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test None after 1st dimension.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 1, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    # Resize tensor with strict checking.
    with self.assertRaises(RuntimeError) as error:
      interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
    self.assertIn(
        'ResizeInputTensorStrict only allows mutating unknown dimensions '
        'identified by -1.', str(error.exception))

    # Resize tensor and invoke.
    interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
    interpreter.allocate_tensors()
    interpreter.invoke()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])

    output_details = interpreter.get_output_details()
    self.assertAllEqual([1, -1, 16, 3], output_details[0]['shape_signature'])

  def testResizeTensorInputStrict(self):
    # Ensures that resize_tensor_input(strict=True) works as expected.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)

    # Resize incorrect value.
    with self.assertRaises(RuntimeError) as error:
      interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
    self.assertIn(
        'ResizeInputTensorStrict only allows mutating unknown dimensions '
        'identified by -1.', str(error.exception))

    # Resize correct value.
    interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
    interpreter.allocate_tensors()

  def testBatchSizeValid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testBatchSizeNonZero(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[None, 4], dtype=dtypes.float32, name='input1')
      in_tensor_2 = array_ops.placeholder(
          shape=[4, 10], dtype=dtypes.float32, name='input2')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('input1', input_details[0]['name'])
    self.assertAllEqual([1, 4], input_details[0]['shape'])
    self.assertEqual('input2', input_details[1]['name'])
    self.assertAllEqual([4, 10], input_details[1]['shape'])

  def testFreezeGraph(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      # Get the second output to ensure freezing properly processes tensor names
      # like 'X:1'.
      out_tensor = nn_ops.top_k(in_tensor + var, name='top_k')[1]
      sess = session.Session()
      sess.run(_global_variables_initializer())

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('top_k:1', output_details[0]['name'])
    self.assertEqual(np.int32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 1], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testGraphviz(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.output_format = lite_constants.GRAPHVIZ_DOT
    graphviz_output = converter.convert()
    self.assertIsNotNone(graphviz_output)

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testDumpGraphviz(self, enable_mlir_converter):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = enable_mlir_converter
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure interpreter is able to allocate and check graphviz data.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    num_items_graphviz = len(os.listdir(graphviz_dir))
    self.assertIsNotNone(num_items_graphviz)
    self.assertIsNotNone(
        os.path.exists(os.path.join(graphviz_dir, 'toco_AT_IMPORT.dot')))
    self.assertIsNotNone(
        os.path.exists(
            os.path.join(graphviz_dir, 'toco_AFTER_TRANSFORMATIONS.dot')))

    # new converter doesn't support `dump_graphviz_video` flag
    if not enable_mlir_converter:
      # Convert model and ensure model is not None.
      converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                    [out_tensor])
      converter.experimental_new_converter = enable_mlir_converter
      graphviz_dir = self.get_temp_dir()
      converter.dump_graphviz_dir = graphviz_dir
      converter.dump_graphviz_video = True
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

      # Ensure graphviz folder has more data after using video flag.
      num_items_graphviz_video = len(os.listdir(graphviz_dir))
      self.assertGreater(num_items_graphviz_video, num_items_graphviz)

  def testDumpConversionSummary(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    log_dir = self.get_temp_dir()
    converter.conversion_summary_dir = log_dir
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    self.assertNotEmpty(os.listdir(log_dir))

  def testDumpConversionSummaryWithOldConverter(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = False
    log_dir = self.get_temp_dir()
    converter.conversion_summary_dir = log_dir
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    # Check nothing is generated under the conversion summary path.
    num_items_conversion_summary = len(os.listdir(log_dir))
    self.assertEqual(num_items_conversion_summary, 0)

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testQuantizeDynamicRange(self, enable_mlir_converter):
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
    float_converter.experimental_new_converter = enable_mlir_converter
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])

    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testQuantizeDynamicRangeDeprecatedPostTrainingQuantizeAttribute(
      self, enable_mlir_converter):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    self.assertFalse(quantized_converter.post_training_quantize)
    quantized_converter.experimental_new_converter = enable_mlir_converter

    quantized_converter.post_training_quantize = True
    self.assertTrue(quantized_converter.post_training_quantize)
    self.assertEqual(quantized_converter.optimizations, [lite.Optimize.DEFAULT])

    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

  def _getIntegerQuantizeModel(self, num_filters=16):
    np.random.seed(0)
    inp = array_ops.placeholder(
        dtype=dtypes.float32, shape=(1, 5, 5, 3), name='input')
    conv = nn_ops.conv2d(
        inp,
        filter=array_ops.ones([3, 3, 3, num_filters]),
        strides=[1, 1, 1, 1],
        padding='SAME')
    output = nn_ops.relu(conv, name='output')

    def calibration_gen():
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    return (inp, output, calibration_gen)

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testQuantizeInt8AllowFloat(self, enable_mlir_converter):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      # Quantize model to Int8: with enable mlir
      ('UseTfliteBuiltinsIntEnableMLIR', [lite.OpsSet.TFLITE_BUILTINS_INT8
                                         ], True),
      # Quantize model to Int8: with disable mlir
      ('UseTfliteBuiltinsIntDisableMLIR', [lite.OpsSet.TFLITE_BUILTINS_INT8
                                          ], False),
      # Quantize model to Int16: with disable mlir
      ('UseTfliteBuiltinsInt16DisableMLIR', [
          lite.OpsSet
          .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
      ], False),
      ('UseTfliteBuiltinsInt16EnableMLIR', [
          lite.OpsSet
          .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
      ], True))
  def testQuantizeInt8And16x8(self, supported_ops, enable_mlir_converter):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_converter.experimental_new_converter = enable_mlir_converter
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert model by specifying target spec (instead of optimizations), since
    # when targeting an integer only backend, quantization is mandatory.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_ops = supported_ops
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testQuantizeInt8InputOutput(self, enable_mlir_converter):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_converter.experimental_new_converter = enable_mlir_converter
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.inference_input_type = dtypes.int8
    quantized_converter.inference_output_type = dtypes.int8
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # The input and output types should be int8.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.int8, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.int8, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testInvalidQuantizeInt8(self, enable_mlir_converter):
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

    # Attempt to convert to quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    # Restricting to int8 type only
    quantized_converter.target_spec.supported_types = [dtypes.int8]
    # A representative dataset is required for full fixed point quantization.
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'representative_dataset is required when specifying '
        'TFLITE_BUILTINS_INT8 or INT8 supported types.', str(error.exception))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testQuantizeUInt8(self, enable_mlir_converter):
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
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {
        'inputA': (0., 1.),
        'inputB': (0., 1.)
    }  # mean, std_dev
    converter.experimental_new_converter = enable_mlir_converter
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.uint8, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((1., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertGreater(output_details[0]['quantization'][0], 0)  # scale

  def testQuantizeUInt8UsingDefaultRangeStats(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    converter.default_ranges_stats = (0, 6)  # min, max
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertGreater(output_details[0]['quantization'][0], 0)  # scale

  @parameterized.named_parameters(
      # Quantize to Float16 even if rep data provided.
      ('UseRepresentativeData', True, False, True, False, False, False, False,
       False),
      # Quantize to Float16 if no rep data provided.
      ('NoRepresentativeData', False, False, True, False, False, False, False,
       False),
      # Quantize to Float16 and set Float16Accumulation
      ('SpecifyFloat16Accumulation', False, False, True, True, False, False,
       False, False),
      # Post training quantization if both rep data and int8 included.
      ('UseSampleDataIncludeInt8', True, True, False, False, False, True, False,
       False),
      # Quantize to Float16 even if rep data provided with mlir.
      ('UseRepresentativeDataMlir', True, False, True, False, False, False,
       True, False),
      # Quantize to Float16 if no rep data provided with mlir.
      ('NoRepresentativeDataMlir', False, False, True, False, False, False,
       True, False),
      # Post training quantization if both rep data and int8 included with mlir.
      ('SampleDataIncludeInt8Mlir', True, True, False, False, False, True, True,
       False),
      # Same as above, but using MLIR quantizer
      ('SampleDataIncludeInt8MlirQuant', True, True, False, False, False, True,
       True, True))
  def testQuantizeFloat16(self, use_rep_data, include_int8,
                          is_float16_quantized, is_float16_accumulation,
                          is_error, is_post_training_quantized,
                          enable_mlir_converter, enable_mlir_quantizer):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    bias_idx = 1 if enable_mlir_converter else 0
    bias_name = 'Conv2D' if enable_mlir_converter else 'Conv2D_bias'

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_converter.experimental_new_converter = enable_mlir_converter
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    interpreter = Interpreter(model_content=float_tflite_model)
    interpreter.allocate_tensors()
    self.assertEqual(interpreter.get_tensor_details()[bias_idx]['name'],
                     bias_name)
    self.assertEqual(interpreter.get_tensor_details()[bias_idx]['dtype'],
                     dtypes.float32)

    # MLIR quantizer has different bias index.
    if enable_mlir_quantizer:
      bias_idx = 2

    # Convert model to quantized version
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [dtypes.float16]
    if include_int8:
      quantized_converter.target_spec.supported_types.append(dtypes.int8)
    if use_rep_data:
      quantized_converter.representative_dataset = calibration_gen
    if is_float16_accumulation:
      quantized_converter.target_spec.experimental_supported_accumulation_type = dtypes.float16  # pylint: disable=line-too-long

    if is_error:
      with self.assertRaises(ValueError) as error:
        quantized_converter.convert()
      self.assertEqual(
          'representative_dataset is required when specifying '
          'TFLITE_BUILTINS_INT8 or INT8 supported types.', str(error.exception))

    else:
      quantized_tflite_model = quantized_converter.convert()
      self.assertIsNotNone(quantized_tflite_model)
      interpreter = Interpreter(model_content=quantized_tflite_model)
      interpreter.allocate_tensors()
      self.assertEqual(interpreter.get_tensor_details()[bias_idx]['name'],
                       bias_name)

      if is_float16_quantized:
        # Verify that bias constant is float16 type.
        self.assertEqual(interpreter.get_tensor_details()[bias_idx]['dtype'],
                         dtypes.float16)
      elif is_post_training_quantized:
        # Verify that bias constants is int32 type.
        self.assertEqual(interpreter.get_tensor_details()[bias_idx]['dtype'],
                         dtypes.int32)
      else:
        raise ValueError('Invalid test options.')

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testInvalidQuantizeFloat16(self, enable_mlir_converter):
    with ops.Graph().as_default():
      inp, output, _ = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Specify float16 quantization
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_converter = enable_mlir_converter
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [dtypes.float16]
    # Specify only int8 builtin ops
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'TFLITE_BUILTINS_INT8 requires smallest supported type to be INT8.',
        str(error.exception))

  @parameterized.named_parameters(('InferenceType_INT8', dtypes.int8),
                                  ('InferenceType_UINT8', dtypes.uint8))
  def testInvalidQuantizeQATModelRequiresInputStats(self, quantized_type):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1.)
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor], [out_tensor])

    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_type = quantized_type
      quantized_converter.convert()
    self.assertEqual(
        'The `quantized_input_stats` flag must be defined when either '
        '`inference_type` flag or `inference_input_type` flag is set to '
        'tf.int8 or tf.uint8. Currently, `inference_type=tf.{}` and '
        '`inference_input_type=None`.'.format(quantized_type.name),
        str(error.exception))

    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_type = dtypes.float32
      quantized_converter.inference_input_type = quantized_type
      quantized_converter.convert()
    self.assertEqual(
        'The `quantized_input_stats` flag must be defined when either '
        '`inference_type` flag or `inference_input_type` flag is set to '
        'tf.int8 or tf.uint8. Currently, `inference_type=tf.float32` and '
        '`inference_input_type=tf.{}`.'.format(quantized_type.name),
        str(error.exception))

    quantized_converter.inference_type = quantized_type
    quantized_converter.inference_input_type = quantized_type

    input_arrays = quantized_converter.get_input_arrays()
    quantized_converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    quantized_converter.convert()

  def testInvalidQuantizeQATModelMissingInputStats(self):
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
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'inputA': (0., 1.)}  # mean, std_dev
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'Quantization input stats are not available for input tensors '
        '\'inputB\'.', str(error.exception))

  def testTrainingTimeAndPostTrainingCalibrateAndQuantize(self):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    converter = lite.TFLiteConverter.from_session(sess, [inp], [output])

    # extra flags to trigger training time quantization conversion
    converter.inference_type = dtypes.int8
    converter.inference_input_type = dtypes.float32
    converter.inference_output_type = dtypes.float32
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    # trigger post-training quantization
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.experimental_new_quantizer = True
    quantized_tflite_model = converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    # calibration only api
    converter._experimental_calibrate_only = True
    calibrated_tflite = converter.convert()
    quantized_tflite_model = mlir_quantize(
        calibrated_tflite, fully_quantize=True)
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertEqual(np.int8, input_details[0]['dtype'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(np.int8, output_details[0]['dtype'])

  def testFloatTocoConverter(self):
    """Tests deprecated test TocoConverter."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the interpreter is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testMultipleOutputNodeNames(self):
    """Tests converting a graph with an op that have multiple outputs."""
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.float32)
      out0, out1, out2, out3 = array_ops.split(
          input_tensor, [1, 1, 1, 1], axis=0)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [input_tensor],
                                                  [out0, out1, out2, out3])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    interpreter.set_tensor(input_details[0]['index'],
                           np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 4)
    self.assertEqual(1.0, interpreter.get_tensor(output_details[0]['index']))
    self.assertEqual(2.0, interpreter.get_tensor(output_details[1]['index']))
    self.assertEqual(3.0, interpreter.get_tensor(output_details[2]['index']))
    self.assertEqual(4.0, interpreter.get_tensor(output_details[3]['index']))

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  @test_util.run_in_graph_and_eager_modes
  def testFunctions(self, enable_mlir_converter):
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
    converter.experimental_new_converter = enable_mlir_converter
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output_node', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInferenceInputOutputTypeFloatDefault(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

  def testInferenceInputOutputTypeQuantizedUint8Default(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

  def testReusingConverterWithDifferentPostTrainingQuantization(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])

    converter.post_training_quantize = True
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    converter.post_training_quantize = False
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  def testResizeWithShape(self):
    with ops.Graph().as_default():
      # Construct a graph with a dynamically shapped input and an internal node
      # that relies on the output of that input's shape.
      in_tensor = array_ops.placeholder(
          shape=[None, None], dtype=dtypes.float32)
      in_tensor2 = [[1, 2], [3, 4]]
      out_tensor = array_ops.reshape(in_tensor2, array_ops.shape(in_tensor))
      sess = session.Session()

    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([1, 1], input_details[0]['shape'])
    self.assertAllEqual([-1, -1], input_details[0]['shape_signature'])

    # Resize tensor and invoke.
    interpreter.resize_tensor_input(0, [4])
    interpreter.allocate_tensors()
    interpreter.invoke()

    # The output should be reshaped properly according to the resized input.
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.int32, output_details[0]['dtype'])
    self.assertAllEqual([4], output_details[0]['shape'])
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertAllEqual([1, 2, 3, 4], output_data)

  def testResizingIntermediateDynamicTensor(self):
    # This is a regression test for the case where shape of dynamic output
    # tensors changes between invocations.
    # See also https://github.com/tensorflow/tensorflow/issues/26549
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[1, 1], dtype=dtypes.float32)
      input2_tensor = array_ops.placeholder(shape=[1], dtype=dtypes.float32)

      # The bug is triggered only when dynamic tensor is intermediate. Putting
      # some other ops around it.
      neg = math_ops.negative(input2_tensor)
      padding = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int32)
      output_tensor = array_ops.pad(input_tensor, padding) + neg

      sess = session.Session()

    converter = lite.TFLiteConverter.from_session(
        sess, [input_tensor, padding, input2_tensor], [output_tensor])
    tflite_model = converter.convert()

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[1, 1], [1, 1]], dtype=np.int32))
    interpreter.invoke()

    # Without the fix, invocation will fail when changing the shape of
    # intermediate dynamic tensors.
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[2, 2], [2, 2]], dtype=np.int32))
    interpreter.invoke()

  def testGraphDebugInfo(self):
    """Test a session has debug info captured."""

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

    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)

    # Check the add node in the inlined function is included.
    func = sess.graph.as_graph_def().library.function[0].signature.name
    self.assertIn(('add@' + six.ensure_str(func)), converter._debug_info.traces)

  def testOutputOnlyModel(self):
    with ops.Graph().as_default():
      out_tensor = random_ops.random_normal(shape=[3])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [], [out_tensor])
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]

    # Empty input array is a valid input.
    self.assertTrue(converter._has_valid_tensors())

    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)


class FromFrozenGraphFile(LiteTest):

  def testFloat(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFloatWithShapesArray(self):
    """Test a shape overriding case."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, ['Placeholder'], ['add'],
        input_shapes={'Placeholder': [2, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])

  def testInvalidShapesArray(self):
    """Test an invalid shape overriding case, which has a wrong input name."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_frozen_graph(
          graph_def_file, ['Placeholder'], ['add'],
          input_shapes={'wrong_input': [2, 16, 16, 3]})

  def testPartialShapesArray(self):
    """Test a shape overriding case, with the only one input among two."""
    with ops.Graph().as_default():
      a = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='a')
      b = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='b')
      _ = math_ops.add(a, b, name='add')
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, ['a', 'b'], ['add'], input_shapes={'a': [2, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])

  def testFreezeGraph(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + var
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Ensure the graph with variables cannot be converted.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual('Please freeze the graph using freeze_graph.py.',
                     str(error.exception))

  def testPbtxt(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pbtxt')
    write_graph(sess.graph_def, '', graph_def_file, True)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInvalidFileNotFound(self):
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph('invalid_file', ['Placeholder'],
                                             ['add'])
    self.assertEqual('File \'invalid_file\' does not exist.',
                     str(error.exception))

  def testInvalidFileBadData(self):
    graph_def_file = os.path.join(self.get_temp_dir(), 'invalid_file')
    with gfile.Open(graph_def_file, 'wb') as temp_file:
      temp_file.write('bad data')
      temp_file.flush()

    # Attempts to convert the invalid model.
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual(
        'Unable to parse input file \'{}\'.'.format(graph_def_file),
        str(error.exception))

  def testFloatTocoConverter(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    """Test a frozen graph doesn't have debug info captured."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    converter.convert()
    # GraphDebugInfo should be none for frozen graph.
    self.assertFalse(converter._debug_info)


class FromFrozenGraphObjectDetection(LiteTest):

  def _initObjectDetectionArgs(self):
    # Initializes the arguments required for the object detection model.
    # Looks for the model file which is saved in a different location internally
    # and externally.
    filename = resource_loader.get_path_to_datafile('testdata/tflite_graph.pb')
    if not os.path.exists(filename):
      filename = os.path.join(
          resource_loader.get_root_dir_with_all_resources(),
          '../tflite_mobilenet_ssd_quant_protobuf/tflite_graph.pb')
      if not os.path.exists(filename):
        raise IOError("File '{0}' does not exist.".format(filename))

    self._graph_def_file = filename
    self._input_arrays = ['normalized_input_image_tensor']
    self._output_arrays = [
        'TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
        'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3'
    ]
    self._input_shapes = {'normalized_input_image_tensor': [1, 300, 300, 3]}

  def testTFLiteGraphDef(self):
    # Tests the object detection model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                                       self._input_arrays,
                                                       self._output_arrays,
                                                       self._input_shapes)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('normalized_input_image_tensor', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 300, 300, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 4)
    self.assertEqual('TFLite_Detection_PostProcess', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 10, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual('TFLite_Detection_PostProcess:1',
                     output_details[1]['name'])
    self.assertAllEqual([1, 10], output_details[1]['shape'])
    self.assertEqual('TFLite_Detection_PostProcess:2',
                     output_details[2]['name'])
    self.assertAllEqual([1, 10], output_details[2]['shape'])
    self.assertEqual('TFLite_Detection_PostProcess:3',
                     output_details[3]['name'])
    self.assertAllEqual([1], output_details[3]['shape'])

  def testTFLiteGraphDefWithControlOutput(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[5, 5], dtype=dtypes.float32, name='input')
      out_tensor = in_tensor + in_tensor
      logging_ops.print_v2(out_tensor)
      sess = session.Session()

    converter = lite.TFLiteConverter(
        sess.graph_def,
        input_tensors=None,
        output_tensors=None,
        input_arrays_with_shape=[('input', [5, 5])],
        output_arrays=None,
        experimental_debug_info_func=None)
    converter._control_output_arrays = ['PrintV2']
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    self.assertEqual(model.operatorCodes[0].builtinCode,
                     schema_fb.BuiltinOperator.ADD)
    self.assertEqual(model.operatorCodes[1].builtinCode,
                     schema_fb.BuiltinOperator.CUSTOM)
    self.assertEqual(model.operatorCodes[1].customCode, b'FlexStringFormat')
    self.assertEqual(model.operatorCodes[2].builtinCode,
                     schema_fb.BuiltinOperator.CUSTOM)
    self.assertEqual(model.operatorCodes[2].customCode, b'FlexPrintV2')

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([5, 5], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 0)

  def testModifyIOToUint8(self):
    # Tests the object detection model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    def representative_dataset_gen():
      for _ in range(2):
        yield [
            np.random.uniform(low=0, high=1,
                              size=(1, 300, 300, 3)).astype(np.float32)
        ]

    converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                                       self._input_arrays,
                                                       self._output_arrays,
                                                       self._input_shapes)
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = {lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_type = dtypes.int8
    converter.inference_input_type = dtypes.uint8
    converter.inference_output_type = dtypes.uint8
    converter.experimental_new_quantizer = True
    converter.quantized_input_stats = {
        'normalized_input_image_tensor': (0., 1.)
    }  # mean, std_dev
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    self.assertIsNotNone(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    quant_opcode_idxs = util.get_quantize_opcode_idx(model)

    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    for op in operators:
      if op.opcodeIndex in quant_opcode_idxs:
        input_type = util._convert_tflite_enum_type_to_tf_type(
            tensors[op.inputs[0]].type)
        if op.outputs[0] in subgraph.outputs:
          self.assertEqual(input_type, dtypes.float32)


class FromSavedModelTest(TestModels):

  def _createSavedModel(self, shape):
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor_1 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputB')
        in_tensor_2 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputA')
        out_tensor = in_tensor_1 + in_tensor_2
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def testSimpleModel(self):
    """Test a SavedModel."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testOldConverterWarning(self):
    """Test if the warning message when using TOCO is logged."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
    log = io.BytesIO() if six.PY2 else io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    warning_message = 'Please consider switching to the new converter'
    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.experimental_new_converter = False
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    self.assertIn(warning_message, log.getvalue())
    logging.root.removeHandler(handler)

  def testNewConverterOptOut(self):
    """Test if the opt out message when using New converter is logged."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
    log = io.BytesIO() if six.PY2 else io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    optout_message = ('Using experimental converter: '
                      'If you encountered a problem')
    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    self.assertIn(optout_message, log.getvalue())
    logging.root.removeHandler(handler)

  def testNoneBatchSize(self):
    """Test a SavedModel, with None in input tensor's shape."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testOrderInputArrays(self):
    """Test a SavedModel ordering of input arrays."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputB', 'inputA'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testShapeOverriding(self):
    """Test a SavedModel with the input_shapes arugment."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_shapes={
            'inputA': [2, 16, 16, 3],
            'inputB': [2, 16, 16, 3]
        })
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testWrongInputShapes(self):
    """Test a SavedModel with a wrong name in the input_shapes argument."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_saved_model(
          saved_model_dir,
          input_arrays=['inputA'],
          input_shapes={'wrong_input': [1, 16, 16, 3]})

  def testSubsetInputShaapes(self):
    """Test a SavedModel with a subset of the input array names of the model."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_arrays=['inputA'],
        input_shapes={'inputA': [1, 16, 16, 3]})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

    # Check case where input shape is None.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': None})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

  def testSimpleModelTocoConverter(self):
    """Test a SavedModel with deprecated TocoConverter."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    """Test a SavedModel has debug info captured."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)


class MyAddLayer(keras.layers.Layer):

  def __init__(self, increment, **kwargs):
    super(MyAddLayer, self).__init__(**kwargs)
    self._increment = increment

  def call(self, inputs):
    return inputs + self._increment

  def get_config(self):
    config = super(MyAddLayer, self).get_config()
    config['increment'] = self._increment
    return config


class FromKerasFile(TestModels, parameterized.TestCase):

  def setUp(self):
    super(FromKerasFile, self).setUp()
    self._keras_file = None
    self._custom_objects = None
    if not context.executing_eagerly():
      keras.backend.clear_session()

  def tearDown(self):
    if self._keras_file:
      os.remove(self._keras_file)
    super(FromKerasFile, self).tearDown()

  def _getSequentialModel(self, include_custom_layer=False):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    if include_custom_layer:
      model.add(MyAddLayer(1.0))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    try:
      fd, self._keras_file = tempfile.mkstemp('.h5')
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    if include_custom_layer:
      self._custom_objects = {'MyAddLayer': MyAddLayer}

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testSequentialModel(self, test_context):
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel()

      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testCustomLayer(self, test_context):
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel(include_custom_layer=True)

      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, custom_objects=self._custom_objects)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(
        self._keras_file, custom_objects=self._custom_objects)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelInputArray(self):
    """Test a Sequential tf.keras model testing input arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_arrays=['invalid-input'])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

    # Valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_arrays=['dense_input'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  def testSequentialModelInputShape(self):
    """Test a Sequential tf.keras model testing input shapes argument."""
    self._getSequentialModel()

    # Passing in shape of invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_shapes={'invalid-input': [2, 3]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))

    # Passing in shape of valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={'dense_input': [2, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check input shape from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertAllEqual([2, 3], input_details[0]['shape'])

  def testSequentialModelOutputArray(self):
    """Test a Sequential tf.keras model testing output arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid output array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, output_arrays=['invalid-output'])
    self.assertEqual("Invalid tensors 'invalid-output' were found.",
                     str(error.exception))

    # Valid output array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, output_arrays=['time_distributed/Reshape_1'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testFunctionalModel(self, test_context):
    """Test a Functional tf.keras model with default inputs."""
    with test_context():
      inputs = keras.layers.Input(shape=(3,), name='input')
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(
          loss=keras.losses.MSE,
          optimizer='sgd',
          metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      model.predict(x)
      fd, self._keras_file = tempfile.mkstemp('.h5')
      try:
        keras.models.save_model(model, self._keras_file)
      finally:
        os.close(fd)

      # Convert to TFLite model.
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def _getFunctionalModelMultipleInputs(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')
    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.mae],
        loss_weights=[1., 0.5])

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    model.predict([input_a_np, input_b_np], batch_size=5)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

  def testFunctionalModelMultipleInputs(self):
    """Test a Functional tf.keras model with multiple inputs and outputs."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([1, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testShapeOverriding(self):
    """Test a Functional tf.keras model with input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={
            'input_a': {2, 3},
            'input_b': {2, 3}
        })
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([2, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([2, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testPartialShapeOverriding(self):
    """Test a Functional tf.keras model with partial input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={'input_a': {2, 3}})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([2, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testWrongShapeOverriding(self):
    """Test a Functional tf.keras model with wrong input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_shapes={'wrong_input': {2, 3}})

  def testFunctionalSequentialModel(self):
    """Test a Functional tf.keras model containing a Sequential model."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model = keras.models.Model(model.input, model.output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    model.predict(x)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelTocoConverter(self):
    """Test a Sequential tf.keras model with deprecated TocoConverter."""
    self._getSequentialModel()

    converter = lite.TocoConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testGraphDebugInfo(self, test_context):
    """Test a Sequential tf.keras model has debug info captured."""
    with test_context():
      self._getSequentialModel()
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      converter.convert()
      self.assertValidDebugInfo(converter._debug_info)

  def testSparsifyModel(self):
    self._getSequentialModel()

    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    converter.optimizations = {lite.Optimize.EXPERIMENTAL_SPARSITY}
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  def testSparsifyQuantizedModel(self):
    self._getSequentialModel()

    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    converter.optimizations = {
        lite.Optimize.DEFAULT, lite.Optimize.EXPERIMENTAL_SPARSITY
    }
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)


class GrapplerTest(TestModels, parameterized.TestCase):

  def testConstantFolding(self):
    ops.disable_eager_execution()
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[3, 3], dtype=dtypes.float32)
      y_const = constant_op.constant([1., 2., 3.])
      y_broadcast = gen_array_ops.broadcast_to(y_const, [3, 3])
      out_tensor = math_ops.matmul(in_tensor, y_broadcast, name='output')
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([3, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([3, 3], output_details[0]['shape'])

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testInputNodeIsNotFolded(self, enable_mlir_converter):
    ops.disable_eager_execution()
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      y_const = constant_op.constant([1., 2., 3.])
      y_add = y_const + y_const
      out_tensor = in_tensor * y_add
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor, y_const],
                                                  [out_tensor])
    converter.experimental_new_converter = enable_mlir_converter
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual('Const', input_details[1]['name'])

  def testGrapplerConstFolding(self):
    # Constant folding converts the following add operation to tf.broadcast_to
    # operation which was not supported by the TFLite at the time this test was
    # added.
    @def_function.function
    def plus_placeholder(x, placeholder):
      return x + placeholder

    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
      out_tensor = plus_placeholder(
          array_ops.zeros([2, 2, 2]),
          array_ops.reshape(in_tensor, shape=[2, 2]))
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])


class DefaultConverterAttrsTest(LiteTest):

  def testAttrs(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])

    # Assert output format.
    self.assertEqual(converter.output_format, lite_constants.TFLITE)

    # Assert the default inference type is float.
    self.assertEqual(converter.inference_type, dtypes.float32)

    # Assert the default inference type overrides are None.
    self.assertIsNone(converter.inference_input_type)
    self.assertIsNone(converter.inference_output_type)

    # Assert the default quantization options are not set.
    self.assertEqual(converter.quantized_input_stats, {})
    self.assertIsNone(converter.default_ranges_stats)
    self.assertFalse(converter.reorder_across_fake_quant)
    self.assertFalse(converter.change_concat_input_ranges)

    # Assert dropping control dependency is enabled by default.
    self.assertIsNotNone(converter.drop_control_dependency)

    # Assert dumping extra information is disabled by default.
    self.assertIsNone(converter.dump_graphviz_dir)
    self.assertFalse(converter.dump_graphviz_video)
    self.assertIsNone(converter.conversion_summary_dir)


class ControlFlowV1OpsTest(LiteTest):

  def testConverterErrorOnControlFlowV1Ops(self):
    graph_def_file = resource_loader.get_path_to_datafile(
        'testdata/control_flow_v1.pbtxt')
    input_arrays = ['a', 'b', 'c', 'd']
    output_arrays = ['Merge']

    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       input_arrays,
                                                       output_arrays)
    with self.assertRaises(ConverterError) as error:
      converter.convert()
    self.assertIn(
        'Failed to functionalize Control Flow V1 ops. Consider using Control '
        'Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/'
        'tf/compat/v1/enable_control_flow_v2.', str(error.exception))


class QuantizationModeTest(LiteTest, parameterized.TestCase):

  @parameterized.named_parameters(
      ('size', lite.Optimize.OPTIMIZE_FOR_SIZE),
      ('latency', lite.Optimize.OPTIMIZE_FOR_LATENCY))
  def testDeprecatedOptionWarning(self, optimization):
    """Test if the warning message when using TOCO is logged."""
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    warning_message = 'please use optimizations=[Optimize.DEFAULT] instead.'
    lite.QuantizationMode([optimization], lite.TargetSpec(), None, None)
    self.assertIn(warning_message, log.getvalue())
    logging.root.removeHandler(handler)


if __name__ == '__main__':
  test.main()
