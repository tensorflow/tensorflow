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
"""TensorFlow Lite Python Interface: Sanity check."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import io
import sys

import numpy as np
import six

# Force loaded shared object symbols to be globally visible. This is needed so
# that the interpreter_wrapper, in one .so file, can see the test_registerer,
# in a different .so file. Note that this may already be set by default.
# pylint: disable=g-import-not-at-top
if hasattr(sys, 'setdlopenflags') and hasattr(sys, 'getdlopenflags'):
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.lite.python.testdata import _pywrap_test_registerer as test_registerer
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


class InterpreterCustomOpsTest(test_util.TensorFlowTestCase):

  def testRegisterer(self):
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'),
        custom_op_registerers=['TF_TestRegisterer'])
    self.assertTrue(interpreter._safe_to_run())
    self.assertEqual(test_registerer.get_num_test_registerer_calls(), 1)

  def testRegistererFailure(self):
    bogus_name = 'CompletelyBogusRegistererName'
    with self.assertRaisesRegexp(
        ValueError, 'Looking up symbol \'' + bogus_name + '\' failed'):
      interpreter_wrapper.InterpreterWithCustomOps(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          custom_op_registerers=[bogus_name])


class InterpreterTest(test_util.TensorFlowTestCase):

  def assertQuantizationParamsEqual(self, scales, zero_points,
                                    quantized_dimension, params):
    self.assertAllEqual(scales, params['scales'])
    self.assertAllEqual(zero_points, params['zero_points'])
    self.assertEqual(quantized_dimension, params['quantized_dimension'])

  def testFloat(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual((0.0, 0), input_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, input_details[0]['quantization_parameters'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((0.0, 0), output_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, output_details[0]['quantization_parameters'])

    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    expected_output = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testUint8(self):
    model_path = resource_loader.get_path_to_datafile(
        'testdata/permute_uint8.tflite')
    with io.open(model_path, 'rb') as model_file:
      data = model_file.read()

    interpreter = interpreter_wrapper.Interpreter(model_content=data)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual((1.0, 0), input_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [1.0], [0], 0, input_details[0]['quantization_parameters'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((1.0, 0), output_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [1.0], [0], 0, output_details[0]['quantization_parameters'])

    test_input = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    expected_output = np.array([[4, 3, 2, 1]], dtype=np.uint8)
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    test_input.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testString(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/gather_string.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.string_, input_details[0]['dtype'])
    self.assertTrue(([10] == input_details[0]['shape']).all())
    self.assertEqual((0.0, 0), input_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, input_details[0]['quantization_parameters'])
    self.assertEqual('indices', input_details[1]['name'])
    self.assertEqual(np.int64, input_details[1]['dtype'])
    self.assertTrue(([3] == input_details[1]['shape']).all())
    self.assertEqual((0.0, 0), input_details[1]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, input_details[1]['quantization_parameters'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.string_, output_details[0]['dtype'])
    self.assertTrue(([3] == output_details[0]['shape']).all())
    self.assertEqual((0.0, 0), output_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, output_details[0]['quantization_parameters'])

    test_input = np.array([1, 2, 3], dtype=np.int64)
    interpreter.set_tensor(input_details[1]['index'], test_input)

    test_input = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    expected_output = np.array([b'b', b'c', b'd'])
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testPerChannelParams(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile('testdata/pc_conv.bin'))
    interpreter.allocate_tensors()

    # Tensor index 1 is the weight.
    weight_details = interpreter.get_tensor_details()[1]
    qparams = weight_details['quantization_parameters']
    # Ensure that we retrieve per channel quantization params correctly.
    self.assertEqual(len(qparams['scales']), 128)

  def testDenseTensorAccess(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile('testdata/pc_conv.bin'))
    interpreter.allocate_tensors()
    weight_details = interpreter.get_tensor_details()[1]
    s_params = weight_details['sparsity_parameters']
    self.assertEqual(s_params, {})

  def testSparseTensorAccess(self):
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            '../testdata/sparse_tensor.bin'),
        custom_op_registerers=['TF_TestRegisterer'])
    interpreter.allocate_tensors()

    # Tensor at index 0 is sparse.
    compressed_buffer = interpreter.get_tensor(0)
    # Ensure that the buffer is of correct size and value.
    self.assertEqual(len(compressed_buffer), 12)
    sparse_value = [1, 0, 0, 4, 2, 3, 0, 0, 5, 0, 0, 6]
    self.assertAllEqual(compressed_buffer, sparse_value)

    tensor_details = interpreter.get_tensor_details()[0]
    s_params = tensor_details['sparsity_parameters']

    # Ensure sparsity parameter returned is correct
    self.assertAllEqual(s_params['traversal_order'], [0, 1, 2, 3])
    self.assertAllEqual(s_params['block_map'], [0, 1])
    dense_dim_metadata = {'format': 0, 'dense_size': 2}
    self.assertAllEqual(s_params['dim_metadata'][0], dense_dim_metadata)
    self.assertAllEqual(s_params['dim_metadata'][2], dense_dim_metadata)
    self.assertAllEqual(s_params['dim_metadata'][3], dense_dim_metadata)
    self.assertEqual(s_params['dim_metadata'][1]['format'], 1)
    self.assertAllEqual(s_params['dim_metadata'][1]['array_segments'],
                        [0, 2, 3])
    self.assertAllEqual(s_params['dim_metadata'][1]['array_indices'], [0, 1, 1])


class InterpreterTestErrorPropagation(test_util.TensorFlowTestCase):

  def testInvalidModelContent(self):
    with self.assertRaisesRegexp(ValueError,
                                 'Model provided has model identifier \''):
      interpreter_wrapper.Interpreter(model_content=six.b('garbage'))

  def testInvalidModelFile(self):
    with self.assertRaisesRegexp(
        ValueError, 'Could not open \'totally_invalid_file_name\''):
      interpreter_wrapper.Interpreter(
          model_path='totally_invalid_file_name')

  def testInvokeBeforeReady(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    with self.assertRaisesRegexp(RuntimeError,
                                 'Invoke called on model that is not ready'):
      interpreter.invoke()

  def testInvalidModelFileContent(self):
    with self.assertRaisesRegexp(
        ValueError, '`model_path` or `model_content` must be specified.'):
      interpreter_wrapper.Interpreter(model_path=None, model_content=None)

  def testInvalidIndex(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    interpreter.allocate_tensors()
    # Invalid tensor index passed.
    with self.assertRaisesRegexp(ValueError, 'Tensor with no shape found.'):
      interpreter._get_tensor_details(4)
    with self.assertRaisesRegexp(ValueError, 'Invalid node index'):
      interpreter._get_op_details(4)


class InterpreterTensorAccessorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    self.interpreter.allocate_tensors()
    self.input0 = self.interpreter.get_input_details()[0]['index']
    self.initial_data = np.array([[-1., -2., -3., -4.]], np.float32)

  def testTensorAccessor(self):
    """Check that tensor returns a reference."""
    array_ref = self.interpreter.tensor(self.input0)
    np.copyto(array_ref(), self.initial_data)
    self.assertAllEqual(array_ref(), self.initial_data)
    self.assertAllEqual(
        self.interpreter.get_tensor(self.input0), self.initial_data)

  def testGetTensorAccessor(self):
    """Check that get_tensor returns a copy."""
    self.interpreter.set_tensor(self.input0, self.initial_data)
    array_initial_copy = self.interpreter.get_tensor(self.input0)
    new_value = np.add(1., array_initial_copy)
    self.interpreter.set_tensor(self.input0, new_value)
    self.assertAllEqual(array_initial_copy, self.initial_data)
    self.assertAllEqual(self.interpreter.get_tensor(self.input0), new_value)

  def testBase(self):
    self.assertTrue(self.interpreter._safe_to_run())
    _ = self.interpreter.tensor(self.input0)
    self.assertTrue(self.interpreter._safe_to_run())
    in0 = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    in0b = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    # Now get rid of the buffers so that we can evaluate.
    del in0
    del in0b
    self.assertTrue(self.interpreter._safe_to_run())

  def testBaseProtectsFunctions(self):
    in0 = self.interpreter.tensor(self.input0)()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegexp(
        RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.allocate_tensors()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegexp(
        RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.invoke()
    # Now test that we can run
    del in0  # this is our only buffer reference, so now it is safe to change
    in0safe = self.interpreter.tensor(self.input0)
    _ = self.interpreter.allocate_tensors()
    del in0safe  # make sure in0Safe is held but lint doesn't complain


class InterpreterDelegateTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._delegate_file = resource_loader.get_path_to_datafile(
        'testdata/test_delegate.so')
    self._model_file = resource_loader.get_path_to_datafile(
        'testdata/permute_float.tflite')

    # Load the library to reset the counters.
    library = ctypes.pydll.LoadLibrary(self._delegate_file)
    library.initialize_counters()

  def _TestInterpreter(self, model_path, options=None):
    """Test wrapper function that creates an interpreter with the delegate."""
    delegate = interpreter_wrapper.load_delegate(self._delegate_file, options)
    return interpreter_wrapper.Interpreter(
        model_path=model_path, experimental_delegates=[delegate])

  def testDelegate(self):
    """Tests the delegate creation and destruction."""
    interpreter = self._TestInterpreter(model_path=self._model_file)
    lib = interpreter._delegates[0]._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    del interpreter

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

  def testMultipleInterpreters(self):
    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)

    interpreter_a = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    interpreter_b = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del delegate
    del interpreter_a

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del interpreter_b

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

  def testDestructionOrder(self):
    """Make sure internal _interpreter object is destroyed before delegate."""
    self.skipTest('TODO(b/142136355): fix flakiness and re-enable')
    # Track which order destructions were doned in
    destructions = []
    def register_destruction(x):
      destructions.append(
          x if isinstance(x, str) else six.ensure_text(x, 'utf-8'))
      return 0
    # Make a wrapper for the callback so we can send this to ctypes
    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    # Make an interpreter with the delegate
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'), experimental_delegates=[delegate])

    class InterpreterDestroyCallback(object):

      def __del__(self):
        register_destruction('interpreter')

    interpreter._interpreter.stuff = InterpreterDestroyCallback()
    # Destroy both delegate and interpreter
    library = delegate._library
    prototype = ctypes.CFUNCTYPE(ctypes.c_int, (ctypes.c_char_p))
    library.set_destroy_callback(prototype(register_destruction))
    del delegate
    del interpreter
    library.set_destroy_callback(None)
    # check the interpreter was destroyed before the delegate
    self.assertEqual(destructions, ['interpreter', 'test_delegate'])

  def testOptions(self):
    delegate_a = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate_a._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 0)

    delegate_b = interpreter_wrapper.load_delegate(
        self._delegate_file, options={
            'unused': False,
            'options_counter': 2
        })
    lib = delegate_b._library

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

    del delegate_a
    del delegate_b

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 2)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

  def testFail(self):
    with self.assertRaisesRegexp(
        # Due to exception chaining in PY3, we can't be more specific here and check that
        # the phrase 'Fail argument sent' is present.
        ValueError,
        r'Failed to load delegate from'):
      interpreter_wrapper.load_delegate(
          self._delegate_file, options={'fail': 'fail'})


if __name__ == '__main__':
  test.main()
