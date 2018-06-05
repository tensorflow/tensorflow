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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.lite.python import lite_constants
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class FromSessionTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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

  def testQuantization(self):
    in_tensor_1 = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
    in_tensor_2 = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
    out_tensor = array_ops.fake_quant_with_min_max_args(
        in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(
        sess, [in_tensor_1, in_tensor_2], [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {
        'inputA': (0., 1.),
        'inputB': (0., 1.)
    }  # mean, std_dev
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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
    self.assertTrue(output_details[0]['quantization'][0] > 0)  # scale

  def testQuantizationInvalid(self):
    in_tensor_1 = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
    in_tensor_2 = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
    out_tensor = array_ops.fake_quant_with_min_max_args(
        in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(
        sess, [in_tensor_1, in_tensor_2], [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'inputA': (0., 1.)}  # mean, std_dev
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'Quantization input stats are not available for input tensors '
        '\'inputB\'.', str(error.exception))

  def testSizeNoneInvalid(self):
    in_tensor = array_ops.placeholder(dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Test invalid shape. None after 1st dimension.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual('Provide an input shape for input array \'Placeholder\'.',
                     str(error.exception))

  def testBatchSizeInvalid(self):
    in_tensor = array_ops.placeholder(
        shape=[1, None, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Test invalid shape. None after 1st dimension.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'None is only supported in the 1st dimension. Tensor '
        '\'Placeholder\' has invalid shape \'[1, None, 16, 3]\'.',
        str(error.exception))

  def testBatchSizeValid(self):
    in_tensor = array_ops.placeholder(
        shape=[None, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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

  def testFreezeGraph(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    var = variable_scope.get_variable(
        'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + var
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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

  def testGraphviz(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    converter.output_format = lite_constants.GRAPHVIZ_DOT
    graphviz_output = converter.convert()
    self.assertTrue(graphviz_output)

  def testInferenceInputType(self):
    in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.uint8)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    converter.inference_input_type = lite_constants.QUANTIZED_UINT8
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

  def testDefaultRangesStats(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    converter.default_ranges_stats = (0, 6)  # min, max
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertTrue(output_details[0]['quantization'][0] > 0)  # scale


class FromFlatbufferFile(test_util.TensorFlowTestCase):

  def testFloat(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + in_tensor
    sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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

  def testFloatWithShapesArray(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + in_tensor
    sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(
        graph_def_file, ['Placeholder'], ['add'],
        input_shapes={'Placeholder': [1, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())

  def testFreezeGraph(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    var = variable_scope.get_variable(
        'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + var
    sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)

    # Ensure the graph with variables cannot be converted.
    with self.assertRaises(ValueError) as error:
      lite.TocoConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                           ['add'])
    self.assertEqual('Please freeze the graph using freeze_graph.py',
                     str(error.exception))

  def testPbtxt(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + in_tensor
    sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pbtxt')
    write_graph(sess.graph_def, '', graph_def_file, True)

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

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

  def testInvalidFile(self):
    graph_def_file = os.path.join(self.get_temp_dir(), 'invalid_file')
    with gfile.Open(graph_def_file, 'wb') as temp_file:
      temp_file.write('bad data')
      temp_file.flush()

    # Attempts to convert the invalid model.
    with self.assertRaises(ValueError) as error:
      lite.TocoConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                           ['add'])
    self.assertEqual(
        'Unable to parse input file \'{}\'.'.format(graph_def_file),
        str(error.exception))


class FromSavedModelTest(test_util.TensorFlowTestCase):

  def _createSavedModel(self, shape):
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
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
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testNoneBatchSize(self):
    """Test a SavedModel, with None in input tensor's shape."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testOrderInputArrays(self):
    """Test a SavedModel ordering of input arrays."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    converter = lite.TocoConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputB', 'inputA'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSubsetInputArrays(self):
    """Test a SavedModel with a subset of the input array names of the model."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    converter = lite.TocoConverter.from_saved_model(
        saved_model_dir,
        input_arrays=['inputA'],
        input_shapes={'inputA': [1, 16, 16, 3]})

    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check case where input shape is None.
    converter = lite.TocoConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': None})

    tflite_model = converter.convert()
    self.assertTrue(tflite_model)


if __name__ == '__main__':
  test.main()
