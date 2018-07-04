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
import tempfile
import numpy as np

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.lite.python import lite_constants
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
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

  # TODO(nupurgarg): Verify value of contents in GraphViz.
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

  # TODO(nupurgarg): Verify value of contents in GraphViz.
  def testDumpGraphviz(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure interpreter is able to allocate and check graphviz data.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    num_items_graphviz = len(os.listdir(graphviz_dir))
    self.assertTrue(num_items_graphviz)

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    converter.dump_graphviz_video = True
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure graphviz folder has more data after using video flag.
    num_items_graphviz_video = len(os.listdir(graphviz_dir))
    self.assertTrue(num_items_graphviz_video > num_items_graphviz)

  def testInferenceInputType(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
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
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())

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

  def testQuantizeWeights(self):
    np.random.seed(0)
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
    float_converter = lite.TocoConverter.from_session(sess, [in_tensor_1],
                                                      [out_tensor])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized weights model.
    quantized_weights_converter = lite.TocoConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_weights_converter.quantize_weights = True
    quantized_weights_tflite = quantized_weights_converter.convert()
    self.assertTrue(quantized_weights_tflite)

    # Ensure that the quantized weights tflite model is smaller.
    self.assertTrue(len(quantized_weights_tflite) < len(float_tflite))


class FromFrozenGraphFile(test_util.TensorFlowTestCase):

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
    self.assertEqual('Please freeze the graph using freeze_graph.py.',
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


class FromKerasFile(test_util.TensorFlowTestCase):

  def setUp(self):
    keras.backend.clear_session()

  def _getSequentialModel(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    try:
      fd, keras_file = tempfile.mkstemp('.h5')
      keras.models.save_model(model, keras_file)
    finally:
      os.close(fd)
    return keras_file

  def testSequentialModel(self):
    """Test a Sequential tf.keras model with default inputs."""
    keras_file = self._getSequentialModel()

    converter = lite.TocoConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    os.remove(keras_file)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('time_distributed/Reshape_1', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSequentialModelInputArray(self):
    """Test a Sequential tf.keras model testing input arrays argument."""
    keras_file = self._getSequentialModel()

    # Invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TocoConverter.from_keras_model_file(
          keras_file, input_arrays=['invalid-input'])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

    # Valid input array.
    converter = lite.TocoConverter.from_keras_model_file(
        keras_file, input_arrays=['dense_input'])
    tflite_model = converter.convert()
    os.remove(keras_file)
    self.assertTrue(tflite_model)

  def testSequentialModelInputShape(self):
    """Test a Sequential tf.keras model testing input shapes argument."""
    keras_file = self._getSequentialModel()

    # Passing in shape of invalid input array has no impact as long as all input
    # arrays have a shape.
    converter = lite.TocoConverter.from_keras_model_file(
        keras_file, input_shapes={'invalid-input': [2, 3]})
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Passing in shape of valid input array.
    converter = lite.TocoConverter.from_keras_model_file(
        keras_file, input_shapes={'dense_input': [2, 3]})
    tflite_model = converter.convert()
    os.remove(keras_file)
    self.assertTrue(tflite_model)

    # Check input shape from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertTrue(([2, 3] == input_details[0]['shape']).all())

  def testSequentialModelOutputArray(self):
    """Test a Sequential tf.keras model testing output arrays argument."""
    keras_file = self._getSequentialModel()

    # Invalid output array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TocoConverter.from_keras_model_file(
          keras_file, output_arrays=['invalid-output'])
    self.assertEqual("Invalid tensors 'invalid-output' were found.",
                     str(error.exception))

    # Valid output array.
    converter = lite.TocoConverter.from_keras_model_file(
        keras_file, output_arrays=['time_distributed/Reshape_1'])
    tflite_model = converter.convert()
    os.remove(keras_file)
    self.assertTrue(tflite_model)

  def testFunctionalModel(self):
    """Test a Functional tf.keras model with default inputs."""
    inputs = keras.layers.Input(shape=(3,), name='input')
    x = keras.layers.Dense(2)(inputs)
    output = keras.layers.Dense(3)(x)

    model = keras.models.Model(inputs, output)
    model.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    model.predict(x)
    fd, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)

    # Convert to TFLite model.
    converter = lite.TocoConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    os.close(fd)
    os.remove(keras_file)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('dense_1/BiasAdd', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFunctionalModelMultipleInputs(self):
    """Test a Functional tf.keras model with multiple inputs and outputs."""
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')
    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    model.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.mae],
        loss_weights=[1., 0.5])

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    model.predict([input_a_np, input_b_np], batch_size=5)
    fd, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)

    # Convert to TFLite model.
    converter = lite.TocoConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    os.close(fd)
    os.remove(keras_file)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('input_a', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('input_b', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(2, len(output_details))
    self.assertEqual('dense_1/BiasAdd', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual('dropout/Identity', output_details[1]['name'])
    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertTrue(([1, 4] == output_details[1]['shape']).all())
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testFunctionalSequentialModel(self):
    """Test a Functional tf.keras model containing a Sequential model."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model = keras.models.Model(model.input, model.output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    model.predict(x)
    fd, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)

    # Convert to TFLite model.
    converter = lite.TocoConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    os.close(fd)
    os.remove(keras_file)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('time_distributed/Reshape_1', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])


if __name__ == '__main__':
  test.main()
