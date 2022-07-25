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
"""Tests for lite.py functionality related to select TF op usage."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.lite.python import lite
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.testdata import double_op
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import autotrackable


class FromSessionTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testFlexMode(self, enable_mlir):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[1, 4], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testFlexWithAutomaticPassThrough(self):
    # Create a graph that has one L2Loss op.
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[4], dtype=dtypes.float32, name='input')
        out_tensor = nn_ops.l2_loss(in_tensor)
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                      [out_tensor])
        converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
        converter._experimental_allow_all_select_tf_ops = True
        tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexL2Loss', tflite_test_util.get_ops_list(tflite_model))

  def testDeprecatedFlags(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[1, 4], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.target_ops = set([lite.OpsSet.SELECT_TF_OPS])

    # Ensure `target_ops` is set to the correct value after flag deprecation.
    self.assertEqual(converter.target_ops, set([lite.OpsSet.SELECT_TF_OPS]))
    self.assertEqual(converter.target_spec.supported_ops,
                     set([lite.OpsSet.SELECT_TF_OPS]))

    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


class FromConcreteFunctionTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  @test_util.run_v2_only
  def testFloat(self, enable_mlir):
    input_data = constant_op.constant(1., shape=[1])
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([4.0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([24.0], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  # Ensure that input TFLite buffer is not reused for ops such as
  # `TensorListSetItem`. The example model has a while loop, and the while body
  # has a `TensorListSetItem` op which takes the output from a `Where` op.
  @test_util.run_v2_only
  def testDisableFlexTensorMemoryReusing(self):

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[2, 3], dtype=tf.float32, name='x')
    ])
    def model(x):
      l = list_ops.tensor_list_reserve(
          element_dtype=tf.int64, element_shape=[None, 1], num_elements=2)
      init_state = (0, x, l)
      condition = lambda i, x, l: i < 2

      def body(i, x, l):
        element = tf.where(x[i])
        l = list_ops.tensor_list_set_item(l, i, element)
        return i + 1, x, l

      _, _, l_final = tf.while_loop(condition, body, init_state)
      return list_ops.tensor_list_stack(l_final, element_dtype=tf.int64)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [model.get_concrete_function()])
    converter.target_spec.supported_ops = set(
        [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS])
    tflite_model = converter.convert()

    # Check the model produces correct result.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 0.0], [0.0, 5.0, 6.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([0, 1, 1, 2], dtype=np.int64)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == np.ndarray.flatten(output_data)).all())


class WithCustomOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def _createGraphWithCustomOp(self, opname='CustomAdd'):
    custom_opdefs_str = (
        'name: \'' + opname + '\' input_arg: {name: \'Input1\' type: DT_FLOAT} '
        'input_arg: {name: \'Input2\' type: DT_FLOAT} output_arg: {name: '
        '\'Output\' type: DT_FLOAT}')

    # Create a graph that has one add op.
    new_graph = graph_pb2.GraphDef()
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 16, 16, 3], dtype=dtypes.float32, name='input')
        out_tensor = in_tensor + in_tensor
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}

        new_graph.CopyFrom(sess.graph_def)

    # Rename Add op name to opname.
    for node in new_graph.node:
      if node.op.startswith('Add'):
        node.op = opname
        del node.attr['T']

    # Register custom op defs to import modified graph def.
    register_custom_opdefs([custom_opdefs_str])

    return (new_graph, inputs, outputs)

  def testFlexWithCustomOp(self):
    new_graph, inputs, outputs = self._createGraphWithCustomOp(
        opname='CustomAdd4')

    # Import to load the custom opdef.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        import_graph_def(new_graph, name='')
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.target_spec.experimental_select_user_tf_ops = ['CustomAdd4']
    tflite_model = converter.convert()

    self.assertIn('FlexCustomAdd4', tflite_test_util.get_ops_list(tflite_model))

  def testFlexWithDoubleOp(self):
    # Create a graph that has one double op.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'model2')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 4], dtype=dtypes.int32, name='input')
        out_tensor = double_op.double(in_tensor)
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.target_spec.experimental_select_user_tf_ops = ['Double']
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexDouble', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.int32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.int32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


class FromSavedModelTest(test_util.TensorFlowTestCase):

  @test_util.run_v2_only
  def testFlexResourceVariables(self):

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable([[0.0, 0.0, 0.0, 0.0]])

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[1, 4], dtype=tf.float32)])
      def eval(self, x):
        # Control flow is needed to generate "FlexReadVariableOp".
        if tf.reduce_mean(x) > 1.0:
          self.v.assign_add([[1.0, 1.0, 1.0, 1.0]])
        return self.v + x

    m = Model()
    to_save = m.eval.get_concrete_function()
    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    tf.saved_model.save(m, save_dir, to_save)
    converter = tf.lite.TFLiteConverter.from_saved_model(save_dir)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    signature_runner = interpreter.get_signature_runner()
    outputs = signature_runner(
        x=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    expected_output = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    self.assertTrue((expected_output == list(outputs.values())[0]).all)

    # Second run.
    outputs = signature_runner(
        x=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    expected_output = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    self.assertTrue((expected_output == list(outputs.values())[0]).all)


class TFQuantizationTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testAddOp(self, tf_quantization_mode):
    root = autotrackable.AutoTrackable()
    root.add_func = def_function.function(lambda x: x + x)
    input_data = tf.reshape(tf.range(4, dtype=tf.float32), [1, 4])
    concrete_func = root.add_func.get_concrete_function(input_data)

    # Convert model and check if the op is not flex.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    if tf_quantization_mode == 'LEGACY_INTEGER':
      self.assertIn('ADD', tflite_test_util.get_ops_list(tflite_model))
    else:
      self.assertIn('FlexAddV2', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testL2LossOp(self, tf_quantization_mode):
    root = autotrackable.AutoTrackable()
    root.l2_loss_func = def_function.function(lambda x: nn_ops.l2_loss(x))  # pylint: disable=unnecessary-lambda
    input_data = tf.range(4, dtype=tf.float32)
    concrete_func = root.l2_loss_func.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexL2Loss', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([15.0], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testConvOpWithBias(self, tf_quantization_mode):

    class ConvModel(autotrackable.AutoTrackable):

      @def_function.function
      def conv_func(self, in_tensor, filter_tensor):
        bias = constant_op.constant(3., shape=[1])
        conv_tensor = tf.nn.conv2d(
            in_tensor,
            filter_tensor,
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC')
        conv_tensor = conv_tensor + bias
        return tf.nn.relu(conv_tensor)

    root = ConvModel()
    input_data = tf.reshape(tf.range(4, dtype=tf.float32), [1, 2, 2, 1])
    filter_data = tf.reshape(tf.range(2, dtype=tf.float32), [1, 2, 1, 1])
    concrete_func = root.conv_func.get_concrete_function(
        input_data, filter_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertCountEqual(['CONV_2D', 'RESHAPE'],
                          tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(
        (1, 2, 2, 1))
    interpreter.set_tensor(input_details[0]['index'], test_input)
    test_filter = np.array([1.0, 0.0], dtype=np.float32).reshape((1, 2, 1, 1))
    interpreter.set_tensor(input_details[1]['index'], test_filter)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[[[4.]], [[6.]]]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


if __name__ == '__main__':
  test.main()
