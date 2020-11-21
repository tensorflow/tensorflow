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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.lite.python import lite
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import tracking


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
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
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


if __name__ == '__main__':
  test.main()
