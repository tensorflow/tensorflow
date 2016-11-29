# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SignatureDef utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils


class SignatureDefUtilsTest(tf.test.TestCase):

  def testBuildSignatureDef(self):
    x = tf.placeholder(tf.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    inputs = dict()
    inputs["foo-input"] = x_tensor_info

    y = tf.placeholder(tf.float32, name="y")
    y_tensor_info = utils.build_tensor_info(y)
    outputs = dict()
    outputs["foo-output"] = y_tensor_info

    signature_def = signature_def_utils.build_signature_def(
        inputs, outputs, "foo-method-name")
    self.assertEqual("foo-method-name", signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    x_tensor_info_actual = signature_def.inputs["foo-input"]
    self.assertEqual("x:0", x_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info_actual.dtype)
    self.assertEqual(1, len(x_tensor_info_actual.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info_actual.tensor_shape.dim[0].size)

    # Check outputs in signature def.
    self.assertEqual(1, len(signature_def.outputs))
    y_tensor_info_actual = signature_def.outputs["foo-output"]
    self.assertEqual("y:0", y_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_FLOAT, y_tensor_info_actual.dtype)
    self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))

  def testRegressionSignatureDef(self):
    input1 = tf.constant("a", name="input-1")
    output1 = tf.constant("b", name="output-1")
    signature_def = signature_def_utils.regression_signature_def(
        input1, output1)

    self.assertEqual(signature_constants.REGRESS_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    x_tensor_info_actual = (
        signature_def.inputs[signature_constants.REGRESS_INPUTS])
    self.assertEqual("input-1:0", x_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, x_tensor_info_actual.dtype)
    self.assertEqual(0, len(x_tensor_info_actual.tensor_shape.dim))

    # Check outputs in signature def.
    self.assertEqual(1, len(signature_def.outputs))
    y_tensor_info_actual = (
        signature_def.outputs[signature_constants.REGRESS_OUTPUTS])
    self.assertEqual("output-1:0", y_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, y_tensor_info_actual.dtype)
    self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))

  def testClassificationSignatureDef(self):
    input1 = tf.constant("a", name="input-1")
    output1 = tf.constant("b", name="output-1")
    output2 = tf.constant("c", name="output-2")
    signature_def = signature_def_utils.classification_signature_def(
        input1, output1, output2)

    self.assertEqual(signature_constants.CLASSIFY_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    x_tensor_info_actual = (
        signature_def.inputs[signature_constants.CLASSIFY_INPUTS])
    self.assertEqual("input-1:0", x_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, x_tensor_info_actual.dtype)
    self.assertEqual(0, len(x_tensor_info_actual.tensor_shape.dim))

    # Check outputs in signature def.
    self.assertEqual(2, len(signature_def.outputs))
    classes_tensor_info_actual = (
        signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES])
    self.assertEqual("output-1:0", classes_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, classes_tensor_info_actual.dtype)
    self.assertEqual(0, len(classes_tensor_info_actual.tensor_shape.dim))
    scores_tensor_info_actual = (
        signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES])
    self.assertEqual("output-2:0", scores_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, scores_tensor_info_actual.dtype)
    self.assertEqual(0, len(scores_tensor_info_actual.tensor_shape.dim))

  def testPredictionSignatureDef(self):
    input1 = tf.constant("a", name="input-1")
    input2 = tf.constant("b", name="input-2")
    output1 = tf.constant("c", name="output-1")
    output2 = tf.constant("d", name="output-2")
    signature_def = signature_def_utils.predict_signature_def(
        {"input-1": input1, "input-2": input2},
        {"output-1": output1, "output-2": output2})

    self.assertEqual(signature_constants.PREDICT_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(2, len(signature_def.inputs))
    input1_tensor_info_actual = (
        signature_def.inputs["input-1"])
    self.assertEqual("input-1:0", input1_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input1_tensor_info_actual.dtype)
    self.assertEqual(0, len(input1_tensor_info_actual.tensor_shape.dim))
    input2_tensor_info_actual = (
        signature_def.inputs["input-2"])
    self.assertEqual("input-2:0", input2_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input2_tensor_info_actual.dtype)
    self.assertEqual(0, len(input2_tensor_info_actual.tensor_shape.dim))

    # Check outputs in signature def.
    self.assertEqual(2, len(signature_def.outputs))
    output1_tensor_info_actual = (
        signature_def.outputs["output-1"])
    self.assertEqual("output-1:0", output1_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, output1_tensor_info_actual.dtype)
    self.assertEqual(0, len(output1_tensor_info_actual.tensor_shape.dim))
    output2_tensor_info_actual = (
        signature_def.outputs["output-2"])
    self.assertEqual("output-2:0", output2_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, output2_tensor_info_actual.dtype)
    self.assertEqual(0, len(output2_tensor_info_actual.tensor_shape.dim))

if __name__ == "__main__":
  tf.test.main()
