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

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import utils


# We'll reuse the same tensor_infos in multiple contexts just for the tests.
# The validator doesn't check shapes so we just omit them.
_STRING = meta_graph_pb2.TensorInfo(
    name="foobar",
    dtype=dtypes.string.as_datatype_enum
)


_FLOAT = meta_graph_pb2.TensorInfo(
    name="foobar",
    dtype=dtypes.float32.as_datatype_enum
)


def _make_signature(inputs, outputs, name=None):
  input_info = {
      input_name: utils.build_tensor_info(tensor)
      for input_name, tensor in inputs.items()
  }
  output_info = {
      output_name: utils.build_tensor_info(tensor)
      for output_name, tensor in outputs.items()
  }
  return signature_def_utils_impl.build_signature_def(input_info, output_info,
                                                      name)


class SignatureDefUtilsTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBuildSignatureDef(self):
    x = array_ops.placeholder(dtypes.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    inputs = dict()
    inputs["foo-input"] = x_tensor_info

    y = array_ops.placeholder(dtypes.float32, name="y")
    y_tensor_info = utils.build_tensor_info(y)
    outputs = dict()
    outputs["foo-output"] = y_tensor_info

    signature_def = signature_def_utils_impl.build_signature_def(
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

  @test_util.run_deprecated_v1
  def testRegressionSignatureDef(self):
    input1 = constant_op.constant("a", name="input-1")
    output1 = constant_op.constant(2.2, name="output-1")
    signature_def = signature_def_utils_impl.regression_signature_def(
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
    self.assertEqual(types_pb2.DT_FLOAT, y_tensor_info_actual.dtype)
    self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))

  @test_util.run_deprecated_v1
  def testClassificationSignatureDef(self):
    input1 = constant_op.constant("a", name="input-1")
    output1 = constant_op.constant("b", name="output-1")
    output2 = constant_op.constant(3.3, name="output-2")
    signature_def = signature_def_utils_impl.classification_signature_def(
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
    self.assertEqual(types_pb2.DT_FLOAT, scores_tensor_info_actual.dtype)
    self.assertEqual(0, len(scores_tensor_info_actual.tensor_shape.dim))

  @test_util.run_deprecated_v1
  def testPredictionSignatureDef(self):
    input1 = constant_op.constant("a", name="input-1")
    input2 = constant_op.constant("b", name="input-2")
    output1 = constant_op.constant("c", name="output-1")
    output2 = constant_op.constant("d", name="output-2")
    signature_def = signature_def_utils_impl.predict_signature_def({
        "input-1": input1,
        "input-2": input2
    }, {"output-1": output1,
        "output-2": output2})

    self.assertEqual(signature_constants.PREDICT_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(2, len(signature_def.inputs))
    input1_tensor_info_actual = (signature_def.inputs["input-1"])
    self.assertEqual("input-1:0", input1_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input1_tensor_info_actual.dtype)
    self.assertEqual(0, len(input1_tensor_info_actual.tensor_shape.dim))
    input2_tensor_info_actual = (signature_def.inputs["input-2"])
    self.assertEqual("input-2:0", input2_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input2_tensor_info_actual.dtype)
    self.assertEqual(0, len(input2_tensor_info_actual.tensor_shape.dim))

    # Check outputs in signature def.
    self.assertEqual(2, len(signature_def.outputs))
    output1_tensor_info_actual = (signature_def.outputs["output-1"])
    self.assertEqual("output-1:0", output1_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, output1_tensor_info_actual.dtype)
    self.assertEqual(0, len(output1_tensor_info_actual.tensor_shape.dim))
    output2_tensor_info_actual = (signature_def.outputs["output-2"])
    self.assertEqual("output-2:0", output2_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, output2_tensor_info_actual.dtype)
    self.assertEqual(0, len(output2_tensor_info_actual.tensor_shape.dim))

  @test_util.run_deprecated_v1
  def testTrainSignatureDef(self):
    self._testSupervisedSignatureDef(
        signature_def_utils_impl.supervised_train_signature_def,
        signature_constants.SUPERVISED_TRAIN_METHOD_NAME)

  @test_util.run_deprecated_v1
  def testEvalSignatureDef(self):
    self._testSupervisedSignatureDef(
        signature_def_utils_impl.supervised_eval_signature_def,
        signature_constants.SUPERVISED_EVAL_METHOD_NAME)

  def _testSupervisedSignatureDef(self, fn_to_test, method_name):
    inputs = {
        "input-1": constant_op.constant("a", name="input-1"),
        "input-2": constant_op.constant("b", name="input-2"),
    }
    loss = {"loss-1": constant_op.constant(0.45, name="loss-1")}
    predictions = {
        "classes": constant_op.constant([100], name="classes"),
    }
    metrics_val = constant_op.constant(100.0, name="metrics_val")
    metrics = {
        "metrics/value": metrics_val,
        "metrics/update_op": array_ops.identity(metrics_val, name="metrics_op"),
    }

    signature_def = fn_to_test(inputs, loss, predictions, metrics)

    self.assertEqual(method_name, signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(2, len(signature_def.inputs))
    input1_tensor_info_actual = (signature_def.inputs["input-1"])
    self.assertEqual("input-1:0", input1_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input1_tensor_info_actual.dtype)
    self.assertEqual(0, len(input1_tensor_info_actual.tensor_shape.dim))
    input2_tensor_info_actual = (signature_def.inputs["input-2"])
    self.assertEqual("input-2:0", input2_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_STRING, input2_tensor_info_actual.dtype)
    self.assertEqual(0, len(input2_tensor_info_actual.tensor_shape.dim))

    # Check outputs in signature def.
    self.assertEqual(4, len(signature_def.outputs))
    self.assertEqual("loss-1:0", signature_def.outputs["loss-1"].name)
    self.assertEqual(types_pb2.DT_FLOAT, signature_def.outputs["loss-1"].dtype)

    self.assertEqual("classes:0", signature_def.outputs["classes"].name)
    self.assertEqual(1, len(signature_def.outputs["classes"].tensor_shape.dim))

    self.assertEqual(
        "metrics_val:0", signature_def.outputs["metrics/value"].name)
    self.assertEqual(
        types_pb2.DT_FLOAT, signature_def.outputs["metrics/value"].dtype)

    self.assertEqual(
        "metrics_op:0", signature_def.outputs["metrics/update_op"].name)
    self.assertEqual(
        types_pb2.DT_FLOAT, signature_def.outputs["metrics/value"].dtype)

  @test_util.run_deprecated_v1
  def testTrainSignatureDefMissingInputs(self):
    self._testSupervisedSignatureDefMissingInputs(
        signature_def_utils_impl.supervised_train_signature_def,
        signature_constants.SUPERVISED_TRAIN_METHOD_NAME)

  @test_util.run_deprecated_v1
  def testEvalSignatureDefMissingInputs(self):
    self._testSupervisedSignatureDefMissingInputs(
        signature_def_utils_impl.supervised_eval_signature_def,
        signature_constants.SUPERVISED_EVAL_METHOD_NAME)

  def _testSupervisedSignatureDefMissingInputs(self, fn_to_test, method_name):
    inputs = {
        "input-1": constant_op.constant("a", name="input-1"),
        "input-2": constant_op.constant("b", name="input-2"),
    }
    loss = {"loss-1": constant_op.constant(0.45, name="loss-1")}
    predictions = {
        "classes": constant_op.constant([100], name="classes"),
    }
    metrics_val = constant_op.constant(100, name="metrics_val")
    metrics = {
        "metrics/value": metrics_val,
        "metrics/update_op": array_ops.identity(metrics_val, name="metrics_op"),
    }

    with self.assertRaises(ValueError):
      signature_def = fn_to_test(
          {}, loss=loss, predictions=predictions, metrics=metrics)

    signature_def = fn_to_test(inputs, loss=loss)
    self.assertEqual(method_name, signature_def.method_name)
    self.assertEqual(1, len(signature_def.outputs))

    signature_def = fn_to_test(inputs, metrics=metrics, loss=loss)
    self.assertEqual(method_name, signature_def.method_name)
    self.assertEqual(3, len(signature_def.outputs))

  def _assertValidSignature(self, inputs, outputs, method_name):
    signature_def = signature_def_utils_impl.build_signature_def(
        inputs, outputs, method_name)
    self.assertTrue(
        signature_def_utils_impl.is_valid_signature(signature_def))

  def _assertInvalidSignature(self, inputs, outputs, method_name):
    signature_def = signature_def_utils_impl.build_signature_def(
        inputs, outputs, method_name)
    self.assertFalse(
        signature_def_utils_impl.is_valid_signature(signature_def))

  def testValidSignaturesAreAccepted(self):
    self._assertValidSignature(
        {"inputs": _STRING},
        {"classes": _STRING, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertValidSignature(
        {"inputs": _STRING},
        {"classes": _STRING},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertValidSignature(
        {"inputs": _STRING},
        {"scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertValidSignature(
        {"inputs": _STRING},
        {"outputs": _FLOAT},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertValidSignature(
        {"foo": _STRING, "bar": _FLOAT},
        {"baz": _STRING, "qux": _FLOAT},
        signature_constants.PREDICT_METHOD_NAME)

  def testInvalidMethodNameSignatureIsRejected(self):
    # WRONG METHOD
    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"classes": _STRING, "scores": _FLOAT},
        "WRONG method name")

  def testInvalidClassificationSignaturesAreRejected(self):
    # CLASSIFY: wrong types
    self._assertInvalidSignature(
        {"inputs": _FLOAT},
        {"classes": _STRING, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"classes": _FLOAT, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"classes": _STRING, "scores": _STRING},
        signature_constants.CLASSIFY_METHOD_NAME)

    # CLASSIFY: wrong keys
    self._assertInvalidSignature(
        {},
        {"classes": _STRING, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs_WRONG": _STRING},
        {"classes": _STRING, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"classes_WRONG": _STRING, "scores": _FLOAT},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {},
        signature_constants.CLASSIFY_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"classes": _STRING, "scores": _FLOAT, "extra_WRONG": _STRING},
        signature_constants.CLASSIFY_METHOD_NAME)

  def testInvalidRegressionSignaturesAreRejected(self):
    # REGRESS: wrong types
    self._assertInvalidSignature(
        {"inputs": _FLOAT},
        {"outputs": _FLOAT},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"outputs": _STRING},
        signature_constants.REGRESS_METHOD_NAME)

    # REGRESS: wrong keys
    self._assertInvalidSignature(
        {},
        {"outputs": _FLOAT},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs_WRONG": _STRING},
        {"outputs": _FLOAT},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"outputs_WRONG": _FLOAT},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {},
        signature_constants.REGRESS_METHOD_NAME)

    self._assertInvalidSignature(
        {"inputs": _STRING},
        {"outputs": _FLOAT, "extra_WRONG": _STRING},
        signature_constants.REGRESS_METHOD_NAME)

  def testInvalidPredictSignaturesAreRejected(self):
    # PREDICT: wrong keys
    self._assertInvalidSignature(
        {},
        {"baz": _STRING, "qux": _FLOAT},
        signature_constants.PREDICT_METHOD_NAME)

    self._assertInvalidSignature(
        {"foo": _STRING, "bar": _FLOAT},
        {},
        signature_constants.PREDICT_METHOD_NAME)

  @test_util.run_v1_only("b/120545219")
  def testOpSignatureDef(self):
    key = "adding_1_and_2_key"
    add_op = math_ops.add(1, 2, name="adding_1_and_2")
    signature_def = signature_def_utils_impl.op_signature_def(add_op, key)
    self.assertIn(key, signature_def.outputs)
    self.assertEqual(add_op.name, signature_def.outputs[key].name)

  @test_util.run_v1_only("b/120545219")
  def testLoadOpFromSignatureDef(self):
    key = "adding_1_and_2_key"
    add_op = math_ops.add(1, 2, name="adding_1_and_2")
    signature_def = signature_def_utils_impl.op_signature_def(add_op, key)

    self.assertEqual(
        add_op,
        signature_def_utils_impl.load_op_from_signature_def(signature_def, key))


if __name__ == "__main__":
  test.main()
