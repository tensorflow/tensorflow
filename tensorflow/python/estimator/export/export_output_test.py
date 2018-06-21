# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.estimator.export import export_output as export_output_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants


class ExportOutputTest(test.TestCase):

  def test_regress_value_must_be_float(self):
    value = array_ops.placeholder(dtypes.string, 1, name="output-tensor-1")
    with self.assertRaises(ValueError) as e:
      export_output_lib.RegressionOutput(value)
    self.assertEqual('Regression output value must be a float32 Tensor; got '
                     'Tensor("output-tensor-1:0", shape=(1,), dtype=string)',
                     str(e.exception))

  def test_classify_classes_must_be_strings(self):
    classes = array_ops.placeholder(dtypes.float32, 1, name="output-tensor-1")
    with self.assertRaises(ValueError) as e:
      export_output_lib.ClassificationOutput(classes=classes)
    self.assertEqual('Classification classes must be a string Tensor; got '
                     'Tensor("output-tensor-1:0", shape=(1,), dtype=float32)',
                     str(e.exception))

  def test_classify_scores_must_be_float(self):
    scores = array_ops.placeholder(dtypes.string, 1, name="output-tensor-1")
    with self.assertRaises(ValueError) as e:
      export_output_lib.ClassificationOutput(scores=scores)
    self.assertEqual('Classification scores must be a float32 Tensor; got '
                     'Tensor("output-tensor-1:0", shape=(1,), dtype=string)',
                     str(e.exception))

  def test_classify_requires_classes_or_scores(self):
    with self.assertRaises(ValueError) as e:
      export_output_lib.ClassificationOutput()
    self.assertEqual("At least one of scores and classes must be set.",
                     str(e.exception))

  def test_build_standardized_signature_def_regression(self):
    input_tensors = {
        "input-1":
            array_ops.placeholder(
                dtypes.string, 1, name="input-tensor-1")
    }
    value = array_ops.placeholder(dtypes.float32, 1, name="output-tensor-1")

    export_output = export_output_lib.RegressionOutput(value)
    actual_signature_def = export_output.as_signature_def(input_tensors)

    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[
        signature_constants.REGRESS_INPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="input-tensor-1:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.REGRESS_OUTPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-1:0",
                                      dtype=dtype_float,
                                      tensor_shape=shape))

    expected_signature_def.method_name = signature_constants.REGRESS_METHOD_NAME
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_classes_only(self):
    """Tests classification with one output tensor."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(
                dtypes.string, 1, name="input-tensor-1")
    }
    classes = array_ops.placeholder(dtypes.string, 1, name="output-tensor-1")

    export_output = export_output_lib.ClassificationOutput(classes=classes)
    actual_signature_def = export_output.as_signature_def(input_tensors)

    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[
        signature_constants.CLASSIFY_INPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="input-tensor-1:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-1:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_both(self):
    """Tests multiple output tensors that include classes and scores."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(
                dtypes.string, 1, name="input-tensor-1")
    }
    classes = array_ops.placeholder(dtypes.string, 1,
                                    name="output-tensor-classes")
    scores = array_ops.placeholder(dtypes.float32, 1,
                                   name="output-tensor-scores")

    export_output = export_output_lib.ClassificationOutput(
        scores=scores, classes=classes)
    actual_signature_def = export_output.as_signature_def(input_tensors)

    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[
        signature_constants.CLASSIFY_INPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="input-tensor-1:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-classes:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-scores:0",
                                      dtype=dtype_float,
                                      tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_scores_only(self):
    """Tests classification without classes tensor."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(
                dtypes.string, 1, name="input-tensor-1")
    }

    scores = array_ops.placeholder(dtypes.float32, 1,
                                   name="output-tensor-scores")

    export_output = export_output_lib.ClassificationOutput(
        scores=scores)
    actual_signature_def = export_output.as_signature_def(input_tensors)

    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[
        signature_constants.CLASSIFY_INPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="input-tensor-1:0",
                                      dtype=dtype_string,
                                      tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-scores:0",
                                      dtype=dtype_float,
                                      tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_predict_outputs_valid(self):
    """Tests that no errors are raised when provided outputs are valid."""
    outputs = {
        "output0": constant_op.constant([0]),
        u"output1": constant_op.constant(["foo"]),
    }
    export_output_lib.PredictOutput(outputs)

    # Single Tensor is OK too
    export_output_lib.PredictOutput(constant_op.constant([0]))

  def test_predict_outputs_invalid(self):
    with self.assertRaisesRegexp(
        ValueError,
        "Prediction output key must be a string"):
      export_output_lib.PredictOutput({1: constant_op.constant([0])})

    with self.assertRaisesRegexp(
        ValueError,
        "Prediction output value must be a Tensor"):
      export_output_lib.PredictOutput({
          "prediction1": sparse_tensor.SparseTensor(
              indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
      })


class MockSupervisedOutput(export_output_lib._SupervisedOutput):
  """So that we can test the abstract class methods directly."""

  def _get_signature_def_fn(self):
    pass


class SupervisedOutputTest(test.TestCase):

  def test_supervised_outputs_valid(self):
    """Tests that no errors are raised when provided outputs are valid."""
    loss = {"my_loss": constant_op.constant([0])}
    predictions = {u"output1": constant_op.constant(["foo"])}
    metrics = {"metrics": (constant_op.constant([0]),
                           constant_op.constant([10])),
               "metrics2": (constant_op.constant([0]),
                            constant_op.constant([10]))}

    outputter = MockSupervisedOutput(loss, predictions, metrics)
    self.assertEqual(outputter.loss["loss/my_loss"], loss["my_loss"])
    self.assertEqual(
        outputter.predictions["predictions/output1"], predictions["output1"])
    self.assertEqual(outputter.metrics["metrics/value"], metrics["metrics"][0])
    self.assertEqual(
        outputter.metrics["metrics2/update_op"], metrics["metrics2"][1])

    # Single Tensor is OK too
    outputter = MockSupervisedOutput(
        loss["my_loss"], predictions["output1"], metrics["metrics"])
    self.assertEqual(outputter.loss, {"loss": loss["my_loss"]})
    self.assertEqual(
        outputter.predictions, {"predictions": predictions["output1"]})
    self.assertEqual(outputter.metrics["metrics/value"], metrics["metrics"][0])

  def test_supervised_outputs_none(self):
    outputter = MockSupervisedOutput(
        constant_op.constant([0]), None, None)
    self.assertEqual(len(outputter.loss), 1)
    self.assertEqual(outputter.predictions, None)
    self.assertEqual(outputter.metrics, None)

  def test_supervised_outputs_invalid(self):
    with self.assertRaisesRegexp(ValueError, "predictions output value must"):
      MockSupervisedOutput(constant_op.constant([0]), [3], None)
    with self.assertRaisesRegexp(ValueError, "loss output value must"):
      MockSupervisedOutput("str", None, None)
    with self.assertRaisesRegexp(ValueError, "metrics output value must"):
      MockSupervisedOutput(None, None, (15.3, 4))
    with self.assertRaisesRegexp(ValueError, "loss output key must"):
      MockSupervisedOutput({25: "Tensor"}, None, None)

  def test_supervised_outputs_tuples(self):
    """Tests that no errors are raised when provided outputs are valid."""
    loss = {("my", "loss"): constant_op.constant([0])}
    predictions = {(u"output1", "2"): constant_op.constant(["foo"])}
    metrics = {("metrics", "twice"): (constant_op.constant([0]),
                                      constant_op.constant([10]))}

    outputter = MockSupervisedOutput(loss, predictions, metrics)
    self.assertEqual(set(outputter.loss.keys()), set(["loss/my/loss"]))
    self.assertEqual(set(outputter.predictions.keys()),
                     set(["predictions/output1/2"]))
    self.assertEqual(set(outputter.metrics.keys()),
                     set(["metrics/twice/value", "metrics/twice/update_op"]))

  def test_supervised_outputs_no_prepend(self):
    """Tests that no errors are raised when provided outputs are valid."""
    loss = {"loss": constant_op.constant([0])}
    predictions = {u"predictions": constant_op.constant(["foo"])}
    metrics = {u"metrics": (constant_op.constant([0]),
                            constant_op.constant([10]))}

    outputter = MockSupervisedOutput(loss, predictions, metrics)
    self.assertEqual(set(outputter.loss.keys()), set(["loss"]))
    self.assertEqual(set(outputter.predictions.keys()), set(["predictions"]))
    self.assertEqual(set(outputter.metrics.keys()),
                     set(["metrics/value", "metrics/update_op"]))

  def test_train_signature_def(self):
    loss = {"my_loss": constant_op.constant([0])}
    predictions = {u"output1": constant_op.constant(["foo"])}
    metrics = {"metrics": (constant_op.constant([0]),
                           constant_op.constant([10]))}

    outputter = export_output_lib.TrainOutput(loss, predictions, metrics)

    receiver = {u"features": constant_op.constant(100, shape=(100, 2)),
                "labels": constant_op.constant(100, shape=(100, 1))}
    sig_def = outputter.as_signature_def(receiver)

    self.assertTrue("loss/my_loss" in sig_def.outputs)
    self.assertTrue("metrics/value" in sig_def.outputs)
    self.assertTrue("predictions/output1" in sig_def.outputs)
    self.assertTrue("features" in sig_def.inputs)

  def test_eval_signature_def(self):
    loss = {"my_loss": constant_op.constant([0])}
    predictions = {u"output1": constant_op.constant(["foo"])}

    outputter = export_output_lib.EvalOutput(loss, predictions, None)

    receiver = {u"features": constant_op.constant(100, shape=(100, 2)),
                "labels": constant_op.constant(100, shape=(100, 1))}
    sig_def = outputter.as_signature_def(receiver)

    self.assertTrue("loss/my_loss" in sig_def.outputs)
    self.assertFalse("metrics/value" in sig_def.outputs)
    self.assertTrue("predictions/output1" in sig_def.outputs)
    self.assertTrue("features" in sig_def.inputs)

if __name__ == "__main__":
  test.main()
