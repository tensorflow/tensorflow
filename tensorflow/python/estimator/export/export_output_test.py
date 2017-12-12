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


if __name__ == "__main__":
  test.main()
