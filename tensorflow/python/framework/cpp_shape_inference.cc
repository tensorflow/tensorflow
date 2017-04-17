/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/python/framework/cpp_shape_inference.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/python/framework/cpp_shape_inference.pb.h"
#include "tensorflow/python/lib/core/py_func.h"

namespace tensorflow {
namespace swig {
namespace {

void ProtoFromShapeHandle(tensorflow::shape_inference::ShapeHandle s,
                          tensorflow::shape_inference::InferenceContext* c,
                          TensorShapeProto* out) {
  if (c->RankKnown(s)) {
    const int32 rank = c->Rank(s);
    for (int i = 0; i < rank; ++i) {
      shape_inference::DimensionHandle d = c->Dim(s, i);
      auto* out_dim = out->add_dim();
      if (c->ValueKnown(d)) {
        out_dim->set_size(c->Value(d));
      } else {
        out_dim->set_size(-1);
      }
    }
  } else {
    out->set_unknown_rank(true);
  }
}

Status RunCppShapeInferenceImpl(
    int graph_def_version, const string& serialized_node_def,
    const std::vector<string>& input_serialized_shapes,
    const std::vector<PyObject*>& input_constant_tensor_values,
    const std::vector<string>& input_constant_tensor_as_shape_values,
    std::vector<string>* output_tensor_shape_protos,
    string* input_tensors_needed_out) {
  tensorflow::NodeDef node;
  if (!node.ParseFromString(serialized_node_def)) {
    return errors::InvalidArgument(
        "Error parsing node_def during cpp shape inference");
  }
  DCHECK_EQ(output_tensor_shape_protos->size(), 0);

  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(node.op(), &op_reg_data));

  if (op_reg_data->shape_inference_fn == nullptr) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node.op(),
        "', did you forget to define it?");
  }

  // Convert input shapes.
  std::vector<TensorShapeProto> input_shapes;
  std::vector<TensorShapeProto> input_handle_shapes;
  std::vector<DataType> input_handle_dtypes;
  input_shapes.resize(input_serialized_shapes.size());
  input_handle_shapes.resize(input_serialized_shapes.size());
  input_handle_dtypes.resize(input_serialized_shapes.size());
  CppShapeInferenceResult tmp;
  for (int i = 0; i < input_serialized_shapes.size(); ++i) {
    tmp.Clear();
    if (!tmp.ParseFromString(input_serialized_shapes[i])) {
      return errors::InvalidArgument(
          "Error parsing shape proto during cpp shape inference");
    }
    input_shapes[i].Swap(tmp.mutable_shape());
    input_handle_dtypes[i] = tmp.handle_dtype();
    input_handle_shapes[i].Swap(tmp.mutable_handle_shape());
  }

  // Convert input tensor values;
  std::vector<Tensor> input_tensor_values(input_constant_tensor_values.size());
  std::vector<const Tensor*> input_tensors;
  for (int i = 0; i < input_constant_tensor_values.size(); ++i) {
    auto* py_val = input_constant_tensor_values[i];
    if (py_val == Py_None) {
      input_tensors.push_back(nullptr);
    } else {
      TF_RETURN_IF_ERROR(
          ConvertNdarrayToTensor(py_val, &input_tensor_values[i]));
      input_tensors.push_back(&input_tensor_values[i]);
    }
  }

  // Convert input tensor-as-shape values;
  std::vector<TensorShapeProto> input_tensor_as_shapes_protos(
      input_constant_tensor_as_shape_values.size());
  for (int i = 0; i < input_constant_tensor_as_shape_values.size(); ++i) {
    if (!input_tensor_as_shapes_protos[i].ParseFromString(
            input_constant_tensor_as_shape_values[i])) {
      return errors::InvalidArgument(
          "Error parsing shape proto during cpp shape inference");
    }
  }

  // Run shape inference.
  tensorflow::shape_inference::InferenceContext c(
      graph_def_version, &node, op_reg_data->op_def, input_shapes,
      input_tensors, input_tensor_as_shapes_protos, input_handle_shapes,
      input_handle_dtypes);
  TF_RETURN_IF_ERROR(c.construction_status());

  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));

  // Convert output shapes.
  output_tensor_shape_protos->resize(c.num_outputs());
  CppShapeInferenceResult out;
  for (int i = 0; i < c.num_outputs(); ++i) {
    out.Clear();
    ProtoFromShapeHandle(c.output(i), &c, out.mutable_shape());
    ProtoFromShapeHandle(c.output_handle_shape(i), &c,
                         out.mutable_handle_shape());
    out.set_handle_dtype(c.output_handle_dtype(i));
    CHECK(out.AppendToString(&(*output_tensor_shape_protos)[i]));
  }

  // Add info about requested inputs.
  CppShapeInferenceInputsNeeded needed;
  for (int i = 0; i < c.num_inputs(); ++i) {
    if (c.requested_input_tensor(i)) {
      needed.add_input_tensors_needed(i);
    }
    if (c.requested_input_tensor_as_partial_shape(i)) {
      needed.add_input_tensors_as_shapes_needed(i);
    }
  }
  *input_tensors_needed_out = needed.SerializeAsString();

  return Status::OK();
}

}  // namespace

std::vector<string> RunCppShapeInference(
    int graph_def_version, const string& serialized_node_def,
    const std::vector<string>& input_serialized_shapes,
    PyObject* input_constant_tensor_values,
    const std::vector<string>& input_constant_tensor_as_shape_values,
    TF_Status* out_status) {
  if (!PyList_Check(input_constant_tensor_values)) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT, "Invalid python value");
    return std::vector<string>();
  }

  std::vector<PyObject*> input_constant_tensor_values_v;
  int cnt = PyList_Size(input_constant_tensor_values);
  for (int i = 0; i < cnt; ++i) {
    input_constant_tensor_values_v.push_back(
        PyList_GetItem(input_constant_tensor_values, i));
  }

  std::vector<string> output;
  string input_tensors_needed_out;
  tensorflow::Status status = RunCppShapeInferenceImpl(
      graph_def_version, serialized_node_def, input_serialized_shapes,
      input_constant_tensor_values_v, input_constant_tensor_as_shape_values,
      &output, &input_tensors_needed_out);

  Set_TF_Status_from_Status(out_status, status);
  if (!status.ok()) {
    return std::vector<string>();
  }
  output.push_back(input_tensors_needed_out);
  return output;
}

}  // namespace swig
}  // namespace tensorflow
