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

namespace tensorflow {
namespace swig {
namespace {

Status RunCppShapeInferenceImpl(
    const string& serialized_node_def,
    const std::vector<string>& input_serialized_shapes,
    std::vector<string>* output_tensor_shape_protos) {
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

  std::vector<TensorShapeProto> input_shapes;
  input_shapes.resize(input_serialized_shapes.size());
  for (int i = 0; i < input_serialized_shapes.size(); ++i) {
    if (!input_shapes[i].ParseFromString(input_serialized_shapes[i])) {
      return errors::InvalidArgument(
          "Error parsing shape proto during cpp shape inference");
    }
  }

  tensorflow::shape_inference::InferenceContext c(
      &node, op_reg_data->op_def, {} /* input_shape_strings */, input_shapes,
      {} /* input_tensors */);
  TF_RETURN_IF_ERROR(c.construction_status());
  TF_RETURN_IF_ERROR(op_reg_data->shape_inference_fn(&c));

  // Convert output shapes.
  output_tensor_shape_protos->resize(c.num_outputs());
  TensorShapeProto out;
  for (int i = 0; i < c.num_outputs(); ++i) {
    const shape_inference::Shape* s = c.output(i);
    out.Clear();
    if (c.RankKnown(s)) {
      const int32 rank = c.Rank(s);
      for (int i = 0; i < rank; ++i) {
        const shape_inference::Dimension* d = c.Dim(s, i);
        auto* out_dim = out.add_dim();
        if (c.ValueKnown(d)) {
          out_dim->set_size(c.Value(d));
        } else {
          out_dim->set_size(-1);
        }
      }
    } else {
      out.set_unknown_rank(true);
    }
    CHECK(out.AppendToString(&(*output_tensor_shape_protos)[i]));
  }
  return Status::OK();
}

}  // namespace

std::vector<string> RunCppShapeInference(
    const string& serialized_node_def,
    const std::vector<string>& input_serialized_shapes, TF_Status* out_status) {
  std::vector<string> output_tensor_shape_protos;
  tensorflow::Status status =
      RunCppShapeInferenceImpl(serialized_node_def, input_serialized_shapes,
                               &output_tensor_shape_protos);
  Set_TF_Status_from_Status(out_status, status);
  return status.ok() ? output_tensor_shape_protos : std::vector<string>();
}

}  // namespace swig
}  // namespace tensorflow
