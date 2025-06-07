/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/culprit_finder/model_metadata_lib.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"

namespace tflite {
namespace tooling {
std::vector<int> ModelMetadata::GetOutputTensorsOfNode(int node_id) {
  std::vector<int> output_tensors;
  for (int output_tensor_id : node_index_to_node_proto_[node_id].outputs()) {
    output_tensors.push_back(output_tensor_id);
  }
  return output_tensors;
}

std::vector<int> ModelMetadata::GetNodeIdsInRange(int start_node,
                                                  int end_node) {
  std::vector<int> node_ids;
  for (int node_id : interpreter_->execution_plan()) {
    if (node_id >= start_node && node_id <= end_node) {
      node_ids.push_back(node_id);
    }
  }
  return node_ids;
}

std::string ModelMetadata::GetNodeIdentifier(int node_index, bool with_index) {
  auto node_proto = node_index_to_node_proto_[node_index];
  if (with_index) {
    return absl::StrFormat("[%s]:%d", node_proto.name(), node_index);
  }
  return node_proto.name();
}

std::string ModelMetadata::GetTensorIdentifier(int tensor_index) {
  auto tensor_proto = tensor_index_to_edge_proto_[tensor_index];
  if (tensor_index_to_src_nodes_.find(tensor_index) ==
      tensor_index_to_src_nodes_.end()) {
    // This is an input tensor.
    return absl::StrFormat("(INPUT)->%d", tensor_index);
  }
  return absl::StrFormat(
      "(%s)->%d",
      GetNodeIdentifier(tensor_index_to_src_nodes_[tensor_index],
                        /*with_index=*/true),
      tensor_index);
}

std::string ModelMetadata::GetNodeShapes(int node_index) {
  auto node_proto = node_index_to_node_proto_[node_index];
  std::string input_shapes = "";
  for (const auto& input : node_proto.inputs()) {
    auto input_edge = tensor_index_to_edge_proto_[input];
    if (input_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
      continue;
    }
    input_shapes += EdgeShapeToString(input_edge);
    if (input != node_proto.inputs().size() - 1) input_shapes += ",";
  }
  std::string output_shapes = "";
  for (const auto& output : node_proto.outputs()) {
    auto output_edge = tensor_index_to_edge_proto_[output];
    if (output_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
      continue;
    }
    output_shapes += EdgeShapeToString(output_edge);
    if (output != node_proto.outputs().size() - 1) output_shapes += ",";
  }
  return "(" + input_shapes + ") -> (" + output_shapes + ")";
}

std::string ModelMetadata::EdgeShapeToString(
    const tflite::profiling::Edge& edge) {
  std::string shape_string = "";
  for (const auto& shape : edge.shape()) {
    shape_string += std::to_string(shape) + ",";
  }
  return tflite::profiling::Edge::DataType_Name(edge.data_type()) + "[" +
         shape_string + "]";
}
}  // namespace tooling
}  // namespace tflite
