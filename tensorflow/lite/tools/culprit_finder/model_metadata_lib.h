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

#ifndef TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
#define TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/model_runtime_info.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace tooling {

// ModelMetadata is used to store the metadata of the model and provide methods
// to get the node identifier, tensor identifier, node shapes, output tensors of
// a node, node ids in a range from an execution plan, etc.
class ModelMetadata {
 public:
  ModelMetadata(
      tflite::Interpreter* interpreter,
      const tflite::profiling::ModelRuntimeDetails& model_runtime_details)
      : interpreter_(interpreter) {
    for (const auto& subgraph : model_runtime_details.subgraphs()) {
      // Only interested in the default subgraph (single signature) for now.
      if (subgraph.subgraph_id() == 0) {
        for (const auto& edge : subgraph.edges()) {
          tensor_index_to_edge_proto_[edge.id()] = edge;
        }
        for (const auto& node : subgraph.nodes()) {
          node_index_to_node_proto_[node.id()] = node;
          for (int output : node.outputs()) {
            tensor_index_to_src_nodes_[output] = node.id();
          }
        }
        break;
      }
    }
  };

  ~ModelMetadata() = default;

  static std::optional<std::unique_ptr<ModelMetadata>> Create(
      tflite::Interpreter* interpreter) {
    tflite::profiling::ModelRuntimeDetails model_runtime_details;
    if (tflite::profiling::GenerateModelRuntimeInfo(
            *interpreter, model_runtime_details) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to generate model runtime info";
      return nullptr;
    }
    return std::make_unique<ModelMetadata>(interpreter, model_runtime_details);
  }

  // Returns a vector of output tensor indices for the given node.
  std::vector<int> GetOutputTensorsOfNode(int node_id);

  // Returns a vector of node ids in the range of [start_node, end_node] from
  // the execution plan.
  std::vector<int> GetNodeIdsInRange(int start_node, int end_node);

  // Returns a string representing the node identifier with or without the
  // index.
  // When with_index is False, The identifier is in the format of "<node_name>".
  // When with_index is True, the identifier is in the format of
  // "[<node_name>]:<node_index>".
  std::string GetNodeIdentifier(int node_index, bool with_index = false);

  // Returns a string representing the tensor identifier. The identifier is
  // in the format of "(<node_identifier_of_source_node>)-><tensor_index>".
  // For tensors that are inputs, the source_node_identifier is INPUT.
  std::string GetTensorIdentifier(int tensor_index);

  // Returns a string representing the node shapes. The shapes are in the format
  // of "(<input_shapes>)->(<output_shapes>)".
  // Each tensors shape is in the format of "[<dim_1>,<dim_2>,...]."
  // In case of multiple inputs or outputs, the shapes are separated by commas.
  std::string GetNodeShapes(int node_index);

 private:
  tflite::Interpreter* interpreter_;
  std::unordered_map<int, profiling::Node> node_index_to_node_proto_;
  std::unordered_map<int, profiling::Edge> tensor_index_to_edge_proto_;
  absl::flat_hash_map<int, int> tensor_index_to_src_nodes_;

  // Returns a string representing the shape of the edge. The shape is in the
  // format of "[<dim_1>,<dim_2>,...]."
  static std::string EdgeShapeToString(const profiling::Edge& edge);
};
}  // namespace tooling
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_MODEL_METADATA_LIB_H_
