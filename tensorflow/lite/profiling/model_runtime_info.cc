/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/model_runtime_info.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_field.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/proto/model_runtime_info.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {

namespace {
Edge::DataType GetEdgeDataTypeFromTfLiteType(TfLiteType type) {
  // LINT.IfChange(EdgeDataTypeTransform)
  if (static_cast<int>(Edge::DataType_MIN) <= static_cast<int>(type) &&
      static_cast<int>(type) <= static_cast<int>(Edge::DataType_MAX)) {
    return static_cast<Edge::DataType>(type);
  }
  // LINT.ThenChange()
  TFLITE_LOG(ERROR) << "Mapping TfLiteType to Edge::DataType failed: " << type;
  return Edge::UNKNOWN_TYPE;
}

TfLiteStatus TfliteIntArrayToRepeatedField(
    const TfLiteIntArray* array, google::protobuf::RepeatedField<int32_t>* repeated_field,
    bool check_for_null = false) {
  if (array == nullptr) {
    return check_for_null ? kTfLiteError : kTfLiteOk;
  }
  repeated_field->Reserve(array->size);
  for (int i = 0; i < array->size; ++i) {
    repeated_field->Add(array->data[i]);
  }
  return kTfLiteOk;
}

TfLiteStatus TfliteTensorToEdge(const TfLiteTensor& tensor, int tensor_index,
                                Edge& edge_proto) {
  edge_proto.set_id(tensor_index);

  const std::string tensor_name =
      tensor.name == nullptr ? "" : std::string(tensor.name);
  edge_proto.set_name(tensor_name);
  edge_proto.set_data_type(GetEdgeDataTypeFromTfLiteType(tensor.type));
  edge_proto.set_size(tensor.bytes);
  edge_proto.set_layout_type(Edge::UNKNOWN);
  edge_proto.set_allocation_type(AllocTypeName(tensor.allocation_type));
  const auto status =
      TfliteIntArrayToRepeatedField(tensor.dims, edge_proto.mutable_shape());

  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to convert tensor.dims to RepeatedField as it "
                         "is null for tensor "
                      << tensor_name << " with index " << tensor_index;
    return status;
  }
  return kTfLiteOk;
}

// Converts a TfLiteNode to a Node proto.
//
// If the node is a delegate node, the type is set to "Delegate/{CustomName}"
// to keep this in sync with the types used in op-profiling.
// If is_node_delegated is true, the node is a TfLite node that has been
// delegated to another node provided by delegated_to_node_id. If
// is_node_delegated is false, delegated_to_node_id is ignored.
TfLiteStatus TfliteNodeToNode(const TfLiteNode& node,
                              const TfLiteRegistration& reg, int node_index,
                              bool is_node_delegated,
                              int32_t delegated_to_node_id, Node& node_proto) {
  node_proto.set_id(node_index);
  if (reg.custom_name != nullptr) {
    node_proto.set_name(reg.custom_name);
    // If this node is delegated, the type is saved as "Delegate/{CustomName}"
    // to keep this in sync with the types used in op-profiling.
    node_proto.set_type((is_node_delegated ? "Delegate/" : "") +
                        std::string(reg.custom_name));
  } else {
    // If this node is not a custom op, the name is set to the builtin op name.
    node_proto.set_name(EnumNamesBuiltinOperator()[reg.builtin_code]);
    node_proto.set_type(std::to_string(reg.builtin_code));
  }

  auto status = TfliteIntArrayToRepeatedField(
      node.inputs, node_proto.mutable_inputs(), /*check_for_null=*/true);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to convert node.inputs to RepeatedField as it "
                         "is null for node "
                      << node_proto.name() << " with index " << node_index;
    return status;
  }
  status = TfliteIntArrayToRepeatedField(
      node.outputs, node_proto.mutable_outputs(), /*check_for_null=*/true);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR)
        << "Failed to convert node.outputs to RepeatedField as it "
           "is null for node "
        << node_proto.name() << " with index " << node_index;
    return status;
  }
  status = TfliteIntArrayToRepeatedField(node.intermediates,
                                         node_proto.mutable_intermediates());
  if (status != kTfLiteOk) {
    return status;
  }
  status = TfliteIntArrayToRepeatedField(node.temporaries,
                                         node_proto.mutable_temporaries());
  if (status != kTfLiteOk) {
    return status;
  }

  if (is_node_delegated) {
    // This node is delegated to another node.
    node_proto.set_delegated_to_node_id(delegated_to_node_id);
  } else if (node.delegate != nullptr) {
    // This node is a delegate node that replaces other TfLite nodes.
    auto delegate_node_details = node_proto.mutable_delegate_node_details();
    delegate_node_details->set_delegate_name(reg.custom_name);
    auto* delegate_params =
        static_cast<TfLiteDelegateParams*>(node.builtin_data);

    status = TfliteIntArrayToRepeatedField(
        delegate_params->nodes_to_replace,
        delegate_node_details->mutable_tflite_node_ids_replaced(),
        /*check_for_null=*/true);
    if (status != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to convert delegate_params->nodes_to_replace"
                           " to RepeatedField as it is null for node "
                        << node_proto.name() << " with index " << node_index;
      return status;
    }
  }

  return kTfLiteOk;
}
}  // namespace
TfLiteStatus GenerateModelRuntimeInfo(const tflite::Interpreter& interpreter,
                                      absl::string_view output_file_path) {
  tflite::profiling::ModelRuntimeDetails model_runtime_details;

  const size_t num_subgraphs = interpreter.subgraphs_size();

  for (int i = 0; i < num_subgraphs; ++i) {
    RuntimeSubgraph* runtime_subgraph = model_runtime_details.add_subgraphs();
    runtime_subgraph->set_subgraph_id(i);
    runtime_subgraph->set_subgraph_type(RuntimeSubgraph::TFLITE_SUBGRAPH);

    const tflite::Subgraph& subgraph = *(interpreter.subgraph(i));
    // Capturing information of all the tensors in this subgraph.
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));

      Edge* edge = runtime_subgraph->add_edges();
      auto status = TfliteTensorToEdge(*tensor, tensor_index, *edge);
      if (status != kTfLiteOk) {
        TFLITE_LOG(ERROR) << "Failed to convert tensor to edge, tensor index: "
                          << tensor_index;
        return status;
      }
    }

    // Iterating over all the nodes in this subgraph.
    const SubgraphDelegationMetadata delegation_metadata =
        GetNodeDelegationMetadata(subgraph);

    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      Node* runtime_node = runtime_subgraph->add_nodes();

      const bool is_node_delegated =
          node.delegate == nullptr &&
          delegation_metadata.is_node_delegated[node_index];

      TfLiteStatus status = TfliteNodeToNode(
          node, reg, node_index, is_node_delegated,
          is_node_delegated ? delegation_metadata.replaced_by_node[node_index]
                            : -1,
          *runtime_node);

      if (status != kTfLiteOk) {
        TFLITE_LOG(ERROR) << "Failed to convert node to runtime node, node "
                             "index: "
                          << node_index;
        return status;
      }
    }

    // Save the execution plan to runtime subgraph.
    runtime_subgraph->mutable_execution_plan()->Add(
        subgraph.execution_plan().begin(), subgraph.execution_plan().end());
  }

  std::ofstream ofs(std::string(output_file_path),
                    std::ios::out | std::ios::binary);
  if (ofs.good()) {
    model_runtime_details.SerializeToOstream(&ofs);
    ofs.close();
  } else {
    TFLITE_LOG(ERROR) << "Failed to open file: " << output_file_path;
    TFLITE_LOG(INFO) << model_runtime_details.DebugString();
  }

  return kTfLiteOk;
}
}  // namespace profiling
}  // namespace tflite
