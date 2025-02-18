// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"

#include <set>
#include <string>

#include "absl/strings/string_view.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

using litert::Error;
using litert::Expected;

namespace {

// We store THR names in a global set as a workaround to b/369144429.
std::set<std::string>* ThrNames = new std::set<std::string>();

absl::string_view ThrNodeIdStr(LiteRtDispatchNodeId node_id) {
  auto str = "node_" + std::to_string(node_id);
  auto iter = ThrNames->find(str);
  if (iter == ThrNames->end()) {
    iter = ThrNames->insert(iter, str);
  }
  return *iter;
}

}  // namespace

absl::string_view ThrEdgeIdStr(LiteRtDispatchEdgeId edge_id) {
  auto str = "edge_" + std::to_string(edge_id);
  auto iter = ThrNames->find(str);
  if (iter == ThrNames->end()) {
    iter = ThrNames->insert(iter, str);
  }
  return *iter;
}

litert::Expected<void> LiteRtDispatchGraphT::AddNode(
    LiteRtDispatchNodeId node_id, LiteRtDispatchNodeType node_type) {
  auto thr_graph_add_sq_node = southbound_.api().thr_graph_add_sq_node;
  if (!thr_graph_add_sq_node) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_add_sq_node not found");
  }

  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrNodeType thr_node_type;
  switch (node_type) {
    case kLiteRtDispatchNodeTypeDsp:
      thr_node_type = kThrNodeTypeDsp;
      break;
    case kLiteRtDispatchNodeTypeNpu:
      thr_node_type = kThrNodeTypeNpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected node type: %d", node_type);
      return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected node type");
  }

  if (auto status =
          thr_graph_add_sq_node(thr_graph_, thr_node_id.data(), thr_node_type);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_sq_node failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_add_sq_node failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::AddEdge(
    LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_add_edge = southbound_.api().thr_graph_add_edge;
  if (!thr_graph_add_edge) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_add_edge not found");
  }

  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrEdgeType thr_edge_type = kThrEdgeNoType;
  if (auto status =
          thr_graph_add_edge(thr_graph_, thr_edge_id.data(), thr_edge_type);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_add_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure, "thr_graph_add_edge failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::ConnectNodeInput(
    LiteRtDispatchNodeId node_id, int input_index,
    LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_input =
      southbound_.api().thr_graph_connect_node_input;
  if (!thr_graph_connect_node_input) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_connect_node_input not found");
  }

  int next_input_index = NextNodeInputIndex(node_id);
  if (input_index != next_input_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected input index %d, expected %d",
               input_index, next_input_index);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected input index");
  }

  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_input(thr_graph_, thr_node_id.data(),
                                                 thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_input_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_input_edge failed");
  }

  AddInputEdge(input_index, edge_id);
  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::ConnectNodeOutput(
    LiteRtDispatchNodeId node_id, int output_index,
    LiteRtDispatchEdgeId edge_id) {
  auto thr_graph_connect_node_output =
      southbound_.api().thr_graph_connect_node_output;
  if (!thr_graph_connect_node_output) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_connect_node_output not found");
  }

  int next_output_index = NextNodeOutputIndex(node_id);
  if (output_index != next_output_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected output index %d, expected %d",
               output_index, next_output_index);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected output index");
  }

  auto thr_node_id = ThrNodeIdStr(node_id);
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_connect_node_output(
          thr_graph_, thr_node_id.data(), thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_output_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_output_edge failed");
  }

  AddOutputEdge(output_index, edge_id);
  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::ConnectGraphInput(
    int input_index, LiteRtDispatchEdgeId edge_id) {
  int next_input_index = NextGraphInputIndex();
  if (input_index != next_input_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected input index %d, expected %d",
               input_index, next_input_index);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected input index");
  }

  auto thr_graph_set_input_edge = southbound_.api().thr_graph_set_input_edge;
  if (!thr_graph_set_input_edge) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_input_edge not found");
  }

  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_input_edge(thr_graph_, thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_input_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_input_edge failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::ConnectGraphOutput(
    int output_index, LiteRtDispatchEdgeId edge_id) {
  int next_output_index = NextGraphOutputIndex();
  if (output_index != next_output_index) {
    LITERT_LOG(LITERT_ERROR, "Unexpected output index %d, expected %d",
               output_index, next_output_index);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected output index");
  }

  auto thr_graph_set_output_edge = southbound_.api().thr_graph_set_output_edge;
  if (!thr_graph_set_output_edge) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_output_edge not found");
  }

  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_graph_set_output_edge(thr_graph_, thr_edge_id.data());
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_set_output_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_set_output_edge failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::AssignNodeFunction(
    LiteRtDispatchNodeId node_id, LiteRtDispatchExecutableHandle exec_handle,
    const char* function_name) {
  auto thr_graph_assign_sq = southbound_.api().thr_graph_assign_sq;
  if (!thr_graph_assign_sq) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_assign_sq not found");
  }

  auto thr_node_id = ThrNodeIdStr(node_id);
  ThrSqContainerHandle sq_handle = exec_handle;
  // An empty function name represent no function name being provided and
  // therefore we must pass a nullptr to the call below, otherwise the SB API
  // will expect a model with a signature. See b/378913220.
  const char* function_name_ptr =
      absl::string_view(function_name).empty() ? nullptr : function_name;
  if (auto status = thr_graph_assign_sq(thr_graph_, thr_node_id.data(),
                                        sq_handle, function_name_ptr);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_assign_sq failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_assign_sq failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::AnnotateGraph(const char* key,
                                                           const char* value) {
  auto thr_graph_annotate_graph = southbound_.api().thr_graph_annotate_graph;
  if (!thr_graph_annotate_graph) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_graph not found");
  }

  if (auto status = thr_graph_annotate_graph(thr_graph_, key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_graph failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_graph failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::AnnotateNode(
    LiteRtDispatchNodeId node_id, const char* key, const char* value) {
  auto thr_graph_annotate_node = southbound_.api().thr_graph_annotate_node;
  if (!thr_graph_annotate_node) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_node not found");
  }

  auto thr_node_id = ThrNodeIdStr(node_id);
  if (auto status =
          thr_graph_annotate_node(thr_graph_, thr_node_id.data(), key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_node failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_node failed");
  }

  return {};
}

litert::Expected<void> LiteRtDispatchGraphT::AnnotateEdge(
    LiteRtDispatchEdgeId edge_id, const char* key, const char* value) {
  auto thr_graph_annotate_edge = southbound_.api().thr_graph_annotate_edge;
  if (!thr_graph_annotate_edge) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_edge not found");
  }

  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status =
          thr_graph_annotate_edge(thr_graph_, thr_edge_id.data(), key, value);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_annotate_edge failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_annotate_edge failed");
  }

  return {};
}
