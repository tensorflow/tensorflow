/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/inject_io_prefetch.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kAutotune[] = "autotune";
constexpr char kFunctionAttrKey[] = "f";
constexpr char kParallelInterleave[] = "ParallelInterleaveDataset";
constexpr char kParallelMap[] = "ParallelMapDataset";
constexpr char kPrefetch[] = "PrefetchDataset";
constexpr std::array<const char*, 5> kAsync = {
    "MapAndBatchDataset", "ParallelBatchDataset", "ParallelInterleaveDataset",
    "ParallelMapDataset", "PrefetchDataset"};
constexpr std::array<const char*, 6> kIo = {
    "ArrayRecordDataset", "FixedLengthRecordDataset", "RecordIODataset",
    "SSTableDataset",     "TextLineDataset",          "TFRecordDataset"};

bool IsAsync(const NodeDef* node) {
  if (!node) {
    return false;
  }
  return absl::c_any_of(kAsync, [&](const char* dataset) {
    return data::MatchesAnyVersion(dataset, node->op());
  });
}

bool IsIo(const NodeDef* node) {
  if (!node) {
    return false;
  }
  return absl::c_any_of(kIo, [&](const char* dataset) {
    return data::MatchesAnyVersion(dataset, node->op());
  });
}

// Returns `true` if `function` performs an IO.
bool IsIo(const FunctionDef& function) {
  for (const auto& node : function.node_def()) {
    if (IsIo(&node)) {
      return true;
    }
  }
  return false;
}

// Returns `true` if `graph` defines a function named `function_name` that
// performs an IO.
bool IsIoFunction(const std::string& function_name,
                  const MutableGraphView& graph) {
  for (const auto& function : graph.graph()->library().function()) {
    if (function.signature().name() == function_name) {
      return IsIo(function);
    }
  }
  return false;
}

// Returns `true` if `node` configures a function that performs an IO.
bool HasIoFunction(const NodeDef* node, const MutableGraphView& graph) {
  if (auto it = node->attr().find(kFunctionAttrKey); it != node->attr().end()) {
    return IsIoFunction(it->second.func().name(), graph);
  }
  return false;
}

// Returns `true` if `node` is a parallel interleave that performs an IO.
bool IsParallelInterleaveWithIo(const NodeDef* node,
                                const MutableGraphView& graph) {
  if (!node || !data::MatchesAnyVersion(kParallelInterleave, node->op())) {
    return false;
  }
  return HasIoFunction(node, graph);
}

bool IsParallelMap(const NodeDef* node) {
  if (!node) {
    return false;
  }
  return data::MatchesAnyVersion(kParallelMap, node->op());
}

bool IsPrefetch(const NodeDef* node) {
  if (!node) {
    return false;
  }
  return node->op() == kPrefetch;
}

// A pair of adjacent nodes `input`->`output`.
struct Edge {
  NodeDef* input;
  NodeDef* output;

  template <typename H>
  friend H AbslHashValue(H h, const Edge& e) {
    return H::combine(std::move(h), e.input, e.output);
  }

  friend bool operator==(const Edge& lhs, const Edge& rhs) {
    return lhs.input == rhs.input && lhs.output == rhs.output;
  }
};

// Inserts a prefetch at `edge` of `graph`.
absl::StatusOr<bool> InjectPrefetch(const Edge& edge, MutableGraphView& graph) {
  NodeDef prefetch;
  graph_utils::SetUniqueGraphNodeName(
      absl::StrCat("inject/io_prefetch", edge.input->name()), graph.graph(),
      &prefetch);
  prefetch.set_op(kPrefetch);
  *prefetch.mutable_input()->Add() = edge.input->name();
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);
  *prefetch.mutable_input()->Add() = autotune_value->name();
  if (!graph_utils::CopyShapesAndTypesAttrs(*edge.input, &prefetch)) {
    return false;
  }
  TF_RETURN_IF_ERROR(graph_utils::SetMetadataName(prefetch.name(), &prefetch));
  NodeDef* added_prefetch = graph.AddNode(std::move(prefetch));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(edge.input->name(), added_prefetch->name()));
  return true;
}

// Searches the input tree of `node` to fill `prefetch_injection_edges` with
// `graph` edges that are eligible for a prefetch injection.
// - `output` is the direct output of `node` in the current path.
// - `output_output` is the direct output of `output` in the current path.
// - `last_async` is the closest async transitive output of `output` in the
// current path.
// - `last_async_output` is the direct output of `last_async` in the current
// path.
// - `last_last_async` is the closest async transitive output of `last_async` in
// the current path.
void GetPrefetchInjectionEdges(
    const MutableGraphView& graph, NodeDef* node, NodeDef* output,
    NodeDef* output_output, NodeDef* last_async, NodeDef* last_async_output,
    NodeDef* last_last_async,
    absl::flat_hash_set<Edge>& prefetch_injection_edges) {
  if (!node) {
    return;
  }
  if (IsAsync(output)) {
    last_last_async = last_async;
    last_async_output = output_output;
    last_async = output;
  }
  if (IsIo(node)) {
    if (IsParallelMap(last_async) && !IsPrefetch(last_last_async)) {
      prefetch_injection_edges.insert({last_async, last_async_output});
    }
    return;
  }
  if (IsParallelInterleaveWithIo(node, graph)) {
    if (!IsPrefetch(last_async)) {
      prefetch_injection_edges.insert({node, output});
    }
    return;
  }
  for (int64_t i = 0; i < node->input_size(); ++i) {
    NodeDef* input = graph_utils::GetInputNode(*node, graph, i);
    GetPrefetchInjectionEdges(graph, /*node=*/input, /*output=*/node,
                              /*output_output=*/output, last_async,
                              last_async_output, last_last_async,
                              prefetch_injection_edges);
  }
}

absl::StatusOr<absl::flat_hash_set<Edge>> GetPrefetchInjectionEdges(
    const GrapplerItem& item, const MutableGraphView& graph) {
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) {
    return absl::flat_hash_set<Edge>();
  }
  if (item.fetch.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected only one fetch node but there were ",
                     item.fetch.size(), ": ", absl::StrJoin(item.fetch, ", ")));
  }
  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  absl::flat_hash_set<Edge> prefetch_injection_edges;
  GetPrefetchInjectionEdges(
      graph, /*node=*/last_node, /*output=*/sink_node,
      /*output_output=*/nullptr,
      /*last_async=*/nullptr, /*last_async_output=*/nullptr,
      /*last_last_async=*/nullptr, prefetch_injection_edges);
  return prefetch_injection_edges;
}

}  // namespace

absl::Status InjectIoPrefetchEligible::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    return absl::OkStatus();
  }
  MutableGraphView graph(output);
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Edge> prefetch_injection_edges,
                      GetPrefetchInjectionEdges(item, graph));
  stats->num_changes += prefetch_injection_edges.size();
  return absl::OkStatus();
}

absl::Status InjectIoPrefetch::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    return absl::OkStatus();
  }
  MutableGraphView graph(output);
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Edge> prefetch_injection_edges,
                      GetPrefetchInjectionEdges(item, graph));
  for (const auto& edge : prefetch_injection_edges) {
    TF_ASSIGN_OR_RETURN(bool success, InjectPrefetch(edge, graph));
    stats->num_changes += success;
  }
  return absl::OkStatus();
}

absl::Status InjectIoPrefetch::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) {
    return absl::OkStatus();
  }
  const std::string& autotune = config->parameter_map().at(kAutotune).s();
  if (autotune == "true") {
    autotune_ = true;
  } else if (autotune == "false") {
    autotune_ = false;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Received an invalid value for parameter ", kAutotune, ": ", autotune));
  }
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(InjectIoPrefetch, "inject_io_prefetch");

}  // namespace grappler
}  // namespace tensorflow
