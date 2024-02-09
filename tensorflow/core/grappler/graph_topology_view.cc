/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/graph_topology_view.h"

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace grappler {

namespace {

template <typename T>
inline void SortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

}  // namespace

Status GraphTopologyView::InitializeFromGraph(
    const GraphDef& graph,
    const absl::Span<const GraphView::Edge> ephemeral_edges,
    bool ignore_control_edges) {
  if (graph_ != nullptr) {
    return errors::InvalidArgument("GraphTopologyView is already initialized.");
  }

  graph_ = &graph;
  num_nodes_ = graph.node_size();
  index_to_node_name_.resize(num_nodes_);
  node_name_to_index_.rehash(num_nodes_);
  fanins_.resize(num_nodes_);
  fanouts_.resize(num_nodes_);

  // Build map from name to index and vice versa.
  for (int node_idx = 0; node_idx < num_nodes_; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    node_name_to_index_.emplace(node.name(), node_idx);
    index_to_node_name_.emplace_back(node.name());
  }

  // 1. Add ephemeral edges to the adjacency lists.
  for (const GraphView::Edge& edge : ephemeral_edges) {
    const auto src = node_name_to_index_.find(edge.src.node->name());
    const bool valid_src = src != node_name_to_index_.end();
    if (!valid_src) {
      const string error_message =
          absl::StrCat("Non-existent src node: ", edge.src.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    const auto dst = node_name_to_index_.find(edge.dst.node->name());
    const bool valid_dst = dst != node_name_to_index_.end();

    if (!valid_dst) {
      const string error_message =
          absl::StrCat("Non-existent dst node: ", edge.dst.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }

    if (valid_dst && valid_src) {
      const int src_idx = src->second;
      const int dst_idx = dst->second;
      if (ignore_control_edges && (src_idx < 0 || dst_idx < 0)) {
        continue;
      }
      fanins_[dst_idx].push_back(src_idx);
      fanouts_[src_idx].push_back(dst_idx);
    }
  }

  // 2. Add graph edges to the adjacency lists.
  for (int node_idx = 0; node_idx < num_nodes_; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    fanins_[node_idx].reserve(node.input_size());

    for (const string& input : node.input()) {
      TensorId tensor = ParseTensorName(input);
      if (ignore_control_edges && IsTensorIdControl(tensor)) {
        continue;
      }
      const auto it = node_name_to_index_.find(tensor.node());
      const bool valid_input = it != node_name_to_index_.end();

      if (!valid_input) {
        const string error_message = absl::StrCat("Non-existent input ", input,
                                                  " in node ", node.name());
        if (skip_invalid_edges_) {
          VLOG(3) << "Skip error: " << error_message;
        } else {
          return errors::InvalidArgument(error_message);
        }
      }

      if (valid_input) {
        const int input_idx = it->second;
        fanins_[node_idx].push_back(input_idx);
        fanouts_[input_idx].push_back(node_idx);
      }
    }

    // Dedup the input list while it's still hot in cache.
    SortAndRemoveDuplicates(&fanins_[node_idx]);
  }

  // Dedup outputs for all the graph nodes.
  for (int node_idx = 0; node_idx < num_nodes_; ++node_idx) {
    SortAndRemoveDuplicates(&fanouts_[node_idx]);
  }

  return absl::OkStatus();
}

Status GraphTopologyView::InitializeFromGraph(
    const GraphDef& graph,
    const absl::Span<const GraphView::Edge> ephemeral_edges) {
  return InitializeFromGraph(graph, ephemeral_edges,
                             /*ignore_control_edges=*/false);
}

Status GraphTopologyView::InitializeFromGraph(const GraphDef& graph,
                                              bool ignore_control_edges) {
  return InitializeFromGraph(graph, absl::Span<GraphView::Edge>(),
                             ignore_control_edges);
}

Status GraphTopologyView::InitializeFromGraph(const GraphDef& graph) {
  return InitializeFromGraph(graph, absl::Span<GraphView::Edge>(),
                             /*ignore_control_edges*/ false);
}

bool GraphTopologyView::HasNode(const absl::string_view node_name) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  const auto it = node_name_to_index_.find(node_name);
  return it != node_name_to_index_.end();
}

const NodeDef* GraphTopologyView::GetNode(
    const absl::string_view node_name) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  const auto it = node_name_to_index_.find(node_name);
  return it == node_name_to_index_.end() ? nullptr : &graph_->node(it->second);
}

const NodeDef* GraphTopologyView::GetNode(int node_idx) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  DCHECK(node_idx >= 0 && node_idx < num_nodes_) << "node_idx is out of range";
  return &graph_->node(node_idx);
}

const absl::optional<int> GraphTopologyView::GetNodeIndex(
    const absl::string_view node_name) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  const auto it = node_name_to_index_.find(node_name);
  DCHECK(it != node_name_to_index_.end()) << "Node doesn't exist in a graph";
  return it == node_name_to_index_.end() ? absl::nullopt
                                         : absl::make_optional(it->second);
}

const absl::optional<int> GraphTopologyView::GetNodeIndex(
    const NodeDef& node) const {
  return GetNodeIndex(node.name());
}

const absl::InlinedVector<int, 4>& GraphTopologyView::GetFanin(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanins_[node_idx] : empty_fanin_;
}

const absl::InlinedVector<int, 2>& GraphTopologyView::GetFanout(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanouts_[node_idx] : empty_fanout_;
}

}  // end namespace grappler
}  // end namespace tensorflow
