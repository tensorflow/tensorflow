/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/tpu_embedding_rewrite_pass_utils.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

// Adds a new TensorFlow graph node, with the output convention matching most TF
// code rather than the order used by Graph::AddNode().
absl::Status AddNode(const NodeDef& n_def, Node** n, Graph* graph) {
  absl::Status add_node_status;
  *n = graph->AddNode(n_def, &add_node_status);
  return add_node_status;
}

// Replaces one TensorFlow graph node with another (specified by a NodeDef),
// moving all the edges.
absl::Status ReplaceNode(const NodeDef& to_def, Node* from, Node** to,
                         Graph* graph) {
  std::vector<const Edge*> edges;
  VLOG(1) << "Node: " << from->DebugString() << " in_edges";
  for (const Edge* edge : from->in_edges()) {
    VLOG(1) << absl::StrFormat("Edge from [%s:%d] to [%s:%d]",
                               edge->src()->name(), edge->src_output(),
                               edge->dst()->name(), edge->dst_input());
    edges.push_back(edge);
  }
  VLOG(1) << "Node: " << from->DebugString() << " out_edges";
  for (const Edge* edge : from->out_edges()) {
    VLOG(1) << absl::StrFormat("Edge from [%s:%d] to [%s:%d]",
                               edge->src()->name(), edge->src_output(),
                               edge->dst()->name(), edge->dst_input());
    if (edge->dst() != from) {
      // Don't copy self-loops multiple times.
      edges.push_back(edge);
    }
  }

  // Add new node.
  TF_RETURN_IF_ERROR(AddNode(to_def, to, graph));
  // Add edges for the new node.
  for (const Edge* edge : edges) {
    Node* new_edge_src = (edge->src() == from ? *to : edge->src());
    Node* new_edge_dst = (edge->dst() == from ? *to : edge->dst());
    VLOG(1) << absl::StrFormat("Adding new edge from [%s:%d] to [%s:%d]",
                               new_edge_src->name(), edge->src_output(),
                               new_edge_dst->name(), edge->dst_input());
    graph->AddEdge(new_edge_src, edge->src_output(), new_edge_dst,
                   edge->dst_input());
  }

  // Remove edges for the original node.
  for (const Edge* edge : edges) {
    graph->RemoveEdge(edge);
  }
  // Remove original node.
  graph->RemoveNode(from);

  return absl::OkStatus();
}

}  // namespace tensorflow
