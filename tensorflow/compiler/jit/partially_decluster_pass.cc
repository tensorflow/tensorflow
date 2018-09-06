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

#include "tensorflow/compiler/jit/partially_decluster_pass.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace {
Status FindNodesToDecluster(const Graph& graph, gtl::FlatSet<Node*>* result,
                            absl::Span<Node* const> post_order) {
  // Find nodes that have at least one user outside their cluster that expects
  // hostmem output.  These nodes should be cloned to outside the cluster to
  // avoid the device-host copy we'd otherwise need.

  MemoryTypeVector input_mtypes, output_mtypes;

  for (Node* n : post_order) {
    absl::optional<absl::string_view> from_cluster = GetXlaClusterForNode(*n);
    if (!from_cluster) {
      continue;
    }

    // We assume the only XLA-auto-clusterable operations with side effects are
    // resource variable updates.  We can't execute these twice.
    if (HasResourceInputOrOutput(*n)) {
      continue;
    }

    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceToDeviceType(n->assigned_device_name(), &device_type));
    TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                          n->def(), &input_mtypes,
                                          &output_mtypes));
    for (const Edge* e : n->out_edges()) {
      Node* dst = e->dst();

      if (e->IsControlEdge()) {
        continue;
      }

      bool edge_incurs_extra_device_to_host_copy;
      if (output_mtypes[e->src_output()] == DEVICE_MEMORY) {
        // If the output of the *TensorFlow* operation is in DEVICE_MEMORY then
        // keep the node clustered -- XLA will also produce the output in device
        // memory and we will get some benefit from clustering.
        edge_incurs_extra_device_to_host_copy = false;
      } else {
        MemoryTypeVector dst_input_mtypes, dst_output_mtypes;
        DeviceType dst_device_type("");
        TF_RETURN_IF_ERROR(
            DeviceToDeviceType(dst->assigned_device_name(), &dst_device_type));
        TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                              dst->def(), &dst_input_mtypes,
                                              &dst_output_mtypes));
        edge_incurs_extra_device_to_host_copy =
            dst_input_mtypes[e->dst_input()] == HOST_MEMORY;
      }

      if (!edge_incurs_extra_device_to_host_copy) {
        continue;
      }

      // Check if `dst` is in a different cluster, unclustered, or about to be
      // partially declustered (here we rely on the post-order traversal order).
      // If yes, decluster `n` to avoid the device-to-host memcpy.
      absl::optional<absl::string_view> dst_cluster =
          result->count(dst) ? absl::nullopt : GetXlaClusterForNode(*dst);
      if (from_cluster != dst_cluster) {
        CHECK(result->insert(n).second);
        break;
      }
    }
  }
  return Status::OK();
}

Status PartiallyDeclusterNode(Graph* graph, Node* n) {
  absl::string_view cluster_name = *GetXlaClusterForNode(*n);
  absl::InlinedVector<const Edge*, 6> out_edges_to_clone;
  for (const Edge* out_edge : n->out_edges()) {
    if (out_edge->IsControlEdge()) {
      continue;
    }

    Node* dst = out_edge->dst();
    absl::optional<absl::string_view> dst_cluster_name =
        GetXlaClusterForNode(*dst);
    if (dst_cluster_name != cluster_name) {
      out_edges_to_clone.push_back(out_edge);
    }
  }

  CHECK(!out_edges_to_clone.empty()) << n->DebugString();

  NodeDef ndef = n->def();
  ndef.set_name(absl::StrCat(n->name(), "/declustered"));
  RemoveFromXlaCluster(&ndef);
  Status s;
  Node* cloned_node = graph->AddNode(ndef, &s);
  cloned_node->set_assigned_device_name(n->assigned_device_name());
  TF_RETURN_IF_ERROR(s);

  for (const Edge* in_edge : n->in_edges()) {
    graph->AddEdge(in_edge->src(), in_edge->src_output(), cloned_node,
                   in_edge->dst_input());
  }

  for (const Edge* out_edge_to_clone : out_edges_to_clone) {
    graph->AddEdge(cloned_node, out_edge_to_clone->src_output(),
                   out_edge_to_clone->dst(), out_edge_to_clone->dst_input());
    graph->RemoveEdge(out_edge_to_clone);
  }

  return Status::OK();
}
}  // namespace

Status PartiallyDeclusterPass::Run(
    const GraphOptimizationPassOptions& options) {
  // NB!  In this pass we assume the only XLA-auto-clusterable operations that
  // may have side effects are resource variable operations so we don't cluster
  // those.  The pass will have to be updated if this assumption becomes
  // invalid.

  Graph* graph = options.graph->get();

  // When deciding whether to decluster a particular node, we base our decision
  // on if we've decided that some of its consumers have to be declustered too.
  // Iterating the graph in post-order guarantees that consumers have been
  // visited before producers.
  std::vector<Node*> post_order;
  GetPostOrder(*graph, &post_order, /*stable_comparator=*/NodeComparatorName(),
               /*edge_filter=*/[](const Edge& edge) {
                 return !edge.src()->IsNextIteration();
               });

  gtl::FlatSet<Node*> nodes_to_partially_decluster;
  TF_RETURN_IF_ERROR(FindNodesToDecluster(
      **options.graph, &nodes_to_partially_decluster, post_order));

  if (VLOG_IS_ON(3)) {
    for (Node* n : post_order) {
      if (nodes_to_partially_decluster.count(n)) {
        VLOG(3) << n->DebugString();
      }
    }
  }

  for (Node* n : post_order) {
    if (nodes_to_partially_decluster.count(n)) {
      TF_RETURN_IF_ERROR(PartiallyDeclusterNode(graph, n));
    }
  }

  nodes_to_partially_decluster.clear();
  TF_RETURN_IF_ERROR(FindNodesToDecluster(
      **options.graph, &nodes_to_partially_decluster, post_order));
  CHECK(nodes_to_partially_decluster.empty());

  return Status::OK();
}
}  // namespace tensorflow
