/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"

#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace {

// A helper for rewriting nodes assigned to a virtual composite device.
class ReplicateHelper {
 public:
  // Replicate the given node to all allowed devices.
  Status ReplicateNode(const Node* node,
                       const std::vector<string>& allowed_devices,
                       Graph* graph) {
    if (replicated_nodes_map_.find(node) != replicated_nodes_map_.end()) {
      return errors::InvalidArgument("Node ", node->name(),
                                     " has been replicated.");
    }

    std::vector<Node*> replicated_nodes(allowed_devices.size());
    for (int i = 0; i < allowed_devices.size(); ++i) {
      const auto& device = allowed_devices.at(i);
      NodeDef node_def = node->def();
      const string suffix = strings::StrCat("/R", i);
      node_def.set_name(
          graph->NewName(strings::StrCat(node_def.name(), suffix)));
      Status status;
      Node* replicated_node = graph->AddNode(node_def, &status);
      TF_RETURN_IF_ERROR(status);
      replicated_node->set_assigned_device_name(device);
      replicated_nodes[i] = replicated_node;
    }
    replicated_nodes_map_.emplace(node, std::move(replicated_nodes));
    return Status::OK();
  }

  // Replace an edge (a regular device -> composite device) with
  // N edges (a regular device -> allowed devices).
  void ReplicateFromRegularDeviceToCompositeDevice(const Edge* edge,
                                                   Graph* graph) const {
    Node* src = edge->src();
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    for (Node* dst : dst_replicated_nodes) {
      graph->AddEdge(src, edge->src_output(), dst, edge->dst_input());
    }
  }

  // Replace an edge (composite device -> composite device) with
  // N edges (allowed devices -> allowed devices).
  Status ReplicateFromCompositeDeviceToCompositeDevice(const Edge* edge,
                                                       Graph* graph) const {
    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    if (src_replicated_nodes.size() != dst_replicated_nodes.size()) {
      return errors::InvalidArgument(
          "Nodes assigned to the same composite device should have the "
          "same number of replicated nodes. Found an edge from node ",
          edge->src()->name(), " (", src_replicated_nodes.size(),
          " replicated nodes) to node ", edge->dst()->name(), " (",
          dst_replicated_nodes.size(), " replicated nodes).");
    }
    for (int i = 0; i < src_replicated_nodes.size(); ++i) {
      graph->AddEdge(src_replicated_nodes.at(i), edge->src_output(),
                     dst_replicated_nodes.at(i), edge->dst_input());
    }
    return Status::OK();
  }

  // Data edge: replace an edge (composite device -> a regular device) with
  // one edge (one allowed device -> a regular device).
  // Control edge: replace an edge (composite device -> a regular device) with
  // N edges (allowed devices -> a regular device).
  Status ReplicateFromCompositeDeviceToRegularDevice(const Edge* edge,
                                                     Graph* graph) const {
    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    Node* dst = edge->dst();
    if (edge->IsControlEdge()) {
      for (Node* replicated_node : src_replicated_nodes) {
        graph->AddControlEdge(replicated_node, dst);
      }
    } else {
      const string& dst_device = dst->assigned_device_name();
      bool found_src_node = false;
      for (Node* replicated_node : src_replicated_nodes) {
        if (replicated_node->assigned_device_name() == dst_device) {
          graph->AddEdge(replicated_node, edge->src_output(), dst,
                         edge->dst_input());
          found_src_node = true;
          break;
        }
      }
      if (!found_src_node) {
        if (edge->src()->type_string() == "_Arg") {
          // This happens when the dst node runs on a host CPU and
          // captures a function with an arg node assigned to the same
          // composite device (e.g. ScanDataset).
          // For this case, we only need to add an edge connecting the arg
          // node in the outer function and the corresponding arg in the
          // inner function, since the host CPU only needs one copy of the
          // ResourceHandle.
          graph->AddEdge(src_replicated_nodes.at(0), edge->src_output(), dst,
                         edge->dst_input());
        } else {
          return errors::InvalidArgument(
              "Dst node should be assigned to an allowed device. Found an "
              "edge from node ",
              edge->src()->name(), " assigned to ",
              edge->src()->assigned_device_name(), " to node ", dst->name(),
              " assigned to ", dst_device);
        }
      }
    }
    return Status::OK();
  }

 private:
  // Map from original nodes to corresponding replicated nodes.
  absl::flat_hash_map<const Node*, std::vector<Node*>> replicated_nodes_map_;
};

// Replicate the nodes in cluster_nodes to all allowed devices.
Status ReplicateNodes(const std::vector<Node*>& cluster_nodes,
                      const std::vector<string>& allowed_devices,
                      ReplicateHelper* helper, Graph* graph) {
  for (Node* n : cluster_nodes) {
    TF_RETURN_IF_ERROR(helper->ReplicateNode(n, allowed_devices, graph));
  }
  return Status::OK();
}

// Replicate the edges connecting original nodes for replicated nodes.
Status ReplicateEdges(const ReplicateHelper& helper,
                      const std::vector<Node*>& cluster_nodes, Graph* graph) {
  for (const auto* node : cluster_nodes) {
    // Replicate input edges.
    for (const Edge* edge : node->in_edges()) {
      Node* src = edge->src();
      if (src->assigned_device_name() != node->assigned_device_name()) {
        // The source node is assigned to a different device.
        helper.ReplicateFromRegularDeviceToCompositeDevice(edge, graph);
      } else {
        // The source node is assigned to the same composite device.
        TF_RETURN_IF_ERROR(
            helper.ReplicateFromCompositeDeviceToCompositeDevice(edge, graph));
      }
    }

    // Replicate output edges.
    for (const Edge* edge : node->out_edges()) {
      Node* dst = edge->dst();
      if (dst->assigned_device_name() != node->assigned_device_name()) {
        // The dst node is assigned to a different device.
        TF_RETURN_IF_ERROR(
            helper.ReplicateFromCompositeDeviceToRegularDevice(edge, graph));
      }
      // The else branch has been covered when iterating over input edges.
    }
  }
  return Status::OK();
}

}  // namespace

Status ReplicatePerReplicaNodesInFunctionGraph(
    const absl::flat_hash_map<string, std::vector<string>>& composite_devices,
    Graph* graph) {
  std::set<string> composite_device_names;
  for (const auto& it : composite_devices) {
    composite_device_names.insert(it.first);
  }
  // Map from a composite device to a cluster of nodes assigned to the
  // composite device.
  absl::flat_hash_map<string, std::vector<Node*>>
      composite_device_to_cluster_nodes;
  for (Node* n : graph->op_nodes()) {
    if (composite_device_names.find(n->assigned_device_name()) !=
        composite_device_names.end()) {
      composite_device_to_cluster_nodes[n->assigned_device_name()].push_back(n);
    }
  }

  for (const auto& it : composite_device_to_cluster_nodes) {
    const std::vector<string>& allowed_devices = composite_devices.at(it.first);
    if (allowed_devices.empty()) {
      return errors::InvalidArgument("No allowed device of composite device: ",
                                     it.first);
    }
    const std::vector<Node*>& cluster_nodes = it.second;
    if (allowed_devices.size() == 1) {
      // Reuse the original nodes if there is only one allowed device.
      for (Node* n : cluster_nodes) {
        n->set_assigned_device_name(allowed_devices.at(0));
      }
      continue;
    }
    ReplicateHelper helper;
    TF_RETURN_IF_ERROR(
        ReplicateNodes(cluster_nodes, allowed_devices, &helper, graph));
    TF_RETURN_IF_ERROR(ReplicateEdges(helper, cluster_nodes, graph));

    // Remove orignial nodes.
    for (auto* n : cluster_nodes) {
      graph->RemoveNode(n);
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
