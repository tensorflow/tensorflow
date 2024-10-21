/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimize_cross_host_control_deps.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

namespace {

absl::Status BuildNoopNode(const Node& source, StringPiece name,
                           const string& device, Graph* graph, Node** node) {
  NodeDefBuilder builder(name, "NoOp", NodeDebugInfo(source));
  if (!device.empty()) {
    builder.Device(device);
  }
  NodeDef def;
  TF_RETURN_IF_ERROR(builder.Finalize(&def));

  TF_ASSIGN_OR_RETURN(*node, graph->AddNode(def));
  if (!device.empty()) {
    (*node)->set_assigned_device_name(device);
  }
  return absl::OkStatus();
}

absl::Status BuildIdentityNNode(const Node& source, StringPiece name,
                                const string& device, Graph* graph,
                                std::vector<NodeDefBuilder::NodeOut>& inputs,
                                Node** node) {
  NodeDefBuilder builder(name, "IdentityN", NodeDebugInfo(source));
  if (!device.empty()) {
    builder.Device(device);
  }
  builder.Input(inputs);

  NodeDef def;
  TF_RETURN_IF_ERROR(builder.Finalize(&def));

  TF_ASSIGN_OR_RETURN(*node, graph->AddNode(def));
  if (!device.empty()) {
    (*node)->set_assigned_device_name(device);
  }
  return absl::OkStatus();
}

absl::Status BuildIdentityNode(const Node& source, StringPiece name,
                               const string& device, Graph* graph,
                               std::vector<NodeDefBuilder::NodeOut>& inputs,
                               Node** node) {
  NodeDefBuilder builder(name, "Identity", NodeDebugInfo(source));
  if (!device.empty()) {
    builder.Device(device);
  }
  builder.Input(inputs[0]);

  NodeDef def;
  TF_RETURN_IF_ERROR(builder.Finalize(&def));

  TF_ASSIGN_OR_RETURN(*node, graph->AddNode(def));
  if (!device.empty()) {
    (*node)->set_assigned_device_name(device);
  }
  return absl::OkStatus();
}

const string& RequestedOrAssignedDevice(const Node* n) {
  if (!n->assigned_device_name().empty()) {
    return n->assigned_device_name();
  }
  return n->requested_device();
}

// Class that assigns a number to each distinct device string, and allows to
// quickly look up whether two devices share the same address space.
class DeviceLookup {
 public:
  DeviceLookup() = default;

  static absl::StatusOr<DeviceLookup> FromGraph(Graph* graph) {
    DeviceLookup lookup;
    for (Node* n : graph->op_nodes()) {
      string device;
      TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
          RequestedOrAssignedDevice(n), &device));
      auto iter = lookup.device_name_to_id_.find(device);
      int id;
      if (iter == lookup.device_name_to_id_.end()) {
        id = lookup.device_name_to_id_.size();
        lookup.device_name_to_id_[device] = id;
        lookup.device_id_to_name_[id] = device;
      } else {
        id = iter->second;
      }
      lookup.node_to_device_id_[n] = id;
    }
    for (auto& [device1, id1] : lookup.device_name_to_id_) {
      for (auto& [device2, id2] : lookup.device_name_to_id_) {
        bool b = DeviceNameUtils::IsSameAddressSpace(device1, device2);
        lookup.is_same_address_space_[std::make_pair(id1, id2)] = b;
      }
    }
    return lookup;
  }

  inline int NodeToDeviceId(const Node* node) {
    return node_to_device_id_[node];
  }

  inline string DeviceIdToName(int id) { return device_id_to_name_[id]; }

  inline bool IsSameAddressSpace(int id1, int id2) {
    return is_same_address_space_[std::make_pair(id1, id2)];
  }

 private:
  absl::flat_hash_map<int, string> device_id_to_name_;
  absl::flat_hash_map<string, int> device_name_to_id_;
  absl::flat_hash_map<const Node*, int> node_to_device_id_;
  absl::flat_hash_map<std::pair<int, int>, bool> is_same_address_space_;
};

}  // namespace

absl::Status OptimizeCrossHostControlOutputEdges(
    Graph* graph, int cross_host_edges_threshold) {
  TF_ASSIGN_OR_RETURN(DeviceLookup lookup, DeviceLookup::FromGraph(graph));

  for (Node* n : graph->op_nodes()) {
    if (n->out_edges().size() < cross_host_edges_threshold) {
      continue;
    }
    absl::flat_hash_map<int, std::vector<const Edge*>> cross_host_control_edges;
    int src_device_id = lookup.NodeToDeviceId(n);
    for (const Edge* edge : n->out_edges()) {
      if (!edge->IsControlEdge() || edge->dst()->IsSink()) {
        continue;
      }

      int dst_device_id = lookup.NodeToDeviceId(edge->dst());

      if (lookup.IsSameAddressSpace(src_device_id, dst_device_id)) {
        continue;
      }
      auto iter = cross_host_control_edges.find(dst_device_id);
      if (iter == cross_host_control_edges.end()) {
        cross_host_control_edges[dst_device_id] = {edge};
      } else {
        iter->second.push_back(edge);
      }
    }
    for (const auto& pair : cross_host_control_edges) {
      if (pair.second.size() < cross_host_edges_threshold) {
        continue;
      }
      string device = lookup.DeviceIdToName(pair.first);
      VLOG(1) << "Optmize cross host output control edge, src node: "
              << n->name()
              << " src device: " << lookup.DeviceIdToName(src_device_id)
              << " dst host device: " << device
              << " edges size: " << pair.second.size();
      Node* control_after;
      TF_RETURN_IF_ERROR(BuildNoopNode(
          *n, graph->NewName(strings::StrCat(n->name(), "/", "control_after")),
          device, graph, &control_after));

      // When adding control edges, set `allow_duplicates` to true since the
      // duplication check is expensive and unnecessary here due to there
      // shouldn't be duplicated control edges introduced by this pass.
      graph->AddControlEdge(n, control_after, /*allow_duplicates=*/true);
      for (const Edge* edge : pair.second) {
        graph->AddControlEdge(control_after, edge->dst(),
                              /*allow_duplicates=*/true);
        graph->RemoveEdge(edge);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status OptimizeCrossHostDataOutputEdges(Graph* graph,
                                              int cross_host_edges_threshold) {
  TF_ASSIGN_OR_RETURN(DeviceLookup lookup, DeviceLookup::FromGraph(graph));

  for (Node* n : graph->op_nodes()) {
    if (n->out_edges().size() < cross_host_edges_threshold) {
      continue;
    }
    absl::flat_hash_map<int, std::vector<const Edge*>> cross_host_edges;
    int src_id = lookup.NodeToDeviceId(n);
    for (const Edge* edge : n->out_edges()) {
      Node* dst = edge->dst();
      if (edge->IsControlEdge() || dst->IsSink()) {
        continue;
      }

      int dst_id = lookup.NodeToDeviceId(dst);

      if (lookup.IsSameAddressSpace(src_id, dst_id)) {
        continue;
      }
      auto iter = cross_host_edges.find(dst_id);
      if (iter == cross_host_edges.end()) {
        cross_host_edges[dst_id] = {edge};
      } else {
        iter->second.push_back(edge);
      }
    }
    for (const auto& pair : cross_host_edges) {
      if (pair.second.size() < cross_host_edges_threshold) {
        continue;
      }
      if (pair.second.empty()) {
        continue;
      }
      int device_id = pair.first;
      // If all our outputs are already going to a single node, we don't
      // need to insert another node. That also makes this transformation
      // idempotent.
      Node* node0 = pair.second[0]->dst();
      if (std::all_of(pair.second.begin(), pair.second.end(),
                      [node0](const Edge* e) { return e->dst() == node0; })) {
        continue;
      }
      string device = lookup.DeviceIdToName(device_id);
      VLOG(1) << "Optimize cross host output edge, src node: " << n->name()
              << " src device: " << lookup.DeviceIdToName(src_id)
              << " dst host device: " << device
              << " edges size: " << pair.second.size();

      Node* data_after;
      std::vector<NodeDefBuilder::NodeOut> inputs;
      inputs.reserve(pair.second.size());
      const Edge* edge0 = pair.second[0];
      if (std::all_of(pair.second.begin(), pair.second.end(),
                      [edge0](const Edge* e) {
                        return e->src() == edge0->src() &&
                               e->src_output() == edge0->src_output();
                      })) {
        // Handle the special case of all inputs being identical, which is when
        // we only need an Identity op with one input.
        // TODO(kramm): Can we break this up further? E.g. what if we have two
        // sets of inputs that are both all identical?
        inputs.emplace_back(edge0->src()->name(), edge0->src_output(),
                            edge0->src()->output_type(edge0->src_output()));
        TF_RETURN_IF_ERROR(BuildIdentityNode(
            *n, graph->NewName(strings::StrCat(n->name(), "/", "data_after")),
            device, graph, inputs, &data_after));

        graph->AddEdge(edge0->src(), edge0->src_output(), data_after, 0);
        int i = 0;
        for (const Edge* edge : pair.second) {
          graph->AddEdge(data_after, 0, edge->dst(), edge->dst_input());
          graph->RemoveEdge(edge);
          i++;
        }
      } else {
        for (const Edge* edge : pair.second) {
          inputs.emplace_back(edge->src()->name(), edge->src_output(),
                              edge->src()->output_type(edge->src_output()));
        }
        TF_RETURN_IF_ERROR(BuildIdentityNNode(
            *n, graph->NewName(strings::StrCat(n->name(), "/", "data_after")),
            device, graph, inputs, &data_after));

        int i = 0;
        for (const Edge* edge : pair.second) {
          graph->AddEdge(data_after, i, edge->dst(), edge->dst_input());
          graph->AddEdge(edge->src(), edge->src_output(), data_after, i);
          graph->RemoveEdge(edge);
          i++;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status OptimizeCrossHostControlInputEdges(
    Graph* graph, int cross_host_edges_threshold) {
  TF_ASSIGN_OR_RETURN(DeviceLookup lookup, DeviceLookup::FromGraph(graph));

  absl::flat_hash_map<Node*, std::vector<const Edge*>> node_control_input_edges;
  for (Node* n : graph->op_nodes()) {
    for (const Edge* edge : n->out_edges()) {
      if (!edge->IsControlEdge() || edge->dst()->IsSink()) {
        continue;
      }
      Node* dst = edge->dst();
      auto iter = node_control_input_edges.find(dst);
      if (iter == node_control_input_edges.end()) {
        node_control_input_edges[dst] = {edge};
      } else {
        node_control_input_edges[dst].push_back(edge);
      }
    }
  }

  for (auto& pair : node_control_input_edges) {
    Node* dst = pair.first;
    const std::vector<const Edge*>& input_edges = pair.second;

    if (input_edges.size() < cross_host_edges_threshold) {
      continue;
    }

    absl::flat_hash_map<int, std::vector<const Edge*>> cross_host_control_edges;
    int dst_device_id = lookup.NodeToDeviceId(dst);

    for (const Edge* edge : input_edges) {
      int src_device_id = lookup.NodeToDeviceId(edge->src());
      if (lookup.IsSameAddressSpace(src_device_id, dst_device_id)) {
        continue;
      }
      auto iter = cross_host_control_edges.find(src_device_id);
      if (iter == cross_host_control_edges.end()) {
        cross_host_control_edges[src_device_id] = {edge};
      } else {
        iter->second.push_back(edge);
      }
    }
    for (const auto& pair : cross_host_control_edges) {
      if (pair.second.size() < cross_host_edges_threshold) {
        continue;
      }
      string src_device = lookup.DeviceIdToName(pair.first);
      VLOG(1) << "Optmize cross host input control edge, dst node: "
              << dst->name()
              << " dst device: " << lookup.DeviceIdToName(dst_device_id)
              << " src host device: " << src_device
              << " edges size: " << pair.second.size();
      Node* control_before;
      TF_RETURN_IF_ERROR(BuildNoopNode(
          *dst,
          graph->NewName(strings::StrCat(dst->name(), "/", "control_before")),
          /*device=*/src_device, graph, &control_before));

      // When adding control edges, set `allow_duplicates` to true since the
      // duplication check is expensive and unnecessary here due to there
      // shouldn't be duplicated control edges introduced by this pass.
      graph->AddControlEdge(control_before, dst, /*allow_duplicates=*/true);
      for (const Edge* edge : pair.second) {
        graph->AddControlEdge(edge->src(), control_before,
                              /*allow_duplicates=*/true);
        graph->RemoveEdge(edge);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
