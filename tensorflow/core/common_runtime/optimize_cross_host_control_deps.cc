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

#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

namespace {

Status BuildNoopNode(const Node& source, StringPiece name, const string& device,
                     Graph* graph, Node** node) {
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
  return OkStatus();
}

const string& RequestedOrAssignedDevice(const Node* n) {
  if (!n->assigned_device_name().empty()) {
    return n->assigned_device_name();
  }
  return n->requested_device();
}

}  // namespace

Status OptimizeCrossHostControlOutputEdges(Graph* graph,
                                           int cross_host_edges_threshold) {
  string src_host_device;
  string dst_host_device;
  for (Node* n : graph->op_nodes()) {
    if (n->out_edges().size() < cross_host_edges_threshold) {
      continue;
    }
    absl::flat_hash_map<string, std::vector<const Edge*>>
        cross_host_control_edges;
    TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
        RequestedOrAssignedDevice(n), &src_host_device));
    for (const Edge* edge : n->out_edges()) {
      if (!edge->IsControlEdge() || edge->dst()->IsSink()) {
        continue;
      }

      TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
          RequestedOrAssignedDevice(edge->dst()), &dst_host_device));
      if (DeviceNameUtils::IsSameAddressSpace(src_host_device,
                                              dst_host_device)) {
        continue;
      }
      auto iter = cross_host_control_edges.find(dst_host_device);
      if (iter == cross_host_control_edges.end()) {
        cross_host_control_edges[dst_host_device] = {edge};
      } else {
        iter->second.push_back(edge);
      }
    }
    for (const auto& pair : cross_host_control_edges) {
      if (pair.second.size() < cross_host_edges_threshold) {
        continue;
      }
      VLOG(1) << "Optmize cross host output control edge, src node: "
              << n->name() << " src device: " << src_host_device
              << " dst host device: " << pair.first
              << " edges size: " << pair.second.size();
      Node* control_after;
      TF_RETURN_IF_ERROR(BuildNoopNode(
          *n, graph->NewName(strings::StrCat(n->name(), "/", "control_after")),
          /*device=*/pair.first, graph, &control_after));
      graph->AddControlEdge(n, control_after);
      for (const Edge* edge : pair.second) {
        graph->AddControlEdge(control_after, edge->dst());
        graph->RemoveEdge(edge);
      }
    }
  }
  return OkStatus();
}

Status OptimizeCrossHostControlInputEdges(Graph* graph,
                                          int cross_host_edges_threshold) {
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

  string src_host_device;
  string dst_host_device;
  for (auto& pair : node_control_input_edges) {
    Node* dst = pair.first;
    const std::vector<const Edge*>& input_edges = pair.second;

    if (input_edges.size() < cross_host_edges_threshold) {
      continue;
    }

    absl::flat_hash_map<string, std::vector<const Edge*>>
        cross_host_control_edges;
    TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
        RequestedOrAssignedDevice(dst), &dst_host_device));
    for (const Edge* edge : input_edges) {
      TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
          RequestedOrAssignedDevice(edge->src()), &src_host_device));
      if (DeviceNameUtils::IsSameAddressSpace(src_host_device,
                                              dst_host_device)) {
        continue;
      }
      auto iter = cross_host_control_edges.find(src_host_device);
      if (iter == cross_host_control_edges.end()) {
        cross_host_control_edges[src_host_device] = {edge};
      } else {
        iter->second.push_back(edge);
      }
    }
    for (const auto& pair : cross_host_control_edges) {
      if (pair.second.size() < cross_host_edges_threshold) {
        continue;
      }
      VLOG(0) << "Optmize cross host input control edge, dst node: "
              << dst->name() << " dst device: " << dst_host_device
              << " src host device: " << pair.first
              << " edges size: " << pair.second.size();
      Node* control_before;
      TF_RETURN_IF_ERROR(BuildNoopNode(
          *dst,
          graph->NewName(strings::StrCat(dst->name(), "/", "control_before")),
          /*device=*/pair.first, graph, &control_before));
      graph->AddControlEdge(control_before, dst);
      for (const Edge* edge : pair.second) {
        graph->AddControlEdge(edge->src(), control_before);
        graph->RemoveEdge(edge);
      }
    }
  }
  return OkStatus();
}

}  // namespace tensorflow
