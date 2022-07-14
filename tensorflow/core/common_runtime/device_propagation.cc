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

#include "tensorflow/core/common_runtime/device_propagation.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace {

const std::string& AssignedOrRequestedDevice(const Node& node) {
  if (!node.assigned_device_name().empty()) {
    return node.assigned_device_name();
  }
  return node.requested_device();
}

void UpdateDeviceFromInputs(
    const device_propagation::NodeFilter& node_filter,
    const device_propagation::DeviceFilter& device_filter, Node* node) {
  if (!AssignedOrRequestedDevice(*node).empty() || !node_filter(*node)) {
    return;
  }
  string proposed_device = "";
  Node* proposed_src = nullptr;
  // Scan the input edges, propagate device assignment from its inputs to this
  // node iff all input nodes has the same device assignment and the device is
  // propagatable (checked by `device_filter`). Some kinds of edges are
  // ignored.
  for (const Edge* e : node->in_edges()) {
    // Ignore control edge.
    if (e->IsControlEdge()) {
      continue;
    }
    Node* src = e->src();
    const string& src_device = AssignedOrRequestedDevice(*src);

    // Ignore LoopCond -> Switch and Enter -> Merge. In other words, the device
    // placement of a Switch op is determined by all its non-LoopCond inputs and
    // that of a Merge op is determined by all its non-Enter inputs.
    if ((node->IsSwitch() && src->IsLoopCond()) ||
        (node->IsMerge() && src->IsEnter())) {
      continue;
    }

    // If a source device is not propagatable, stop.
    if (!device_filter(src_device)) return;

    if (proposed_src == nullptr) {
      proposed_device = src_device;
      proposed_src = src;
    } else if (proposed_device != src_device) {
      // The device assignments of some input nodes are not the same. Stop.
      return;
    }
  }
  if (proposed_src) {
    node->set_assigned_device_name(proposed_src->assigned_device_name());
    node->set_requested_device(proposed_src->requested_device());
  }
}

}  // namespace

void PropagateDevices(const device_propagation::NodeFilter& node_filter,
                      const device_propagation::DeviceFilter& device_filter,
                      Graph* graph) {
  ReverseDFS(*graph, {}, [&node_filter, &device_filter](Node* node) {
    UpdateDeviceFromInputs(node_filter, device_filter, node);
  });
}

}  // namespace tensorflow
