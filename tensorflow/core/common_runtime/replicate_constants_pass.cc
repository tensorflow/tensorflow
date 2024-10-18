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

#include "tensorflow/core/common_runtime/replicate_constants_pass.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

// Maximum size constant to replicate.
constexpr int64_t kMaxSize = 16;

// Set `node`'s name to <original-name>/replicate/_<unique-index>
void SetUniqueName(Graph* graph, Node* node) {
  node->set_name(graph->NewName(absl::StrCat(node->name(), "/replicate")));
}

// `node` has an output control edge.
bool HasControlOut(Node* node) {
  auto control_out_it =
      std::find_if(node->out_edges().begin(), node->out_edges().end(),
                   [](const auto& e) { return e->IsControlEdge(); });
  return control_out_it != node->out_edges().end();
}

// `node`'s device is a CPU.
bool HasCpuDevice(const Node* node) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(), &device))
    return false;
  return device.type == "CPU";
}

// Convert the CPU device name to the corresponding CPU device name. If
// multiple local CPU devices are enabled, the CPU device name will also
// contain the device id.
absl::Status DeviceNameToCpuDeviceNameWithDeviceId(const string& device_name,
                                                   string* host_device_name) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_name, &device)) {
    return absl::InternalError(
        absl::StrCat("Could not parse device name ", device_name));
  }
  // If aggressive constant replication is enabled and the dst node is on CPU.
  // We just use the device name of the dst for the src.
  if (flags::Global().enable_aggressive_constant_replication.value() &&
      device.type == "CPU") {
    *host_device_name = device_name;
  } else {
    // If not, assigning the corresponding CPU 0 to it.
    device.type = "CPU";
    device.has_type = true;
    device.id = 0;
    device.has_id = true;
    *host_device_name = DeviceNameUtils::ParsedNameToString(device);
  }
  return absl::OkStatus();
}

// Get the CPU device on the same host as dst.
absl::Status GetDestinationCpuDevice(const Node* dst, std::string* device) {
  if (!dst->has_assigned_device_name())
    return absl::AbortedError(
        absl::StrCat("Node name: ", dst->name(), " has no assigned device."));
  return DeviceNameToCpuDeviceNameWithDeviceId(dst->assigned_device_name(),
                                               device);
}

// Collect the successor edges of the constant. Group them by the device of the
// successor.
absl::Status GetSuccessorEdges(
    Node* node,
    absl::btree_map<std::string, std::vector<const Edge*>>& device_to_edges) {
  for (const auto& edge : node->out_edges()) {
    const Node* dst = edge->dst();
    std::string device;
    TF_RETURN_IF_ERROR(GetDestinationCpuDevice(dst, &device));
    if (!device_to_edges.count(device)) device_to_edges.insert({device, {}});
    device_to_edges[device].push_back(edge);
  }
  return absl::OkStatus();
}

// Replicate the constant to each successor device.
void ReplicateToEachDevice(
    Graph* graph, Node* node,
    absl::btree_map<std::string, std::vector<const Edge*>>& device_to_edges) {
  for (const auto& pair : device_to_edges) {
    Node* copy = graph->CopyNode(node);
    SetUniqueName(graph, copy);
    const std::string device = pair.first;
    copy->set_assigned_device_name(device);
    // Set the successor edges to ops on this device.
    for (const Edge* edge : pair.second) {
      graph->AddEdge(copy, edge->src_output(), edge->dst(), edge->dst_input());
    }
    // Replicate in edges that are control.
    for (Node* src : node->in_nodes()) {
      graph->AddControlEdge(src, copy, true);
    }
  }
  graph->RemoveNode(node);
}

}  // namespace

absl::Status ReplicateConstantsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "replicate_constants_pass will replicate constants with "
             "number-of-elements <= "
          << kMaxSize;

  if (options.graph == nullptr) {
    VLOG(1) << "No graph in replicate_constants_pass.";
    return absl::OkStatus();
  }
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("before_replicate_constants_pass", *graph,
                               options.flib_def);
  }
  int64_t min_skipped = std::numeric_limits<int64_t>::max();
  int64_t max_skipped = std::numeric_limits<int64_t>::min();
  for (Node* node : graph->nodes()) {
    if (!node->IsConstant()) continue;

    // For performance, skip when there is at most one successor.
    if (node->out_edges().size() <= 1) continue;

    // Skip if the constant has a control successor. Replicating constants with
    // control successors would require relpicating these control edges, which
    // could result in even more message passing.
    if (HasControlOut(node)) continue;

    // Skip if the constant is too large.
    const TensorProto* value = nullptr;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "value", &value));
    TF_ASSIGN_OR_RETURN(TensorShape shape,
                        TensorShape::BuildTensorShape(value->tensor_shape()));
    if (shape.num_elements() > kMaxSize) {
      min_skipped = std::min(min_skipped, shape.num_elements());
      max_skipped = std::max(max_skipped, shape.num_elements());
      continue;
    }

    // Skip if there is no assigned device.
    if (!node->has_assigned_device_name()) continue;

    // Skip when the original constant is not on a CPU, because is not clear
    // whether replicating from non-CPU to CPU is valid.
    if (!HasCpuDevice(node)) continue;

    // Collect successor edges, per device.
    absl::btree_map<std::string, std::vector<const Edge*>> device_to_edges;
    TF_RETURN_IF_ERROR(GetSuccessorEdges(node, device_to_edges));

    // Skip if all successors are on the same device.
    if (device_to_edges.size() <= 1) continue;

    // Replicate the constant to each successor device.
    ReplicateToEachDevice(graph, node, device_to_edges);
  }
  if (min_skipped != std::numeric_limits<int64_t>::max()) {
    VLOG(1) << "replicate_constants_pass skipped replicating constants with "
               "number of elements in the range "
            << min_skipped << " to " << max_skipped << ".";
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("after_replicate_constants_pass", *graph,
                               options.flib_def);
  }
  return absl::OkStatus();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 3,
                      ReplicateConstantsPass);

}  // namespace tensorflow
