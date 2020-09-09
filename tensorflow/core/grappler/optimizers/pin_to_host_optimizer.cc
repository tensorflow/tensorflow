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

#include "tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace grappler {
namespace internal {

// TODO(williamchan): Change this constant to be something smarter, maybe
// dynamically determined.
constexpr int64 kTensorMaxSize = 64;

// All the nodes that should be denylisted and not swapped.
bool IsDenylisted(const NodeDef& node) {
  return
      // Collective ops should not be swapped.
      IsCollective(node) ||
      // ControlFlow ops should not be swapped.
      IsControlFlow(node) ||
      // NoOp ops should not be swapped (due to group dependencies).
      IsNoOp(node);
}

// Check if Tensor is either a string or is integer and small size
bool IsTensorSmall(const OpInfo::TensorProperties& prop) {
  if (prop.dtype() == DataType::DT_STRING) {
    return true;
  }

  // Check type to be int32 or int64.
  if (prop.dtype() != DataType::DT_INT32 &&
      prop.dtype() != DataType::DT_INT64 &&
      prop.dtype() != DataType::DT_FLOAT) {
    return false;
  }

  // Check size known and small.
  const int64 size = NumCoefficients(prop.shape());
  if (size < 0 || size > kTensorMaxSize) {
    return false;
  }

  return true;
}

// Find KernelDef for `node`, greedily return first found from `devices`.
Status TryFindKernelDef(const std::vector<DeviceType>& devices,
                        const NodeDef& node, const KernelDef** kdef) {
  for (const DeviceType& device : devices) {
    const KernelDef* kernel = nullptr;
    Status s = FindKernelDef(device, node, &kernel, nullptr);
    if (s.ok()) {
      if (kdef) {
        *kdef = kernel;
      }
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find KernelDef for op: ", node.op());
}

// Checks if a node's output port is host friendly.
// Roughly this means checking if the output port is on Host memory.
Status IsNodeOutputPortHostFriendly(const GraphView& graph,
                                    GraphProperties* properties,
                                    const NodeDef& node, int port_id,
                                    bool* is_candidate) {
  *is_candidate = false;

  // Make sure we are not a denylisted op.
  if (IsDenylisted(node)) {
    return Status::OK();
  }

  // Check to make sure we have the right properties (i.e., statically shaped).
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false, /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/false));
  }
  const auto& output_properties = properties->GetOutputProperties(node.name());
  int output_properties_size = output_properties.size();
  if (port_id >= output_properties_size) {
    LOG(WARNING) << "port_id=" << port_id
                 << " but output_properties.size()=" << output_properties.size()
                 << "\n"
                 << node.DebugString();
    return Status::OK();
  }
  if (!IsTensorSmall(output_properties[port_id])) {
    return Status::OK();
  }

  // These nodes may be optimized away downstream (even if pinned to Host), we
  // should (recursively) check their source.
  if (IsIdentity(node) || IsIdentityNSingleInput(node)) {
    for (const auto& fanin : graph.GetFanins(node, false)) {
      bool fanin_candidate = false;
      TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
          graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
      if (!fanin_candidate) {
        return Status::OK();
      }
    }
    *is_candidate = true;
    return Status::OK();
  }

  // Check if op's device is on CPU.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Check if op's output port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return Status::OK();
  }

  // Map the port_id to output_arg_id.
  const int output_arg_id = OpOutputPortIdToArgId(node, *op, port_id);
  if (output_arg_id < 0) {
    LOG(WARNING) << "Invalid port: " << port_id << "!\n"
                 << node.DebugString() << "\n"
                 << op->DebugString();
    return Status::OK();
  }

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = TryFindKernelDef({node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node,
                       &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return Status::OK();
  }

  // Check if the output_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->output_arg(output_arg_id).name() == host_memory_arg) {
      *is_candidate = true;
      break;
    }
  }

  return Status::OK();
}

// Checks if a node's input port is Host friendly.
// Roughly this means checking if the input port is on Host memory.
bool IsNodeInputPortHostFriendly(const NodeDef& node, int port_id) {
  // If node is on Host, assume its inputs are Host friendly.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    return true;
  }

  // Check if op's input port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return false;
  }
  const int input_arg_id = OpInputPortIdToArgId(node, *op, port_id);

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = internal::TryFindKernelDef(
      {node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node, &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return false;
  }

  // Check if the input_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->input_arg(input_arg_id).name() == host_memory_arg) {
      return true;
    }
  }

  return false;
}

// Checks if a node is a candidate to pin to Host.
// The rough algorithm is as follows:
// 1] Check if node is denylisted.
// 2] Check if node can run on Host.
// 3] Check all input/outputs are Host "friendly" (atm, friendly means small,
//    ints, and pinned to Host).
Status IsNodeHostCandidate(const GraphView& graph, GraphProperties* properties,
                           const NodeDef& node, bool* is_candidate) {
  *is_candidate = false;

  // Check if node already on CPU.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Skip these node types.
  if (IsDenylisted(node)) {
    return Status::OK();
  }

  // Check the node can be run on CPU.
  Status s = TryFindKernelDef({DEVICE_CPU}, node, nullptr);
  if (!s.ok()) {
    return Status::OK();
  }

  // Check all inputs are Host friendly.
  for (const GraphView::OutputPort& fanin :
       graph.GetFanins(node, /*include_controlling_nodes=*/false)) {
    bool fanin_candidate = false;
    TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
        graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
    if (!fanin_candidate) {
      return Status::OK();
    }
  }

  // Check all outputs are Host friendly.
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false, /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/false));
  }
  for (const auto& prop : properties->GetOutputProperties(node.name())) {
    if (!IsTensorSmall(prop)) {
      return Status::OK();
    }
  }

  *is_candidate = true;
  return Status::OK();
}

// Tries to find a Host device from `devices`. Returns empty string if no
// matching Host device is found.
string TryFindHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, const string& device) {
  // Force this node onto the CPU.
  if (device.empty() && has_device_cpu) {
    return "/device:CPU:0";
  } else if (absl::StrContains(device, DEVICE_GPU)) {
    // Sometimes the cluster can have:
    //   devices = {"/device:CPU:0", "/device:XLA_GPU:0"}
    // and we need to handle them properly.
    for (const auto& device_match :
         {std::pair<string, string>("GPU", "CPU:0"),
          std::pair<string, string>("/device", "/device:CPU:0")}) {
      const string device_host =
          strings::StrCat(device.substr(0, device.rfind(device_match.first)),
                          device_match.second);
      if (devices.find(device_host) != devices.end()) {
        return device_host;
      }
    }
  }

  // We couldn't find an appropriate Host device, return no device.
  return "";
}
}  // end namespace internal

Status PinToHostOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  // Skip all TPU graphs.
  if (IsTPUGraphDef(*optimized_graph)) {
    return Status::OK();
  }

  GraphProperties properties(item);
  GraphView graph(optimized_graph);

  gtl::FlatSet<string> devices;
  if (cluster) {
    const std::vector<string> device_names = cluster->GetDeviceNames();
    devices.insert(device_names.begin(), device_names.end());
  } else {
    devices = {"/device:CPU:0"};
  }

  const bool has_device_cpu = devices.find("/device:CPU:0") != devices.end();

  // Topologically sort the graph, so that we traverse the nodes in order. This
  // will help us discover producer->consumer chains of Host ops.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));

  // All the Const nodes, and their original devices in topological order.
  std::vector<std::pair<NodeDef*, string>> const_nodes;

  for (auto& node : *optimized_graph->mutable_node()) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    bool is_candidate = false;
    TF_RETURN_IF_ERROR(
        internal::IsNodeHostCandidate(graph, &properties, node, &is_candidate));
    if (!is_candidate) {
      continue;
    }

    string device =
        internal::TryFindHostDevice(devices, has_device_cpu, node.device());
    if (!device.empty()) {
      // Keep track of all Const nodes that we swapped.
      if (IsConstant(node)) {
        const_nodes.emplace_back(&node, node.device());
      }
      VLOG(2) << "Moving node " << node.name() << " to device " << device;
      *node.mutable_device() = std::move(device);
    }
  }

  // Traverse all `const_nodes`, and map them back to GPU greedily.
  for (auto& it : const_nodes) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = it.first;
    const string& device = it.second;

    // Check all the consumers of this node, if any of them are not on CPU, swap
    // this node back onto the original device.
    for (const GraphView::InputPort& fanout : graph.GetFanouts(*node, false)) {
      // The consumer is not Host friendly, swap it back to the original device.
      if (!internal::IsNodeInputPortHostFriendly(*fanout.node,
                                                 fanout.port_id)) {
        VLOG(2) << "Swapping node " << node->name() << " back to device "
                << device;
        node->set_device(device);
        break;
      }
    }
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
