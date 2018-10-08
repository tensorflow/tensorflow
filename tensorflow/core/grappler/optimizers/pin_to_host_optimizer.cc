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
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace grappler {

namespace internal {

namespace {
// TODO(williamchan): Change this constant to be something smarter, maybe
// dynamically determined.
constexpr int64 kTensorMaxSize = 64;

struct OpDevicePortHasher {
  std::size_t operator()(const std::tuple<string, string, int>& x) const {
    uint64 code = Hash64Combine(Hash64(std::get<0>(x)), Hash64(std::get<1>(x)));

    return Hash64Combine(code, hash<int>()(std::get<2>(x)));
  }
};
using OpDevicePortOnHostMap =
    gtl::FlatMap<std::tuple<string, string, int>, bool, OpDevicePortHasher>;

// All the nodes that should be blacklisted and not swapped.
bool IsBlacklisted(const NodeDef& node) {
  return
      // Collective ops should not be swapped.
      IsCollective(node) ||
      // ControlFlow ops should not be swapped.
      IsControlFlow(node) ||
      // NoOp ops should not be swapped (due to group dependencies).
      IsNoOp(node);
}

// Check if Tensor is integer and small size.
bool IsTensorIntegerAndSmall(const OpInfo::TensorProperties& prop) {
  // Check type to be int32 or int64.
  if (prop.dtype() != DataType::DT_INT32 &&
      prop.dtype() != DataType::DT_INT64) {
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
Status IsNodeOutputPortHostFriendly(
    const GraphView& graph, GraphProperties* properties, const NodeDef& node,
    int port_id, OpDevicePortOnHostMap* op_device_outport_pinned_to_host_cache,
    bool* is_candidate) {
  *is_candidate = false;

  // Make sure we are not a blacklisted op.
  if (IsBlacklisted(node)) {
    return Status::OK();
  }

  // Check to make sure we have the right properties (i.e., statically shaped).
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false));
  }
  const auto& output_properties = properties->GetOutputProperties(node.name());
  if (port_id >= output_properties.size()) {
    LOG(WARNING) << "port_id=" << port_id
                 << " but output_properties.size()=" << output_properties.size()
                 << "\n"
                 << node.DebugString();
    return Status::OK();
  }
  if (!IsTensorIntegerAndSmall(output_properties[port_id])) {
    return Status::OK();
  }

  // These nodes may be optimized away downstream (even if pinned to Host), we
  // should (recusively) check their source.
  if (IsIdentity(node)) {
    for (const auto& fanin : graph.GetFanins(node, false)) {
      bool fanin_candidate = false;
      TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
          graph, properties, *fanin.node, fanin.port_id,
          op_device_outport_pinned_to_host_cache, &fanin_candidate));
      if (!fanin_candidate) {
        return Status::OK();
      }
    }
    *is_candidate = true;
    return Status::OK();
  }

  // Check if op's device is on CPU.
  if (str_util::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Check `op_device_outport_pinned_to_host_cache` for our
  // {op, device, port_id} combo to see if the arg is pinned on Host.
  const std::tuple<string, string, int> cache_key(node.op(), node.device(),
                                                  port_id);
  auto it = op_device_outport_pinned_to_host_cache->find(cache_key);
  if (it != op_device_outport_pinned_to_host_cache->end()) {
    *is_candidate = it->second;
    return Status::OK();
  }

  // Check if op's output port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    op_device_outport_pinned_to_host_cache->emplace(cache_key, false);
    return Status::OK();
  }

  // Map the port_id to output_arg_id.
  const int output_arg_id = OpOutputPortIdToArgId(node, *op, port_id);
  if (output_arg_id < 0) {
    LOG(WARNING) << "Invalid port: " << port_id << "!\n"
                 << node.DebugString() << "\n"
                 << op->DebugString();
    op_device_outport_pinned_to_host_cache->emplace(cache_key, false);
    return Status::OK();
  }

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = TryFindKernelDef({node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node,
                       &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    op_device_outport_pinned_to_host_cache->emplace(cache_key, false);
    return Status::OK();
  }

  // Check if the output_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->output_arg(output_arg_id).name() == host_memory_arg) {
      *is_candidate = true;
      break;
    }
  }

  op_device_outport_pinned_to_host_cache->emplace(cache_key, *is_candidate);

  return Status::OK();
}

// Checks if a node's input port is Host friendly.
// Roughly this means checking if the input port is on Host memory.
bool IsNodeInputPortHostFriendly(
    const NodeDef& node, int port_id,
    OpDevicePortOnHostMap* op_device_inport_pinned_to_host_cache) {
  // If node is on Host, assume its inputs are Host friendly.
  if (str_util::StrContains(node.device(), DEVICE_CPU)) {
    return true;
  }

  // Check `op_device_inport_pinned_to_host_cache` for our
  // {op, device, port_id} combo to see if the arg is pinned on Host.
  std::tuple<string, string, int> cache_key(node.op(), node.device(), port_id);
  auto it = op_device_inport_pinned_to_host_cache->find(cache_key);
  if (it != op_device_inport_pinned_to_host_cache->end()) {
    return it->second;
  }

  // Check if op's input port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    op_device_inport_pinned_to_host_cache->emplace(cache_key, false);
    return false;
  }
  const int input_arg_id = OpInputPortIdToArgId(node, *op, port_id);

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = internal::TryFindKernelDef(
      {node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node, &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    op_device_inport_pinned_to_host_cache->emplace(cache_key, false);
    return false;
  }

  // Check if the input_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->input_arg(input_arg_id).name() == host_memory_arg) {
      op_device_inport_pinned_to_host_cache->emplace(cache_key, true);
      return true;
    }
  }

  op_device_inport_pinned_to_host_cache->emplace(cache_key, false);

  return false;
}

// Checks if a node is a candidate to pin to Host.
// The rough algorithm is as follows:
// 1] Check if node is blacklisted.
// 2] Check if node can run on Host.
// 3] Check all input/outputs are Host "friendly" (atm, friendly means small,
//    ints, and pinned to Host).
Status IsNodeHostCandidate(
    const GraphView& graph, GraphProperties* properties, const NodeDef& node,
    OpDevicePortOnHostMap* op_device_outport_pinned_to_host_cache,
    bool* is_candidate) {
  *is_candidate = false;

  // Skip these node types.
  if (IsBlacklisted(node)) {
    return Status::OK();
  }

  // Check if node already on CPU.
  if (str_util::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Check the node can be run on CPU.
  Status s = TryFindKernelDef({DEVICE_CPU}, node, nullptr);
  if (!s.ok()) {
    return Status::OK();
  }

  // Check all outputs are Host friendly.
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false));
  }
  for (const auto& prop : properties->GetOutputProperties(node.name())) {
    if (!IsTensorIntegerAndSmall(prop)) {
      return Status::OK();
    }
  }

  // Check all inputs are Host friendly.
  for (const GraphView::OutputPort& fanin :
       graph.GetFanins(node, /*include_controlling_nodes=*/false)) {
    bool fanin_candidate = false;
    TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
        graph, properties, *fanin.node, fanin.port_id,
        op_device_outport_pinned_to_host_cache, &fanin_candidate));
    if (!fanin_candidate) {
      return Status::OK();
    }
  }

  *is_candidate = true;
  return Status::OK();
}

bool IsTPUGraphDef(const GraphDef& def) {
  for (const auto& node : def.node()) {
    if (node.op() == "TPUCompile" || node.op() == "TPUExecute" ||
        node.op() == "TPUPartitionedCall") {
      return true;
    }
  }
  return false;
}
}  // end namespace

// Tries to swap `device` to a Host device from `devices`. Returns true iff
// there was a swap.
bool TrySwapToHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, string* device) {
  // Force this node onto the CPU.
  if (device->empty() && has_device_cpu) {
    *device = "/device:CPU:0";
    return true;
  } else if (str_util::StrContains(*device, DEVICE_GPU)) {
    // Sometimes the cluster can have:
    //   devices = {"/device:CPU:0", "/device:XLA_GPU:0"}
    // and we need to handle them properly.
    for (const auto& device_match :
         {std::pair<string, string>("GPU", "CPU:0"),
          std::pair<string, string>("/device", "/device:CPU:0")}) {
      const string device_host =
          strings::StrCat(device->substr(0, device->rfind(device_match.first)),
                          device_match.second);
      if (devices.find(device_host) != devices.end()) {
        *device = device_host;
        return true;
      }
    }
  }

  // We couldn't find an appropriate Host device, return false.
  return false;
}

}  // end namespace internal

Status PinToHostOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  // Skip all TPU graphs.
  if (internal::IsTPUGraphDef(*optimized_graph)) {
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

  // Cache to map {op, device, port} -> bool on whether it is pinned to host.
  internal::OpDevicePortOnHostMap op_device_outport_pinned_to_host_cache;
  internal::OpDevicePortOnHostMap op_device_inport_pinned_to_host_cache;

  for (auto& node : *optimized_graph->mutable_node()) {
    bool is_candidate = false;
    TF_RETURN_IF_ERROR(internal::IsNodeHostCandidate(
        graph, &properties, node, &op_device_outport_pinned_to_host_cache,
        &is_candidate));
    if (!is_candidate) {
      continue;
    }

    const string original_device = node.device();
    const bool swapped = internal::TrySwapToHostDevice(devices, has_device_cpu,
                                                       node.mutable_device());
    // Keep track of all Const nodes that we swapped.
    if (swapped && IsConstant(node)) {
      const_nodes.emplace_back(&node, original_device);
    }
  }

  // Traverse all `const_nodes`, and map them back to GPU greedily.
  for (auto& it : const_nodes) {
    NodeDef* node = it.first;
    const string& device = it.second;

    // Check all the consumers of this node, if any of them are not on CPU, swap
    // this node back onto the original device.
    for (const GraphView::InputPort& fanout : graph.GetFanouts(*node, false)) {
      // The consumer is not Host friendly, swap it back to the original device.
      if (!internal::IsNodeInputPortHostFriendly(
              *fanout.node, fanout.port_id,
              &op_device_inport_pinned_to_host_cache)) {
        node->set_device(device);
        break;
      }
    }
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
