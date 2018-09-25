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
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace grappler {
namespace internal {

// TODO(williamchan): Change this constant to be something smarter, maybe
// dynamically determined.
constexpr int64 kTensorMaxSize = 64;

// Find KernelDef for `node`.
Status TryFindKernelDef(const NodeDef& node, const KernelDef** kdef) {
  // Try find KernelDef for node.device, else GPU or CPU.
  for (const DeviceType& device :
       {node.device().c_str(), DEVICE_GPU, DEVICE_CPU}) {
    Status s = FindKernelDef(device, node, kdef, nullptr);
    if (s.ok()) {
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find KernelDef for op: ", node.op());
}

// Check if all node's inputs are pinned to CPU memory.
bool AreAllNodeInputsPinnedToHost(const GraphView& graph, const NodeDef& node) {
  // Loop through all the inputs excluding the controlling nodes.
  for (const GraphView::OutputPort& fanin : graph.GetFanins(node, false)) {
    // Check if (the fanin) op's device is on CPU.
    if (str_util::StrContains(fanin.node->device(), DEVICE_CPU)) {
      continue;
    }

    // Check if (the fanin) op's output port is pinned to HostMemory.
    const OpDef* fanin_odef = nullptr;
    Status s = OpRegistry::Global()->LookUpOpDef(fanin.node->op(), &fanin_odef);
    if (!s.ok()) {
      LOG(INFO) << "Could not find OpDef for : " << fanin.node->op();
      return false;
    }

    const int output_arg_id =
        OpOutputPortIdToArgId(*fanin.node, *fanin_odef, fanin.port_id);
    if (output_arg_id < 0) {
      LOG(WARNING) << "Invalid port: " << fanin.port_id << "!\n"
                   << node.DebugString() << "\n"
                   << fanin.node->DebugString() << "\n"
                   << fanin_odef->DebugString();
      return false;
    }

    const KernelDef* fanin_kdef = nullptr;
    s = TryFindKernelDef(*fanin.node, &fanin_kdef);
    if (!s.ok()) {
      LOG(INFO) << "Could not find KernelDef for : " << fanin.node->op();
      return false;
    }

    bool fanin_pinned = false;
    for (const string& host_memory_arg : fanin_kdef->host_memory_arg()) {
      if (fanin_odef->output_arg(output_arg_id).name() == host_memory_arg) {
        fanin_pinned = true;
        break;
      }
    }

    if (!fanin_pinned) {
      return false;
    }
  }

  return true;
}

bool IsTensorIntegerAndSmall(const OpInfo::TensorProperties& prop) {
  // Check if Tensor is integer and small size.

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

bool AreAllNodeInputsAndOutputsIntsAndSmall(const GraphProperties& properties,
                                            const NodeDef& node) {
  for (const auto& prop : properties.GetInputProperties(node.name())) {
    if (!IsTensorIntegerAndSmall(prop)) {
      return false;
    }
  }

  for (const auto& prop : properties.GetOutputProperties(node.name())) {
    if (!IsTensorIntegerAndSmall(prop)) {
      return false;
    }
  }
  return true;
}

string TryFindHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, const string& device) {
  // Force this node onto the CPU.
  if (device.empty() && has_device_cpu) {
    return "/device:CPU:0";
  } else if (str_util::StrContains(device, DEVICE_GPU)) {
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

  // We couldn't find an appropriate Host device, return original device.
  return device;
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

// All the nodes that should be blacklisted and not swapped.
bool IsBlacklisted(const NodeDef& node) { return IsCollective(node); }
}  // end namespace internal

Status PinToHostOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  // Skip all TPU graphs.
  if (internal::IsTPUGraphDef(*optimized_graph)) {
    return Status::OK();
  }

  GraphProperties properties(item);
  bool has_properties = false;
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
    // Check if node already on CPU.
    if (str_util::StrContains(node.device(), DEVICE_CPU)) {
      continue;
    }

    // Skip these node types.
    if (internal::IsBlacklisted(node)) {
      continue;
    }

    // Check the node can be run on CPU.
    Status s = FindKernelDef(DEVICE_CPU, node, nullptr, nullptr);
    if (!s.ok()) {
      continue;
    }

    // Check all input's are pinned to CPU.
    if (!internal::AreAllNodeInputsPinnedToHost(graph, node)) {
      continue;
    }

    if (!has_properties) {
      // This is an expensive call, call it lazily.
      TF_RETURN_IF_ERROR(properties.InferStatically(false));
      has_properties = true;
    }

    // Check all inputs and outputs are integers and small.
    if (!internal::AreAllNodeInputsAndOutputsIntsAndSmall(properties, node)) {
      continue;
    }

    if (IsConstant(node)) {
      const_nodes.emplace_back(&node, node.device());
    }
    // Try and swap the device to Host.
    node.set_device(
        internal::TryFindHostDevice(devices, has_device_cpu, node.device()));
  }

  // Traverse all `const_nodes`, and map them back to GPU greedily.
  for (auto& it : const_nodes) {
    NodeDef* node = it.first;
    const string& device = it.second;

    // Check all the consumers of this node, if any of them are on the original
    // device, swap this node back onto the original device.
    for (const GraphView::InputPort& fanout : graph.GetFanouts(*node, false)) {
      if (fanout.node->device() == device) {
        node->set_device(device);
        break;
      }
    }
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
