/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

#include <math.h>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

Costs CombineCosts(const Costs& left, const Costs& right) {
  CHECK_NE(left.max_memory, kMemoryUnknown);
  CHECK_NE(left.max_per_op_buffers, kMemoryUnknown);
  CHECK_NE(left.max_per_op_streaming, kMemoryUnknown);

  Costs result = left;
  result.execution_time += right.execution_time;
  if (right.max_memory != kMemoryUnknown) {
    result.max_memory += right.max_memory;
  }
  if (right.max_per_op_buffers != kMemoryUnknown) {
    result.max_per_op_buffers =
        std::max(left.max_per_op_buffers, right.max_per_op_buffers);
  }
  if (right.max_per_op_streaming != kMemoryUnknown) {
    result.max_per_op_streaming =
        std::max(left.max_per_op_streaming, right.max_per_op_streaming);
  }
  VLOG(3) << "costs execution_time=" << result.execution_time.count()
          << " max_memory=" << result.max_memory
          << " max_per_op_buffers=" << result.max_per_op_buffers
          << " max_per_op_streaming=" << result.max_per_op_streaming;
  return result;
}
}  // namespace

VirtualScheduler::VirtualScheduler(const GrapplerItem* grappler_item,
                                   const bool use_static_shapes,
                                   Cluster* cluster)
    :  // TODO(dyoon): Use a better way than FIFO.
      ready_nodes_(new FIFOManager()),
      graph_costs_(Costs::ZeroCosts()),
      graph_properties_(*grappler_item),
      cluster_(cluster),
      grappler_item_(grappler_item),
      use_static_shapes_(use_static_shapes),
      placer_(cluster) {
  initialized_ = false;
}

Status VirtualScheduler::Init() {
  // Init() preprocesses the input grappler_item and graph_properties to extract
  // necessary information for emulating tensorflow op scheduling and
  // construct internal data structures (NodeState and DeviceState) for virtual
  // scheduling.

  // Construct graph properties.
  Status status;
  if (use_static_shapes_) {
    status = graph_properties_.InferStatically();
  } else {
    status = graph_properties_.InferDynamically(cluster_);
  }
  if (!status.ok()) {
    return status;
  }

  const auto& graph = grappler_item_->graph;
  const auto& fetch_nodes = grappler_item_->fetch;

  // Get the nodes that would run to output fetch_nodes.
  std::vector<const NodeDef*> nodes =
      ComputeTransitiveFanin(graph, fetch_nodes);

  // TODO(dyoon): this is a bit inefficient as name_to_node is already built in
  // ComputeTransitiveFanin().
  // Once ComputeTransitiveFanin is complete, only the nodes that can be reached
  // from the fetch nodes are scheduled. So the scheduled nodes should be
  // exactly the same as those executed for real. One possible discrepancy could
  // be the control flow nodes, where tf only executes one path.
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : nodes) {
    name_to_node[node->name()] = node;
  }

  // Build node_map; for each node, create its NodeState and connect its inputs
  // and outputs.
  for (const auto* curr_node : nodes) {
    auto& curr_node_state = GetNodeStateOrCreateIt(curr_node);
    const string curr_node_device = DeviceName(curr_node);
    for (const string& input_node_name : curr_node->input()) {
      // Note that input_node_name may be in <prefix><node_name>:<port_num>
      // format, where <prefix> (e.g., "^" for control dependency) and
      // ":<port_num>" may be omitted. NodeName() extracts only the node_name.
      const NodeDef* input_node = name_to_node[NodeName(input_node_name)];

      CHECK(input_node);
      const string in_device = DeviceName(input_node);
      const auto input_node_port_num = NodePosition(input_node_name);

      if (curr_node_device == in_device) {
        // Same device: connect input_node and curr_node directly.
        curr_node_state.inputs.push_back(
            std::make_pair(input_node, input_node_port_num));
        auto& input_node_state = GetNodeStateOrCreateIt(input_node);
        input_node_state.outputs[input_node_port_num].push_back(curr_node);
      } else {
        if (cached_recv_nodes_.count(input_node) > 0 &&
            cached_recv_nodes_[input_node].count(curr_node_device) > 0) {
          // Different device, but found an already-cached copy (a _Recv op);
          // connect the _Recv to curr_node.
          const auto* recv_op =
              cached_recv_nodes_[input_node][curr_node_device];
          // recv_op's output port is hard-coded to zero.
          curr_node_state.inputs.push_back(std::make_pair(recv_op, 0));
          auto& input_node_state = node_map_.at(recv_op);
          input_node_state.outputs[0].push_back(curr_node);
        } else {
          // Different device, no cached copy; transfer input_node to the
          // curr_node's device.
          auto send_and_recv =
              CreateSendRecv(input_node, curr_node, input_node_name);
          // Note that CreateSendRecv() already connected input/output between
          // _Send and _Recv ops.
          const auto* send = send_and_recv.first;
          const auto* recv = send_and_recv.second;
          // recv_op's output port is hard-coded to zero.
          curr_node_state.inputs.push_back(std::make_pair(recv, 0));
          auto& input_node_state = GetNodeStateOrCreateIt(input_node);
          input_node_state.outputs[input_node_port_num].push_back(send);

          // Cache the _Recv op for future use.
          cached_recv_nodes_[input_node][curr_node_device] = recv;
        }
      }
    }

    if (curr_node->input().empty()) {
      // Node without input: ready at time 0.
      curr_node_state.time_ready = Costs::Duration();
      ready_nodes_->AddNode(curr_node);
    }

    if (IsPersistentNode(curr_node)) {
      auto& device_state = device_[curr_node_device];
      for (int port_num = 0;
           port_num < curr_node_state.output_properties.size(); ++port_num) {
        device_state.persistent_nodes.insert(
            std::make_pair(curr_node, port_num));
      }
    }
  }

  if (ready_nodes_->Empty()) {
    return Status(error::UNAVAILABLE, "No ready nodes in the graph.");
  }

  initialized_ = true;
  return Status::OK();
}

void VirtualScheduler::MaybeUpdateInputOutput(const NodeDef* node) {
  CHECK(!initialized_) << "MaybeUpdateInputOutput is called after Init().";
  // This method is called when NodeState is created and adds input and output
  // properties for a few exceptional cases that GraphProperties cannot provide
  // input/output properties.
  if (IsSend(*node) || IsRecv(*node)) {
    auto& node_state = node_map_[node];
    auto& inputs = node_state.input_properties;
    auto& outputs = node_state.output_properties;

    // _Send and _Recv ops are created from VirtualScheduler, so
    // there should be no inputs TensorProperties.
    CHECK(inputs.empty());
    CHECK(outputs.empty());
    const auto& attr = node->attr();
    // This is the original input source to the _Send and _Recv, and this
    // string includes "^" if it was control dependency, and output port
    /// (e.g., ":2") if the input source had multiple outputs.
    const auto& input_source_name = attr.at(kAttrInputSrc).s();
    if (IsControlInput(input_source_name)) {
      // Control dependency; regardless of the input source tensor size,
      // send 4B.
      OpInfo::TensorProperties control_message;
      control_message.set_dtype(DT_FLOAT);
      control_message.mutable_shape()->add_dim()->set_size(1);
      auto* value = control_message.mutable_value();
      value->add_float_val(1);
      inputs.push_back(control_message);
      outputs.push_back(control_message);
    } else {
      auto output_properties =
          graph_properties_.GetOutputProperties(NodeName(input_source_name));
      // Like with HasInputProperties, if a node does not have output
      // properties, it's likely it was pruned during the shape inference run.
      if (!output_properties.empty()) {
        const auto input_node_port_num = NodePosition(input_source_name);
        // Use the input source's output property as _Send and _Recv's input
        // property.
        CHECK_GT(output_properties.size(), input_node_port_num);
        inputs.push_back(output_properties[input_node_port_num]);
        outputs.push_back(output_properties[input_node_port_num]);
      }
    }
  }
}

float VirtualScheduler::Round2(const float x) const {
  // Not using std::round from <cmath> here because not all platforms seem to
  // support that (specifically Android).
  return ::round(100.0 * x) / 100.0;
}

bool VirtualScheduler::IsPersistentNode(const NodeDef* node) const {
  // Variables are persistent nodes.
  return IsVariable(*node);
}

string VirtualScheduler::DeviceName(const NodeDef* node) const {
  return placer_.get_canonical_device_name(*node);
}

string VirtualScheduler::ChannelDeviceName(const NodeDef* from,
                                           const NodeDef* to) const {
  CHECK(!initialized_) << "ChannelDeviceName is called after Init().";

  return kChannelDevice + ": " + DeviceName(from) + " to " + DeviceName(to);
}

std::pair<const NodeDef*, const NodeDef*> VirtualScheduler::CreateSendRecv(
    const NodeDef* from, const NodeDef* to, const string& input_name) {
  CHECK(!initialized_) << "CreateSendRecv is called after Init().";

  // Connect "from" node to "to" node with _Send and _Recv such that
  // from -> _Send -> _Recv -> to.
  // _Send is placed on "Channel" device, and _Recv is on the same device
  // as "to" node.
  // input_node_name is the string from the "to" node to identify which output
  // we get from the "from" node.

  // Note that we use NodeState for scheduling, so _Send and _Recv
  // NodeDefs created here need not be correct: in terms of name,
  // input names, attrs, etc.

  auto input_node_port_num = NodePosition(input_name);

  // _Send op.
  auto* send = new NodeDef();
  send->set_name("Send " + from->name() + " from " + DeviceName(from) + " to " +
                 DeviceName(to));
  send->set_op("_Send");
  send->add_input(from->name());
  send->set_device(ChannelDeviceName(from, to));
  auto& send_attr = *(send->mutable_attr());
  send_attr[kAttrInputSrc].set_s(input_name);
  send_attr[kAttrSrcDevice].set_s(DeviceName(from));
  send_attr[kAttrDstDevice].set_s(DeviceName(to));

  // _Recv op.
  auto* recv = new NodeDef();
  recv->set_name("Recv " + from->name() + " on " + DeviceName(to));
  recv->set_op("_Recv");
  recv->add_input(send->name());
  recv->set_device(DeviceName(to));
  auto& recv_attr = *(recv->mutable_attr());
  recv_attr[kAttrInputSrc].set_s(input_name);

  // NodeState for _Send op.
  auto& send_node_state = GetNodeStateOrCreateIt(send);
  send_node_state.device_name = send->device();  // Set Channel device.
  send_node_state.inputs.push_back(std::make_pair(from, input_node_port_num));
  send_node_state.outputs[0].push_back(recv);

  // NodeState for _Recv op.
  auto& recv_node_state = GetNodeStateOrCreateIt(recv);
  recv_node_state.inputs.push_back(std::make_pair(send, 0));
  recv_node_state.outputs[0].push_back(to);

  // Keep the created nodes.
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(send));
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(recv));

  // Return _Send and _Recv.
  return std::make_pair(send, recv);
}

NodeInfo VirtualScheduler::GetCurrNodeInfo() const {
  const NodeDef* node = ready_nodes_->GetCurrNode();

  // Get the device from the placer.
  DeviceProperties device;
  device = placer_.get_device(*node);

  // Special case for _Send op.
  if (IsSend(*node)) {
    device.set_type(kChannelDevice);
  }

  // Construct NodeInfo.
  const auto& node_state = node_map_.at(node);
  NodeInfo node_info;
  node_info.name = node->name();
  node_info.device_name = node_state.device_name;
  auto& op_info = node_info.op_info;
  op_info.set_op(node->op());
  *op_info.mutable_attr() = node->attr();
  for (auto& input : node_state.input_properties) {
    *op_info.add_inputs() = input;
  }
  for (auto& output : node_state.output_properties) {
    *op_info.add_outputs() = output;
  }
  op_info.mutable_device()->Swap(&device);
  return node_info;
}

NodeState& VirtualScheduler::GetNodeStateOrCreateIt(const NodeDef* node) {
  CHECK(!initialized_) << "GetNodeStateOrCreateIt is called after Init().";

  auto it = node_map_.find(node);
  if (it == node_map_.end()) {
    // Not found; create a NodeState for this node.
    it = node_map_.emplace(node, NodeState()).first;
    auto& node_state = it->second;
    node_state.input_properties =
        graph_properties_.GetInputProperties(node->name());
    node_state.output_properties =
        graph_properties_.GetOutputProperties(node->name());

    // Some ops may need further processing to the input / output properties:
    // _Send and _Recv.
    MaybeUpdateInputOutput(node);

    if (!IsSend(*node)) {
      node_state.device_name = DeviceName(node);
      // For _Send op, device_name will be set to Channel in CreateSendRecv().
    }

    // Initialize output port related data:
    // Assume the size of OutputProperties represents the number of output ports
    // of this node.
    for (int i = 0; i < node_state.output_properties.size(); ++i) {
      node_state.time_no_references[i] = Costs::Duration::max();
      node_state.num_outputs_executed[i] = 0;
      // Populate an empty vector for each port. The caller will add nodes
      // that use this port as input.
      node_state.outputs[i] = {};
    }
    // Port_num -1 is for control dependency.
    node_state.time_no_references[-1] = Costs::Duration::max();
    node_state.num_outputs_executed[-1] = 0;
    node_state.outputs[-1] = {};
  }
  return it->second;
}

int64 VirtualScheduler::CalculateOutputSize(
    const std::vector<OpInfo::TensorProperties>& output_properties,
    const int port_num) const {
  if (port_num < 0) {
    return 4;  // 4B for control dependency.
  }

  if (port_num >= output_properties.size()) {
    VLOG(3) << "VirtualScheduler::CalculateOutputSize() -- "
            << "port_num: " << port_num
            << " >= output_properties.size(): " << output_properties.size();
    return 0;
  }

  const auto& output = output_properties[port_num];
  int64 output_size = DataTypeSize(BaseType(output.dtype()));

  for (const auto& dim : output.shape().dim()) {
    auto dim_size = dim.size();
    if (dim_size < 0) {
      // Zero output size if there's any unknown dim.
      output_size = 0;
      VLOG(3) << "VirtualScheduler::CalculateOutputSize() -- "
              << "unknown dim: " << output_size;
      break;
    }
    output_size *= dim_size;
  }

  return output_size;
}

Costs& VirtualScheduler::FindOrCreateZero(const string& op_name,
                                          std::map<string, Costs>* op_cost) {
  auto it = op_cost->find(op_name);
  if (it == op_cost->end()) {
    // Note that default constructor of Costs sets some memory related fields
    // to unknown values so we should explicitly initialize it with ZeroCosts.
    it = op_cost->emplace(op_name, Costs::ZeroCosts()).first;
  }
  return it->second;
}

bool VirtualScheduler::MarkCurrNodeExecuted(const Costs& node_costs) {
  // Update graph_costs_ and per-op costs.
  graph_costs_ = CombineCosts(graph_costs_, node_costs);
  const auto* node = ready_nodes_->GetCurrNode();
  const auto& op_name = node->op();

  // Also keep track of op counts and times per op (with their shapes).
  NodeInfo node_info = GetCurrNodeInfo();
  string node_description = GetOpDescription(node_info.op_info);
  op_counts_[node_description] += 1;
  op_costs_[node_description] =
      node_costs.execution_time.asMicroSeconds().count();

  auto& op_cost = FindOrCreateZero(op_name, &op_to_cost_);
  op_cost = CombineCosts(op_cost, node_costs);

  // Update node and device states.
  auto& node_state = node_map_[node];
  auto& device = device_[node_state.device_name];
  device.nodes_executed.push_back(node);
  // Node is scheduled when the device is available AND all the inputs are
  // ready; hence, time_scheduled is time_ready if time_ready > device curr
  // time.
  node_state.time_scheduled =
      std::max(device.GetCurrTime(), node_state.time_ready);
  // Override device curr time with the time_scheduled.
  device.device_costs.execution_time = node_state.time_scheduled;
  device.device_costs = CombineCosts(device.device_costs, node_costs);
  auto curr_time = device.GetCurrTime();
  node_state.time_finished = curr_time;

  // Update device memory usage.
  if (!IsPersistentNode(node)) {
    for (const auto& port_num_output_pair : node_state.outputs) {
      int port_num = port_num_output_pair.first;
      // There's a chance that a specific output is not used at all.
      if (node_state.outputs[port_num].empty()) {
        node_state.time_no_references[port_num] = curr_time;
      } else {
        device.memory_usage +=
            CalculateOutputSize(node_state.output_properties, port_num);
        device.nodes_in_memory.insert(std::make_pair(node, port_num));
      }
    }
  }

  // Update device's per-op cost.
  auto& device_op_cost = FindOrCreateZero(op_name, &device.op_to_cost);
  device_op_cost = CombineCosts(device_op_cost, node_costs);

  VLOG(2) << "Op scheduled -- name: " << node->name() << ", op: " << node->op()
          << ", device: " << node->device()
          << ", ready: " << node_state.time_ready.count()
          << ", scheduled: " << node_state.time_scheduled.count()
          << ", finished: " << node_state.time_finished.count();

  // Increment num_inputs_ready of the output nodes
  for (const auto& port_num_output_pair : node_state.outputs) {
    for (auto* output_node : port_num_output_pair.second) {
      auto& output_state = node_map_[output_node];
      output_state.num_inputs_ready++;
      if (output_state.num_inputs_ready == output_state.inputs.size()) {
        // This output node is now ready.
        output_state.time_ready = curr_time;
        ready_nodes_->AddNode(output_node);
      }
    }
  }

  // Increment num_outputs_executed of the input nodes.
  for (const auto& input_port : node_state.inputs) {
    auto* input = input_port.first;
    auto port = input_port.second;
    auto& input_state = node_map_[input];
    input_state.num_outputs_executed[port]++;
    if (input_state.num_outputs_executed[port] ==
            input_state.outputs[port].size() &&
        !IsPersistentNode(input)) {
      // All the outputs are executed; no reference to this output port of
      // input node.
      input_state.time_no_references[port] = curr_time;
      auto& input_device = device_[input_state.device_name];
      input_device.memory_usage -=
          CalculateOutputSize(input_state.output_properties, port);

      input_device.nodes_in_memory.erase(std::make_pair(input, port));
    }
  }

  if (!IsPersistentNode(node)) {
    // Now that output memory is added and used up nodes are deallocated,
    // check max memory usage.
    if (device.memory_usage > device.max_memory_usage) {
      device.max_memory_usage = device.memory_usage;
      device.mem_usage_snapshot_at_peak = device.nodes_in_memory;
    }
  }

  // Remove the current node; assume FIFO.
  ready_nodes_->RemoveCurrNode();

  return !ready_nodes_->Empty();
}

Costs VirtualScheduler::Summary() const {
  // Print out basic execution summary.
  VLOG(1) << "Expected execution time: " << graph_costs_.execution_time.count();
  VLOG(1) << "Expected max memory: " << graph_costs_.max_memory;
  VLOG(1) << "Expected max per-op buffers: " << graph_costs_.max_per_op_buffers;
  VLOG(1) << "Expected max per-op streaming buffers: "
          << graph_costs_.max_per_op_streaming;

  VLOG(1) << "Per-op execution time:";
  for (const auto& op_cost_pair : op_to_cost_) {
    const auto& op = op_cost_pair.first;
    const auto& cost = op_cost_pair.second.execution_time.count();
    if (cost) {  // Skip printing out zero-cost ops.
      VLOG(1) << " + " << op << " : " << cost;
    }
  }

  // Print per device summary
  VLOG(1) << "Devices:";
  Costs critical_path_costs = Costs::ZeroCosts();

  for (const auto& device : device_) {
    const auto& name = device.first;
    const auto& state = device.second;

    std::map<string, int64> op_to_memory;
    // First profile only persistent memory usage.
    int64 persistent_memory_usage = 0;
    std::set<string> persisent_ops;
    for (const auto& node_port : state.persistent_nodes) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      const auto output_size =
          CalculateOutputSize(node_map_.at(node).output_properties, port);
      persistent_memory_usage += output_size;
      op_to_memory[node->op()] += output_size;
      persisent_ops.insert(node->op());
    }
    int64 max_memory_usage = persistent_memory_usage + state.max_memory_usage;
    critical_path_costs.estimated_max_memory_per_device[name] =
        max_memory_usage;

    VLOG(1) << "Device = " << name
            << ", num_nodes = " << state.nodes_executed.size()
            << ", execution_time = " << state.GetCurrTime().count()
            << ", memory usage: "
            << "persistenst = "
            << Round2(persistent_memory_usage / 1024.0 / 1024.0 / 1024.0)
            << " GB, peak = "
            << Round2(state.max_memory_usage / 1024.0 / 1024.0 / 1024.0)
            << " GB, total = "
            << Round2(max_memory_usage / 1024.0 / 1024.0 / 1024.0)
            << " GB, at the end: " << state.memory_usage << " B";

    VLOG(1) << "Per-op execution time (and memory usage at peak memory usage):";
    // Profile non-persistent op memory usage.
    for (const auto& node_port : state.mem_usage_snapshot_at_peak) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      op_to_memory[node->op()] +=
          CalculateOutputSize(node_map_.at(node).output_properties, port);
    }
    for (const auto& op_cost_pair : state.op_to_cost) {
      const auto& op = op_cost_pair.first;
      const auto& cost = op_cost_pair.second.execution_time.count();
      const float mem_usage_gb =
          Round2(op_to_memory[op] / 1024.0 / 1024.0 / 1024.0);
      int64 op_mem_usage = op_to_memory.at(op);
      const float mem_usage_percent =
          max_memory_usage > 0 ? Round2(100.0 * op_mem_usage / max_memory_usage)
                               : 0.0;
      if (cost || mem_usage_percent > 1.0) {
        // Print out only non-zero cost ops or ops with > 1% memory usage.
        VLOG(1) << " + " << op << " : " << cost << " (" << mem_usage_gb
                << " GB [" << mem_usage_percent << "%] "
                << (persisent_ops.count(op) > 0 ? ": persistent op)" : ")");
      }
    }
    if (critical_path_costs.execution_time <= state.GetCurrTime()) {
      critical_path_costs = state.device_costs;
    }
  }

  // Also log the op description and their corresponding counts.
  VLOG(2) << "Node description, counts, cost:";
  for (const auto& item : op_counts_) {
    VLOG(2) << "Node: " << item.first << ", Count: " << item.second
            << ", Individual Cost: " << op_costs_.at(item.first);
  }

  VLOG(1) << "Critical path execution time: "
          << critical_path_costs.execution_time.count();
  return critical_path_costs;
}

}  // end namespace grappler
}  // end namespace tensorflow
