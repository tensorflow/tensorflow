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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/utils.h"
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
                                   const string& default_device_type,
                                   Cluster* cluster, VirtualPlacer* placer)
    : graph_properties_(*grappler_item),
      graph_costs_(Costs::ZeroCosts()),
      // TODO(dyoon): Use a better way than FIFO.
      ready_nodes_(new FIFOManager()),
      cluster_(cluster),
      grappler_item_(grappler_item),
      use_static_shapes_(use_static_shapes),
      default_device_type_(default_device_type),
      placer_(placer) {
  initialized_ = false;
}

Status VirtualScheduler::Init() {
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

  // First, get the nodes that would run to output fetch_nodes.
  std::vector<const NodeDef*> nodes =
      ComputeTransitiveFanin(graph, fetch_nodes);

  // TODO(dyoon): this is a bit inefficient as name_to_node is already built in
  // ComputeTransitiveFanin().
  //
  // Once ComputeTransitiveFanin is complete, only the nodes that can be reached
  // from the fetch nodes are scheduled. So the scheduled nodes should be
  // exactly the same as those executed for real. One possible discrepancy could
  // be the control flow nodes, where tf only executes one path.
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : nodes) {
    name_to_node[node->name()] = node;
  }

  // Build node_map.
  for (const auto* curr_node : nodes) {
    auto& curr_node_state = GetNodeStateOrCreateIt(curr_node);
    const string curr_node_device = DeviceName(curr_node);
    for (const string& input_node_name : curr_node->input()) {
      // Note that input_node_name may be in <node_name>:<output_number> format,
      // where ":<output_number>" may be omitted. NodeName() extracts only the
      // node_name (prefeix "^", if there was for control input, is also
      // deleted).
      const NodeDef* input_node = name_to_node[NodeName(input_node_name)];
      CHECK(input_node);
      // Add input_to_curr_node to curr_node's input, and
      // add output_to_input_node to input_source_node's output.
      // Default values for when input_node and curr_node on the same device.
      const NodeDef* input_to_curr_node = input_node;
      const NodeDef* input_source_node = input_node;
      const NodeDef* output_to_input_node = curr_node;
      const string in_device = DeviceName(input_node);
      if (curr_node_device != in_device) {
        if (cached_ops_.count(input_node) > 0 &&
            cached_ops_[input_node].count(curr_node_device) > 0) {
          // Different device, but found an already-transferred copy; connect
          // the cached node to curr_node.
          input_to_curr_node = cached_ops_[input_node][curr_node_device];
          input_source_node = input_to_curr_node;
          output_to_input_node = curr_node;
        } else {
          // Different device, no cached copy; transfer input_node to the
          // curr_node's device.
          auto sendrecv_and_identity =
              TransferNode(input_node, curr_node, input_node_name);
          const auto* sendrecv = sendrecv_and_identity.first;
          const auto* identity = sendrecv_and_identity.second;
          input_to_curr_node = identity;
          input_source_node = input_node;
          output_to_input_node = sendrecv;

          // Cache the identity op for future use.
          cached_ops_[input_node][curr_node_device] = identity;
        }
      }
      curr_node_state.inputs.push_back(input_to_curr_node);

      // Note that we do not care output number (in case a tf op has multiple
      // outputs), as VirtualScheduler only cares which nodes become ready as
      // a node is executed.
      auto& input_node_state = GetNodeStateOrCreateIt(input_source_node);
      input_node_state.outputs.push_back(output_to_input_node);
    }

    if (curr_node->input().empty()) {
      curr_node_state.time_ready =
          Costs::Duration();  // Node without input: ready at time 0.
      ready_nodes_->AddNode(curr_node);
    }
  }

  if (ready_nodes_->Empty()) {
    return Status(error::UNAVAILABLE, "No ready nodes in the graph.");
  }

  initialized_ = true;
  return Status::OK();
}

void VirtualScheduler::MaybeUpdateInputProperties(
    const NodeDef* node, std::vector<OpInfo::TensorProperties>* inputs) const {
  if (IsSendOp(node) || IsRecvOp(node)) {
    // _Send and _Recv ops are inserted from VirtualScheduler, so
    // there should be no inputs TensorProperties.
    CHECK_EQ(inputs->size(), 0);
    const auto& attr = node->attr();
    // This is the original input source to the _Send and _Recv, and this
    // string includes "^" if it was control dependency, and output port
    /// (e.g., ":2") if the input source had multiple outputs.
    const auto& input_source_name = attr.at(kAttrInputSrc).s();
    if (input_source_name[0] == '^') {
      // Control dependency; regardless of the input source tensor size,
      // send 4B.
      OpInfo::TensorProperties control_message;
      control_message.set_dtype(DT_FLOAT);
      control_message.mutable_shape()->add_dim()->set_size(1);
      auto* value = control_message.mutable_value();
      value->add_float_val(1);
      inputs->push_back(control_message);
    } else {
      // Like with HasInputProperties, if a node does not have output
      // properties, it's likely it was pruned during the shape inference run.
      if (graph_properties_.HasOutputProperties(NodeName(input_source_name))) {
        const auto input_position = NodePosition(input_source_name);
        // Use the input source's output property as _Send and _Recv's input
        // property.
        auto outputs =
            graph_properties_.GetOutputProperties(NodeName(input_source_name));
        CHECK_GT(outputs.size(), input_position);
        inputs->push_back(outputs[input_position]);
      }
    }
  }
}

bool VirtualScheduler::IsSendOp(const NodeDef* node) const {
  return node->op() == kSend;
}

bool VirtualScheduler::IsRecvOp(const NodeDef* node) const {
  return node->op() == kRecv;
}

string VirtualScheduler::DeviceName(const NodeDef* node) const {
  // TODO(dyoon): integrate this part with VirtualPlacer.
  if (IsSendOp(node)) {
    const auto& node_state = node_map_.at(node);
    const auto* from = node_state.inputs[0];
    const auto* to = node_state.outputs[0];
    return ChannelDeviceName(from, to);
  } else {
    return node->device().empty() ? "/" + default_device_type_ + ":0"
                                  : node->device();
  }
}

string VirtualScheduler::ChannelDeviceName(const NodeDef* from,
                                           const NodeDef* to) const {
  return kChannelDevice + ": " + DeviceName(from) + " to " + DeviceName(to);
}

std::pair<const NodeDef*, const NodeDef*> VirtualScheduler::TransferNode(
    const NodeDef* from, const NodeDef* to, const string& input_name) {
  // Connect "from" node to "to" node with _Send and _Recv such that
  // from -> _Send -> _Recv -> to.
  // _Send is placed on "Channel" device, and _Recv is on the same device
  // as "to" node.
  // input_node_name is the string from the "to" node to identify which output
  // we get from the "from" node.

  // Note that we use NodeState for scheduling, so _Send and _Recv
  // NodeDefs created here need not be correct: in terms of name,
  // input names, attrs, etc.

  // _Send op.
  auto* send = new NodeDef();
  send->set_name("Send " + from->name() + " from " + DeviceName(from) + " to " +
                 DeviceName(to));
  send->set_op(kSend);
  send->add_input(from->name());
  send->set_device(ChannelDeviceName(from, to));
  auto& send_attr = *(send->mutable_attr());
  send_attr[kAttrInputSrc].set_s(input_name);
  send_attr[kAttrSrcDevice].set_s(DeviceName(from));
  send_attr[kAttrDstDevice].set_s(DeviceName(to));

  // _Recv op.
  auto* recv = new NodeDef();
  recv->set_name("Recv " + from->name() + " on " + DeviceName(to));
  recv->set_op(kRecv);
  recv->add_input(send->name());
  recv->set_device(DeviceName(to));
  auto& recv_attr = *(recv->mutable_attr());
  recv_attr[kAttrInputSrc].set_s(input_name);

  // Update NodeState for _Send and _Recv ops.
  auto& send_node_state = GetNodeStateOrCreateIt(send);
  send_node_state.inputs.push_back(from);
  send_node_state.outputs.push_back(recv);
  auto& recv_node_state = GetNodeStateOrCreateIt(recv);
  recv_node_state.inputs.push_back(send);
  recv_node_state.outputs.push_back(to);

  // Keep the created nodes.
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(send));
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(recv));

  // Return _Send and _Recv.
  return std::make_pair(send, recv);
}

NodeInfo VirtualScheduler::GetCurrNodeInfo() const {
  const NodeDef* node = ready_nodes_->GetCurrNode();
  std::vector<OpInfo::TensorProperties> inputs =
      graph_properties_.GetInputProperties(node->name());
  // Some ops created within VirtualScheduler may need further processing to
  // the input properties.
  MaybeUpdateInputProperties(node, &inputs);

  // This is for compatibility; we can just use palcer_->get_device() for all
  // cases, once VirtualCluster is properly set up.
  DeviceProperties device;
  if (placer_) {
    device = placer_->get_device(*node);
  }
  if (device.type() == "UNKNOWN") {
    string device_type;
    int device_id;
    DeviceNameUtils::ParsedName parsed;
    if (!node->device().empty() &&
        DeviceNameUtils::ParseFullName(DeviceName(node), &parsed)) {
      device_type = parsed.type;
      device_id = parsed.id;
    } else {
      device_type = default_device_type_;
      device_id = 0;
    }
    if (device_type == "GPU") {
      device = GetLocalGPUInfo(device_id);
    } else if (device_type == "CPU") {
      device = GetLocalCPUInfo();
    }
  }

  // Special case for _Send op.
  if (IsSendOp(node)) {
    device.set_type(kChannelDevice);
  }

  NodeInfo node_info;
  node_info.name = node->name();
  node_info.device_name = graph_properties_.GetDeviceName(node->name());
  node_info.outputs = graph_properties_.GetOutputProperties(node->name());
  auto& op_info = node_info.op_info;
  op_info.set_op(node->op());
  *op_info.mutable_attr() = node->attr();
  for (auto& input : inputs) {
    op_info.add_inputs()->Swap(&input);
  }
  op_info.mutable_device()->Swap(&device);
  // add some more to the node_info.
  return node_info;
}

NodeState& VirtualScheduler::GetNodeStateOrCreateIt(const NodeDef* node) {
  auto it = node_map_.find(node);
  if (it == node_map_.end()) {
    it = node_map_.emplace(node, NodeState()).first;
  }
  return it->second;
}

Costs& VirtualScheduler::FindOrCreateZero(const string& op_name,
                                          std::map<string, Costs>* op_cost) {
  auto it = op_cost->find(op_name);
  if (it == op_cost->end()) {
    it = op_cost->emplace(op_name, Costs::ZeroCosts()).first;
  }
  return it->second;
}

bool VirtualScheduler::PopCurrNode() {
  const auto* node = ready_nodes_->GetCurrNode();
  auto& node_state = node_map_[node];
  auto& device = device_[DeviceName(node)];
  auto curr_time = device.GetCurrTime();

  // Increment num_inputs_ready of the output nodes.
  for (auto* output : node_state.outputs) {
    auto& output_state = node_map_[output];
    output_state.num_inputs_ready++;
    if (output_state.num_inputs_ready == output_state.inputs.size()) {
      // This output node is now ready.
      output_state.time_ready = curr_time;
      ready_nodes_->AddNode(output);
    }
  }

  // Increment num_outputs_executed of the input nodes.
  for (auto* input : node_state.inputs) {
    auto& input_state = node_map_[input];
    input_state.num_outputs_executed++;
    if (input_state.num_outputs_executed == input_state.outputs.size()) {
      // All the outputs are executed; no reference to this input nodel
      input_state.time_no_reference = curr_time;
      // TODO(dyoon): collect device memory usage; note that this input node
      // use device memory between time_scheduled and time_no_reference.
    }
  }

  // Remove the current node; assume FIFO.
  ready_nodes_->RemoveCurrNode();

  return !ready_nodes_->Empty();
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
  auto& device = device_[DeviceName(node)];
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

  // Update device's per-op cost.
  auto& device_op_cost = FindOrCreateZero(op_name, &device.op_to_cost);
  device_op_cost = CombineCosts(device_op_cost, node_costs);

  VLOG(2) << "Op scheduled -- name: " << node->name() << ", op: " << node->op()
          << ", device: " << node->device()
          << ", ready: " << node_state.time_ready.count()
          << ", scheduled: " << node_state.time_scheduled.count()
          << ", finished: " << node_state.time_finished.count();

  return PopCurrNode();
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
    VLOG(1) << "Device = " << name
            << ", num_nodes = " << state.nodes_executed.size()
            << ", execution_time = " << state.GetCurrTime().count();
    VLOG(1) << "Per-op execution time:";
    for (const auto& op_cost_pair : state.op_to_cost) {
      const auto& op = op_cost_pair.first;
      const auto& cost = op_cost_pair.second.execution_time.count();
      if (cost) {  // Skip printing out zero-cost ops.
        VLOG(1) << " + " << op << " : " << cost;
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
