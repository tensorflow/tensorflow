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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

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
  VLOG(2) << "costs execution_time=" << result.execution_time.count()
          << " max_memory=" << result.max_memory
          << " max_per_op_buffers=" << result.max_per_op_buffers
          << " max_per_op_streaming=" << result.max_per_op_streaming;
  return result;
}
}  // namespace

VirtualScheduler::VirtualScheduler(const GraphDef& graph,
                                   const std::vector<string>& fetch_nodes)
    : graph_costs_(Costs::ZeroCosts()),
      // TODO(dyoon): Use a better way than FIFO.
      ready_nodes_(new FIFOManager()) {
  // First, get the nodes that would run to output fetch_nodes.
  std::vector<const NodeDef*> nodes =
      ComputeTransitiveFanin(graph, fetch_nodes);

  // TODO(dyoon): this is a bit inefficient as name_to_node is already built in
  // ComputeTransitiveFanin().
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : graph.node()) {
    name_to_node[node.name()] = &node;
  }

  // Build node_map.
  for (const auto* node : nodes) {
    auto& node_state = GetNodeStateOrCreateIt(node);
    // TODO(dyoon): add SendRecv considering devices and control dependency.
    for (const string& input : node->input()) {
      const NodeDef* in = name_to_node[NodeName(input)];
      CHECK(in);
      node_state.inputs.push_back(in);
      auto& input_node_state = GetNodeStateOrCreateIt(in);
      input_node_state.outputs.push_back(node);
    }
    if (node->input().empty()) {
      node_state.time_ready =
          Costs::Duration();  // Node without input: ready at time 0.
      ready_nodes_->AddNode(node);
    }
  }
}

const NodeDef* VirtualScheduler::GetCurrNode() const {
  return ready_nodes_->GetCurrNode();
}

NodeState& VirtualScheduler::GetNodeStateOrCreateIt(const NodeDef* node) {
  auto it = node_map_.find(node);
  if (it == node_map_.end()) {
    it = node_map_.emplace(node, NodeState()).first;
  }
  return it->second;
}

bool VirtualScheduler::MarkCurrNodeExecuted(const Costs& node_costs) {
  // Update graph_costs_ and per-op costs.
  graph_costs_ = CombineCosts(graph_costs_, node_costs);
  const auto* node = GetCurrNode();
  const auto& op_name = node->op();

  auto it = op_to_cost_.find(op_name);
  if (it == op_to_cost_.end()) {
    it = op_to_cost_.emplace(op_name, Costs::ZeroCosts()).first;
  }
  auto& op_cost = it->second;
  op_cost = CombineCosts(op_cost, node_costs);

  // Update node and device states.
  auto& node_state = node_map_[node];
  auto& device = device_[node->device()];
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
  {
    auto it = device.op_to_cost.find(op_name);
    if (it == device.op_to_cost.end()) {
      it = device.op_to_cost.emplace(op_name, Costs::ZeroCosts()).first;
    }
    auto& op_cost = it->second;
    op_cost = CombineCosts(op_cost, node_costs);

    VLOG(2) << "Op scheduled -- name: " << node->name()
            << ", op: " << node->op() << ", device: " << node->device()
            << ", ready: " << node_state.time_ready.count()
            << ", scheduled: " << node_state.time_scheduled.count()
            << ", finished: " << node_state.time_finished.count();

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
  }

  // Remove the current node; assume FIFO.
  ready_nodes_->RemoveCurrNode();
  return !ready_nodes_->Empty();  // True if not empty.
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

  VLOG(1) << "Critical path execution time: "
          << critical_path_costs.execution_time.count();
  return critical_path_costs;
}

}  // end namespace grappler
}  // end namespace tensorflow
