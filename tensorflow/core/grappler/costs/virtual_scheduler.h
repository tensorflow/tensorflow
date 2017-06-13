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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_

#include <list>
#include <memory>
#include <unordered_map>

#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

struct NodeState {
  // A node (i.e., an op) takes a set of input:port pairs and produces
  // a set of output ports.

  // Cross references to input and output nodes from graphdef.
  std::vector<std::pair<const NodeDef*, int>> inputs;  // Input, port pairs.
  // List of output nodes (a list of nodes that takes this output port as input)
  // keyed by port_num. Note that port_num -1 is used for control dependency.
  std::unordered_map<int, std::vector<const NodeDef*>> outputs;

  // Info from GraphProperties.
  std::vector<OpInfo::TensorProperties> input_properties;
  std::vector<OpInfo::TensorProperties> output_properties;

  // Canonical device name used within VirtualScheduler.
  string device_name;

  // States updated as scheduling nodes.
  int num_inputs_ready;
  std::unordered_map<int, int> num_outputs_executed;
  Costs::Duration time_ready;
  Costs::Duration time_scheduled;
  Costs::Duration time_finished;
  // Time that all the consumers are executed (hence, no need to keep this
  // output in memory), keyed by port_num.
  std::unordered_map<int, Costs::Duration> time_no_references;

  // Note that a node may have multiple output ports. The length of outputs,
  // num_outputs_executed, and time_no_references should be
  // identical when a NodeState is fully initialized.
  // They should be 1 + output_properties.size() as we add [-1] for control
  // dependency.

  // Node will be ready to be executed at time_ready, scheduled at
  // time_scheduled, and finishes execution at time_finished.
  // Each output port uses up memory space from time_scheduled to its
  // time_no_references.

  NodeState() {
    num_inputs_ready = 0;
    time_ready = Costs::Duration::max();
    time_scheduled = Costs::Duration::max();
    time_finished = Costs::Duration::max();
    // Note that num_outputs_executed and time_no_references are not initialized
    // here, since we don't know the size (i.e., # outputs for this node).
  }
};

struct DeviceState {
  // Nodes executed on this device in execution order.
  std::vector<const NodeDef*> nodes_executed;

  // Nodes currently allocated in memory: set of NodeDef* and port_num pairs
  // so that we can track which output of the node is in memory.
  std::set<std::pair<const NodeDef*, int>> nodes_in_memory;

  // Nodes allocated in memory persistently: e.g., Variables.
  std::set<std::pair<const NodeDef*, int>> persistent_nodes;

  // Snapshot of nodes_in_memory, when memory usage is at peak.
  // Same to nodes_in_memory, it's a set of NodeDef* and port_num pairs.
  std::set<std::pair<const NodeDef*, int>> mem_usage_snapshot_at_peak;

  Costs device_costs;
  std::map<string, Costs> op_to_cost;    // Per-op cost.
  std::map<string, int64> op_to_memory;  // Per-op memory usage at peak usage.
  int64 memory_usage;
  int64 max_memory_usage;

  DeviceState() {
    device_costs = Costs::ZeroCosts();
    memory_usage = 0;
    max_memory_usage = 0;
  }

  Costs::Duration GetCurrTime() const { return device_costs.execution_time; }
};

// ReadyNodeManager (abstract class):
// Keeps ready nodes and picks the best one to be scheduled.
class ReadyNodeManager {
 public:
  ReadyNodeManager() {}
  virtual ~ReadyNodeManager() {}
  virtual void AddNode(const NodeDef* node) = 0;
  virtual const NodeDef* GetCurrNode() const = 0;
  virtual void RemoveCurrNode() = 0;
  virtual bool Empty() const = 0;
};

class FIFOManager : public ReadyNodeManager {
 public:
  FIFOManager() : ReadyNodeManager() {}
  ~FIFOManager() override {}
  void AddNode(const NodeDef* node) override { nodes_.push_back(node); }
  const NodeDef* GetCurrNode() const override { return nodes_.front(); }
  void RemoveCurrNode() override { nodes_.pop_front(); }
  bool Empty() const override { return nodes_.empty(); }

 private:
  std::list<const NodeDef*> nodes_;
};

// A wrapper struct to OpInfo proto.
// TODO(dyoon): once we extend OpInfo or implement a better interface, and  then
// delete this wrapper struct.
struct NodeInfo {
  OpInfo op_info;
  string name;
  string device_name;
};

// The virtual scheduler emulates execution of nodes in a graph, considering
// dependencies, device, etc.
class VirtualScheduler {
 public:
  VirtualScheduler(const GrapplerItem* grappler_item,
                   const bool use_static_shapes, Cluster* cluster);

  // Initializes NodeState and DeviceState from grappler_item_ and
  // graph_properties_.
  Status Init();

  NodeInfo GetCurrNodeInfo() const;

  // Returns true if there is any node to be scheduled.
  bool MarkCurrNodeExecuted(const Costs& node_costs);

  // Prints out summary of execution (timing, memory usage, etc.)
  Costs Summary() const;

 protected:
  // GetDeviceStates and GetNodeStates are currently for testing purpuse only.
  // Retrieves detailed scheduling results.
  const std::unordered_map<string, DeviceState>& GetDeviceStates() const {
    return device_;
  }
  const std::unordered_map<const NodeDef*, NodeState>& GetNodeStates() const {
    return node_map_;
  }

  // Returns the size of output at port_num (unit: bytes). A special case is
  // port_num -1, which is for control dependency and assumed to be 4 bytes.
  int64 CalculateOutputSize(
      const std::vector<OpInfo::TensorProperties>& output_properties,
      const int port_num) const;

 private:
  // Constants.
  const string kAttrInputSrc = "input_source_";
  const string kAttrSrcDevice = "src_device_";
  const string kAttrDstDevice = "dst_device_";
  const string kChannelDevice = "Channel";

  // Methods called from Init(). Fails if initialize_ is set.
  void MaybeUpdateInputOutput(const NodeDef* node);
  NodeState& GetNodeStateOrCreateIt(const NodeDef* node);
  std::pair<const NodeDef*, const NodeDef*> CreateSendRecv(
      const NodeDef* from, const NodeDef* to, const string& input_name);
  string DeviceName(const NodeDef* node) const;
  string ChannelDeviceName(const NodeDef* from, const NodeDef* to) const;

  // Helper methods.
  Costs& FindOrCreateZero(const string& op_name,
                          std::map<string, Costs>* op_cost);
  float Round2(const float x) const;
  bool IsPersistentNode(const NodeDef* node) const;

  // Scheduler states:
  std::unique_ptr<ReadyNodeManager> ready_nodes_;
  std::unordered_map<const NodeDef*, NodeState> node_map_;
  std::unordered_map<string, DeviceState> device_;

  // Pool of NodeDefs for SendRecv and Identity ops created.
  std::vector<std::unique_ptr<NodeDef>> additional_nodes_;
  // Cache of nodes transferred to another device.
  std::unordered_map<const NodeDef*, std::unordered_map<string, const NodeDef*>>
      cached_recv_nodes_;

  // Stats:
  std::map<string, int> op_counts_;  // Op counts with key with input shape.
  std::map<string, int> op_costs_;   // Individual op costs (with input shapes).
  Costs graph_costs_;                // Graph cost.
  std::map<string, Costs> op_to_cost_;  // Per-op cost.

  // Auxilliary data structures for constructing NodeState and DeviceState.
  GraphProperties graph_properties_;
  Cluster* cluster_;                   // Not owned.
  const GrapplerItem* grappler_item_;  // Not owned.
  bool use_static_shapes_;
  bool initialized_;

  VirtualPlacer placer_;  // owned.
};

}  // namespace grappler
}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
