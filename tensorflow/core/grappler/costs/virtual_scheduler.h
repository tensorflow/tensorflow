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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_

#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_context.h"
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

  struct NodePairHash {
   public:
    const std::size_t operator()(
        const std::pair<const NodeDef*, int>& element) const {
      return std::hash<const NodeDef*>()(element.first);
    }
  };

  // Nodes currently allocated in memory: set of NodeDef* and port_num pairs
  // so that we can track which output of the node is in memory.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      nodes_in_memory;

  // Nodes allocated in memory persistently: e.g., Variables.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      persistent_nodes;

  // Snapshot of nodes_in_memory, when memory usage is at peak.
  // Same to nodes_in_memory, it's a set of NodeDef* and port_num pairs.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      mem_usage_snapshot_at_peak;

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
  virtual void Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_state) {}
  virtual void AddNode(const NodeDef* node) = 0;
  virtual const NodeDef* GetCurrNode() = 0;
  virtual void RemoveCurrNode() = 0;
  virtual bool Empty() const = 0;
};

class FIFOManager : public ReadyNodeManager {
 public:
  FIFOManager() : ReadyNodeManager() {}
  ~FIFOManager() override {}
  void Init(const std::unordered_map<const NodeDef*, NodeState>* node_state)
      override {}
  void AddNode(const NodeDef* node) override { nodes_.push_back(node); }
  const NodeDef* GetCurrNode() override {
    CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
    return nodes_.front();
  }
  void RemoveCurrNode() override { nodes_.pop_front(); }
  bool Empty() const override { return nodes_.empty(); }

 private:
  std::list<const NodeDef*> nodes_;
};

// The LIFOManager schedules nodes by returning the last one added to the
// scheduler. A node is executed and then its ready outputs are newly added to
// the scheduler, so the LIFOManager will return outputs to a node following
// that node's execution.
class LIFOManager : public ReadyNodeManager {
 public:
  LIFOManager() : ReadyNodeManager() {}
  ~LIFOManager() override {}
  void Init(const std::unordered_map<const NodeDef*, NodeState>* node_state)
      override {}
  void AddNode(const NodeDef* node) override { nodes_.push_back(node); }
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override { return nodes_.empty(); }

 private:
  std::list<const NodeDef*> nodes_;
  // Keep track of the current node being executed by saving its position.
  // Necessary because nodes may be added to the end of the list while a node is
  // executing, and we want to remove the correct node (the one that is
  // executing) rather than the new ones being added.
  std::list<const NodeDef*>::iterator curr_pos_ = nodes_.end();
};

// FirstReadyManager picks a node with the minimum time_ready value.
// Behavior is unknown if there are more than one nodes with the minimum
// time_ready value (it depends on C++ STL push_heap and pop_heap).
class FirstReadyManager : public ReadyNodeManager {
 public:
  FirstReadyManager();
  void Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_state) override;
  ~FirstReadyManager() override {}
  void AddNode(const NodeDef* node) override { waiting_queue_.push_back(node); }
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override;

 private:
  // Move all the nodes in the waiting_queue_ to nodes_.
  void DrainWaitingQueue();

  // nodes_ is the main queue, where we construct heap, and the front is the
  // current node.
  std::vector<const NodeDef*> nodes_;
  // Newly added nodes are added to waiting_queue_. That way, GetCurrNode(),
  // which returns the front of the nodes_, always returns the same node,
  // even if any of new nodes has time_ready smaller than the current node's.
  std::vector<const NodeDef*> waiting_queue_;
  // Comparator functor for heap; stl heap is max heap, so we use "greater than"
  // functor for keeping the smallest time_ready node at the front of heap.
  std::function<bool(const NodeDef*, const NodeDef*)> greater_;

  // NodeState structure from VirtualScheduler to get time_ready of ready nodes.
  // Not owned by FirstReadyManager.
  const std::unordered_map<const NodeDef*, NodeState>* node_state_;
};

// CompositeNodeManager has a few other NodeManagers: per-device LIFO for normal
// ops (neither _Send nor _Recv) and FirstReadyManagers for _Send ops and _Recv
// ops, and then it chooses FirstReady among the ops chosen from each
// internal NodeManagers. The objective is to maximize producer-consumer
// locality within device, while processing nodes across devices, including
// _Send and _Recv, fairly, in terms of their time_ready.
class CompositeNodeManager : public ReadyNodeManager {
 public:
  CompositeNodeManager();
  ~CompositeNodeManager() override {}

  void Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_state) override;
  void AddNode(const NodeDef* node) override;
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override;

 private:
  // Internal ready node managers:
  // LIFO for normal ops to maximize producer consumer locality.
  // One LIFO per device.
  std::unordered_map<string, LIFOManager> ops_lifo_map_;
  // FirstReady for send and recv. Handle send and recv separately ensures that
  // send and recv do not block previously read ops with LIFO schedule.
  FirstReadyManager send_manager_;
  FirstReadyManager recv_manager_;

  // NodeState structure from VirtualScheduler to get time_ready of ready nodes.
  // Not owned by FirstReadyManager.
  const std::unordered_map<const NodeDef*, NodeState>* node_state_;

  // Cached curr node. Set back to nullptr from RemoveCurrNode().
  const NodeDef* curr_node_;
};

// The virtual scheduler emulates execution of nodes in a graph, considering
// dependencies, device, etc.
class VirtualScheduler {
 public:
  VirtualScheduler(const GrapplerItem* grappler_item,
                   const bool use_static_shapes, Cluster* cluster,
                   ReadyNodeManager* ready_nodes);
  // Initializes NodeState and DeviceState from grappler_item_ and
  // graph_properties_.
  Status Init();

  OpContext GetCurrNode() const;

  // Returns true if there is any node to be scheduled.
  bool MarkCurrNodeExecuted(const Costs& node_costs);

  // Prints out summary of execution (timing, memory usage, etc.)
  Costs Summary() const;
  // Like the above, but writes detailed stats to RunMetadata.
  // If metadata is nullptr, then just calls and return Summary().
  Costs Summary(RunMetadata* metadata);
  // Methods called from constructor.
  static ReadyNodeManager* ReadyNodeManagerFactory(
      const string& ready_node_manager);

  // Return per device peak memory usage.
  const std::unordered_map<string, int64> GetPeakMemoryUsage() const;

 protected:
  const std::unordered_map<string, DeviceState>* GetDeviceStates() const {
    return &device_;
  }
  const std::unordered_map<const NodeDef*, NodeState>* GetNodeStates() const {
    return &node_map_;
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
  string SanitizedDeviceName(const NodeDef* node) const;
  string ChannelDeviceName(const NodeDef* from, const NodeDef* to) const;

  // Helper methods.
  Costs& FindOrCreateZero(const string& op_name,
                          std::map<string, Costs>* op_cost);
  float Round2(const float x) const;
  bool IsPersistentNode(const NodeDef* node) const;

  // Scheduler states:
  ReadyNodeManager* ready_nodes_;  // Not owned.
  std::unordered_map<const NodeDef*, NodeState> node_map_;
  std::unordered_map<string, DeviceState> device_;

  // Pool of NodeDefs for SendRecv and Identity ops created.
  std::vector<std::unique_ptr<NodeDef>> additional_nodes_;

  // Stats:
  std::map<string, int> op_counts_;  // Op counts with key with input shape.
  // Individual op costs (with input shapes).
  // Boolean field for whether the cost is accurate.
  std::map<string, std::pair<int, bool>> op_costs_;

  Costs graph_costs_;                   // Graph cost.
  std::map<string, Costs> op_to_cost_;  // Per-op cost.

  // Auxiliary data structures for constructing NodeState and DeviceState.
  GraphProperties graph_properties_;
  Cluster* cluster_;  // Not owned.

  const GrapplerItem* grappler_item_;  // Not owned.
  bool use_static_shapes_;
  bool initialized_;

  VirtualPlacer placer_;  // owned.
};

}  // namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
