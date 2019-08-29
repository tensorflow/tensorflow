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

  // How many times this node has been executed, e.g. in a while loop.
  int execution_count;

  // Output shape incompatible between shape annotation and shape inference.
  bool shape_incompatible;

  NodeState() {
    num_inputs_ready = 0;
    time_ready = Costs::Duration::max();
    time_scheduled = Costs::Duration::max();
    time_finished = Costs::Duration::max();
    execution_count = 0;
    shape_incompatible = false;
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
  std::map<string, Costs> op_to_cost;  // Per-op cost.

  int64 memory_usage;      // Current temporary memory usage
  int64 max_memory_usage;  // Max temporary memory usage

  // Shape annotation statistics.
  struct ShapeAnnotationStats {
    // Number of ops with shape annotated.
    int64 num_ops_annotated = 0;
    // Number of ops executed multiple times (e.g. in a loop).
    int64 num_ops_executed_more_than_once = 0;
    // Number of ops executed: account for execution count.
    int64 num_ops_executed = 0;
    // Number of ops with dynamic shapes (e.g. shape changes in a loop).
    int64 num_ops_with_dynamic_shapes = 0;
    // Number of ops with incompatible shapes between annotation and shape
    // inference.
    int64 num_ops_with_incompatible_shapes = 0;
  } shape_annotation_stats;

  DeviceState() {
    device_costs = Costs::ZeroCosts();
    device_costs.num_ops_total = 0;
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
  virtual Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) {
    return Status::OK();
  }
  virtual void AddNode(const NodeDef* node) = 0;
  virtual const NodeDef* GetCurrNode() = 0;
  virtual void RemoveCurrNode() = 0;
  virtual bool Empty() const = 0;
};

class FIFOManager : public ReadyNodeManager {
 public:
  FIFOManager() : ReadyNodeManager() {}
  ~FIFOManager() override {}
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

// Abstract class that maintains a heap/priority queue for scheduling ready
// nodes. Derived class needs to implement the Greater() function which returns
// the comparator for the heap.
class HeapReadyManager : public ReadyNodeManager {
 public:
  HeapReadyManager();
  Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) override;
  ~HeapReadyManager() override {}
  void AddNode(const NodeDef* node) override { waiting_queue_.push_back(node); }
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override;

 protected:
  virtual std::function<bool(const NodeDef*, const NodeDef*)> Greater() = 0;
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
  const std::unordered_map<const NodeDef*, NodeState>* node_map_;
};

// FirstReadyManager picks a node with the minimum time_ready value.
// Behavior is deterministic when there are more than one nodes with the minimum
// time_ready value with unique node names as the tie-breaker.
class FirstReadyManager : public HeapReadyManager {
 public:
  FirstReadyManager() : HeapReadyManager() {}
  ~FirstReadyManager() override {}

 protected:
  std::function<bool(const NodeDef*, const NodeDef*)> Greater() override;
};

// PriorityReadyManager uses the given node priorities when picking up next node
// from all the ready nodes.
class PriorityReadyManager : public HeapReadyManager {
 public:
  PriorityReadyManager() : HeapReadyManager() {}
  ~PriorityReadyManager() override {}
  void AddNode(const NodeDef* node) override;

  // Note this should be called after Init().
  Status SetPriority(const std::unordered_map<string, int>& node_priority);

 protected:
  std::function<bool(const NodeDef*, const NodeDef*)> Greater() override;

 private:
  // A map from unique node name to priority. Lower number means higher
  // priority.
  std::unordered_map<string, int> node_priority_;
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

  Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) override;
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
  // Not owned by CompositeReadyManager.
  const std::unordered_map<const NodeDef*, NodeState>* node_map_;

  // Cached curr node. Set back to nullptr from RemoveCurrNode().
  const NodeDef* curr_node_;
};

// Constructs a ready node manager from the given string.
std::unique_ptr<ReadyNodeManager> ReadyNodeManagerFactory(
    const string& ready_node_manager);

// The virtual scheduler emulates execution of nodes in a graph, considering
// dependencies, device, etc.
class VirtualScheduler {
 public:
  // Does not take ownership of cluster or ready_nodes.
  VirtualScheduler(const bool use_static_shapes,
                   const bool use_aggressive_shape_inference, Cluster* cluster,
                   ReadyNodeManager* ready_nodes,
                   std::unique_ptr<VirtualPlacer> placer);

  // Initializes the scheduler for the specific grappler item.
  // Should be called immediately after the c'tor or when the scheduler will be
  // reused for a new grappler item. All internal states of the scheduler
  // related to the previous grappler item will be reset/cleared.
  //
  // This function should be called at least once after the scheduler is
  // constructed. An uninitialized or failed-to-initialize scheduler will cause
  // undefined behavior.
  Status Init(const GrapplerItem* item);

  OpContext GetCurrNode() const;

  // Returns true if there is any node to be scheduled.
  bool MarkCurrNodeExecuted(const Costs& node_costs);

  // Prints out summary of execution (timing, memory usage, etc.)
  Costs Summary() const;
  // Like the above, but writes detailed stats to RunMetadata.
  // If metadata is nullptr, then just calls and return Summary().
  Costs Summary(RunMetadata* metadata);
  // Generates RunMetadata's step_stats and partition_graphs fields from results
  // of the virtual execution of the graph.
  void GenerateRunMetadata(RunMetadata* metadata);

  // Returns per device memory usage.
  const std::unordered_map<string, int64> GetPeakMemoryUsage() const;
  const std::unordered_map<string, int64> GetPersistentMemoryUsage() const;

  // Returns VirtualScheduler (read only) device and node states.
  const std::unordered_map<string, DeviceState>* GetDeviceStates() const {
    return &device_;
  }
  const std::unordered_map<const NodeDef*, NodeState>* GetNodeStates() const {
    return &node_map_;
  }

  void enable_mem_usage_tracking() { track_mem_usage_snapshot_ = true; }

 private:
  // Methods called from Init(). Fails if initialize_ is set.
  void MaybeUpdateInputOutput(const NodeDef* node);
  NodeState& GetNodeStateOrCreateIt(const NodeDef* node);
  std::pair<const NodeDef*, const NodeDef*> CreateSendRecv(
      const NodeDef* from, const NodeDef* to, const NodeDef* input_node,
      const string& input_name);
  string DeviceName(const NodeDef* node) const;
  string SanitizedDeviceName(const NodeDef* node) const;
  string ChannelDeviceName(const NodeDef* from, const NodeDef* to) const;

  // Helper methods.
  void AddOutputNodesToReadyQueue(const NodeDef* node,
                                  const Costs::Duration& curr_time);

  // Scheduler states:
  ReadyNodeManager* ready_nodes_;  // Not owned.
  std::unordered_map<const NodeDef*, NodeState> node_map_;
  std::unordered_map<string, DeviceState> device_;

  // Pool of NodeDefs for SendRecv and Identity ops created.
  std::vector<std::unique_ptr<NodeDef>> additional_nodes_;

  // Stats:
  // Op counts with key with input shape.
  // Example key: "[Op=AssignSub, input_shapes=[[7,1,160,160][7,1,160,160]]"
  std::map<string, int> op_counts_;
  // Individual op costs with key with input shape.
  // Integer field for execution time in micro seconds.
  // Boolean field for whether the cost is accurate.
  std::map<string, std::pair<int, bool>> op_costs_;

  Costs graph_costs_;                   // Graph cost.
  std::map<string, Costs> op_to_cost_;  // Per-op cost.

  // Auxiliary data structures for constructing NodeState and DeviceState.
  std::unique_ptr<GraphProperties> graph_properties_;  // Initialized in Init().
  Cluster* cluster_;                                   // Not owned.

  const GrapplerItem* grappler_item_;  // Not owned.
  bool use_static_shapes_;
  bool initialized_;
  bool track_mem_usage_snapshot_;
  const bool use_aggressive_shape_inference_;

  std::unique_ptr<VirtualPlacer> placer_;
};

}  // namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
