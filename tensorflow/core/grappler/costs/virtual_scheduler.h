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
  std::vector<const NodeDef*> inputs;
  std::vector<const NodeDef*> outputs;
  int num_inputs_ready;
  int num_outputs_executed;
  Costs::Duration time_ready;
  Costs::Duration time_scheduled;
  Costs::Duration time_finished;
  Costs::Duration time_no_reference;

  // Node will be ready to be executed at time_ready, scheduled at
  // time_scheduled, and finishes execution at time_finished.
  // Between time_scheduled and time_no_reference, the node's output tensor
  // needs to be on the device, using up device memory.

  NodeState() {
    num_inputs_ready = 0;
    num_outputs_executed = 0;
    time_ready = Costs::Duration::max();
    time_scheduled = Costs::Duration::max();
    time_finished = Costs::Duration::max();
    time_no_reference = Costs::Duration::max();
  }
};

struct DeviceState {
  std::vector<const NodeDef*> nodes_executed;
  Costs device_costs;
  std::map<string, Costs> op_to_cost;  // Per-op cost.

  DeviceState() { device_costs = Costs::ZeroCosts(); }

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
  std::vector<OpInfo::TensorProperties> outputs;
};

// The virtual scheduler emulates execution of nodes in a graph, considering
// dependencies, device, etc.
class VirtualScheduler {
 public:
  VirtualScheduler(const GrapplerItem* grappler_item,
                   const bool use_static_shapes,
                   const string& default_device_type, Cluster* cluster,
                   VirtualPlacer* placer);

  Status Init();

  NodeInfo GetCurrNodeInfo() const;
  bool MarkCurrNodeExecuted(const Costs& node_costs);

  Costs Summary() const;

 private:
  const string kSend = "_Send";
  const string kRecv = "_Recv";
  const string kAttrInputSrc = "input_source_";
  const string kAttrSrcDevice = "src_device_";
  const string kAttrDstDevice = "dst_device_";
  const string kChannelDevice = "Channel";
  const string kDefaultDevice = "/CPU:0";

  const NodeDef* GetCurrNode() const;
  void MaybeUpdateInputProperties(
      const NodeDef* node, std::vector<OpInfo::TensorProperties>* inputs) const;
  NodeState& GetNodeStateOrCreateIt(const NodeDef* node);
  std::pair<const NodeDef*, const NodeDef*> TransferNode(
      const NodeDef* from, const NodeDef* to, const string& input_name);
  string DeviceName(const NodeDef* node) const;
  string ChannelDeviceName(const NodeDef* from, const NodeDef* to) const;
  Costs& FindOrCreateZero(const string& op_name,
                          std::map<string, Costs>* op_cost);
  bool IsSendOp(const NodeDef* node) const;
  bool IsRecvOp(const NodeDef* node) const;

  GraphProperties graph_properties_;
  std::map<string, int> op_counts_;  // Op counts with key with input shape.
  std::map<string, int> op_costs_;   // Individual op costs (with input shapes).
  Costs graph_costs_;                   // Graph cost.
  std::map<string, Costs> op_to_cost_;  // Per-op cost.
  std::unique_ptr<ReadyNodeManager> ready_nodes_;
  std::unordered_map<const NodeDef*, NodeState> node_map_;
  std::unordered_map<string, DeviceState> device_;
  // Pool of NodeDefs for SendRecv and Identity ops created.
  std::vector<std::unique_ptr<NodeDef>> additional_nodes_;
  // Cache of ops transferred to another device.
  std::unordered_map<const NodeDef*, std::unordered_map<string, const NodeDef*>>
      cached_ops_;
  Cluster* cluster_;                   // Not owned.
  const GrapplerItem* grappler_item_;  // Not owned.
  bool use_static_shapes_;
  bool initialized_;

  // TODO(dyoon): Once VirtualCluster takes care of device names properly,
  // move VirtualPlacer into the scheduler; also, delete default_device_type_.
  const string default_device_type_;
  VirtualPlacer* placer_;  // Not owned.
};

}  // namespace grappler
}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
