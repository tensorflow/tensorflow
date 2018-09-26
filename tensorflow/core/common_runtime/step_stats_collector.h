/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Allocator;
class AllocatorMemoryUsed;
class CostModelManager;
class Graph;
class Node;
class NodeExecStats;
class OpKernelContext;
class StepStats;
class StepStatsCollector;
class Tensor;
class TrackingAllocator;

// Statistics collection interface for individual node execution.
//
// See `NodeExecStatsWrapper` for a concrete implementation of this interface
// that interfaces with the `Session` layer.
class NodeExecStatsInterface {
 public:
  virtual ~NodeExecStatsInterface() {}

  // Called when the statistics collection for the node has finished. Once this
  // method is called, the caller should not make assumptions about the validity
  // of this object.
  virtual void Done(const string& device) = 0;

  // Called immediately after this node starts being processed by the executor.
  virtual void RecordExecutorStarted() = 0;

  // Called immediately before this node's `Compute()` or `ComputeAsync()`
  // method is called.
  virtual void RecordComputeStarted() = 0;

  // Called immediately after this node's `Compute()` method returned (or, for
  // asynchronous operations, the callback passed to its `ComputeAsync()` method
  // was called).
  virtual void RecordComputeEnded() = 0;

  // Called immediately after this executor finishes processing this node.
  virtual void RecordExecutorEnded() = 0;

  // Records information about the memory allocated during the execution of this
  // node.
  virtual void SetMemory(OpKernelContext* ctx) = 0;

  // Records information about the tensor produced by this node at the given
  // output slot.
  virtual void SetOutput(int slot, const Tensor* tensor) = 0;

  // Records information about the tensors that were accessed during the
  // execution of this node.
  virtual void SetReferencedTensors(const TensorReferenceVector& tensors) = 0;

  // Records the absolute time in nanoseconds at which this node became
  // runnable (i.e. was scheduled for execution).
  virtual void SetScheduled(int64 nanos) = 0;
};

// Wraps NodeExecStats and adds allocation to it.
class NodeExecStatsWrapper : public NodeExecStatsInterface {
 public:
  // Does not take ownership of `node` or `step_stats_collector`.
  NodeExecStatsWrapper(const Node* node,
                       StepStatsCollector* step_stats_collector);

  // Takes ownership of 'stats' but not `node` or `step_stats_collector`.
  NodeExecStatsWrapper(std::unique_ptr<NodeExecStats> stats, const Node* node,
                       StepStatsCollector* step_stats_collector);

  // Destructor calls Finalize() to release the TrackingAllocators.
  ~NodeExecStatsWrapper() { Finalize(); }

  void Done(const string& device) override;
  void RecordExecutorStarted() override;
  void RecordComputeStarted() override;
  void RecordComputeEnded() override;
  void RecordExecutorEnded() override;
  void SetMemory(OpKernelContext* ctx) override;
  void SetOutput(int slot, const Tensor* tensor) override;
  void SetReferencedTensors(const TensorReferenceVector& tensors) override;
  void SetScheduled(int64 nanos) override;

 private:
  friend class StepStatsCollector;

  NodeExecStats* stats() { return stats_.get(); }

  // Populates stats_ and releases TrackingAllocator.
  void Finalize();

  // Does not take ownership of the `allocator`.
  // Takes ownership of `tracking_allocator`.
  void AddAllocation(Allocator* allocator,
                     TrackingAllocator* tracking_allocator);

  gtl::InlinedVector<std::pair<AllocatorMemoryUsed*, TrackingAllocator*>, 2>
      allocations_;
  std::unique_ptr<NodeExecStats> stats_;
  const Node* const node_;                          // Not owned.
  StepStatsCollector* const step_stats_collector_;  // Not owned.
};

// Statistics collection interface for step execution.
//
// See `StepStatsCollector` for a concrete implementation of this interface
// that interfaces with the `Session` layer.
class StepStatsCollectorInterface {
 public:
  virtual ~StepStatsCollectorInterface() {}

  // Creates an instance of `NodeExecStatsInterface` that should be used for
  // collecting statistics about individual node execution.
  virtual NodeExecStatsInterface* CreateNodeExecStats(const Node* node) = 0;

  // Generates a string reporting the currently used memory based
  // on ResourceExhausted OOM `err` message.
  // `err` message needs to contain device name and allocator name, e.g.:
  // "ResourceExhaustedError: OOM when allocating tensor ...
  // on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc"
  virtual string ReportAllocsOnResourceExhausted(const string& err) = 0;
};

// StepStatsCollector manages the collection of a StepStats object.
// The StepStats object holds multiple DeviceStats.
// Each DeviceStats object holds multiple NodeExecStats.
class StepStatsCollector : public StepStatsCollectorInterface {
 public:
  // Does not take ownership of `step_stats`.
  explicit StepStatsCollector(StepStats* step_stats);

  // BuildCostModel builds or updates a CostModel managed by cost_model_manager,
  // using the currently collected DeviceStats associated with the devices in
  // device_map.
  void BuildCostModel(
      CostModelManager* cost_model_manager,
      const std::unordered_map<string, const Graph*>& device_map);

  // Saves node statistics to the DeviceStats object associated with device.
  // Should be called before Finalize.
  void Save(const string& device, NodeExecStats* node_stats_pb);
  void Save(const string& device, NodeExecStatsWrapper* node_stats);

  NodeExecStatsInterface* CreateNodeExecStats(const Node* node) override;
  string ReportAllocsOnResourceExhausted(const string& err) override;

  // The following 2 Finalize methods populate the StepStats passed
  // from the constructor. Calling it more than once won't have any effect.
  // User shouldn't call Save() methods after Finalize.
  void Finalize();
  // swaps the content of StepStats* from constructor with 'ss'.
  void FinalizeAndSwap(StepStats* step_stats);

 private:
  // TODO(suharshs): Make this configurable if its not possible to find a value
  // that works for all cases.
  static const uint64 kMaxCollectedNodes = 1 << 20;

  typedef std::vector<std::unique_ptr<NodeExecStatsWrapper>> NodeStatsVector;

  void FinalizeInternal() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  bool finalized_ GUARDED_BY(mu_);
  std::unordered_map<string, NodeStatsVector> dev_stats_ GUARDED_BY(mu_);
  StepStats* step_stats_ GUARDED_BY(mu_);
  uint64 collected_nodes_ GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
