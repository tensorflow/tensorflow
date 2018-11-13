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
class Tensor;
class TrackingAllocator;

// Wraps NodeExecStats and adds allocation to it.
class NodeExecStatsWrapper {
 public:
  NodeExecStatsWrapper(const string& node_name);
  // Owns 'stats'.
  NodeExecStatsWrapper(NodeExecStats* stats);

  // Destructor calls Finalize() to release the TrackingAllocators.
  ~NodeExecStatsWrapper() { Finalize(); }

  // Records the absolute time in nanoseconds at which this node became
  // runnable (i.e. was scheduled for execution).
  void SetScheduled(int64 nanos) {
    stats_->set_scheduled_micros(nanos / EnvTime::kMicrosToNanos);
    stats_->set_scheduled_nanos(nanos);
  }

  // Called immediately after this node starts being processed by the executor.
  void RecordExecutorStarted() {
    int64 now_nanos = Env::Default()->NowNanos();
    stats_->set_all_start_micros(now_nanos / EnvTime::kMicrosToNanos);
    stats_->set_all_start_nanos(now_nanos);
  }

  // Called immediately before this node's `Compute()` or `ComputeAsync()`
  // method is called.
  void RecordComputeStarted() {
    int64 now_nanos = Env::Default()->NowNanos();
    DCHECK_NE(stats_->all_start_micros(), 0);
    DCHECK_NE(stats_->all_start_nanos(), 0);
    stats_->set_op_start_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                    stats_->all_start_micros());
    stats_->set_op_start_rel_nanos(now_nanos - stats_->all_start_nanos());
  }

  // Called immediately after this node's `Compute()` method returned (or, for
  // asynchronous operations, the callback passed to its `ComputeAsync()` method
  // was called).
  void RecordComputeEnded() {
    int64 now_nanos = Env::Default()->NowNanos();
    DCHECK_NE(stats_->all_start_micros(), 0);
    DCHECK_NE(stats_->all_start_nanos(), 0);
    stats_->set_op_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                  stats_->all_start_micros());
    stats_->set_op_end_rel_nanos(now_nanos - stats_->all_start_nanos());
  }

  // Called immediately after this executor finishes processing this node.
  void RecordExecutorEnded() {
    int64 now_nanos = Env::Default()->NowNanos();
    DCHECK_NE(stats_->all_start_micros(), 0);
    DCHECK_NE(stats_->all_start_nanos(), 0);
    stats_->set_all_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                   stats_->all_start_micros());
    stats_->set_all_end_rel_nanos(now_nanos - stats_->all_start_nanos());
  }

  // Records information about the tensor produced by this node at the given
  // output slot.
  void SetOutput(int slot, const Tensor* v);

  // Records information about the memory allocated during the execution of this
  // node.
  void SetMemory(OpKernelContext* ctx);

  // Records information about the tensors that were accessed during the
  // execution of this node.
  void SetReferencedTensors(const TensorReferenceVector& tensors);

  // Sets the timeline_label field of the wrapped NodeExecStats, using data
  // from *node. Returns true iff the node is a transfer node.
  bool SetTimelineLabel(const Node* node);

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
};

// Statistics collection interface for individual node execution.
//
// See `StepStatsCollector` for a concrete implementation of this interface
// that interfaces with the `Session` layer.
class StepStatsCollectorInterface {
 public:
  virtual ~StepStatsCollectorInterface() {}

  // Saves `stats` to the collector.
  virtual void Save(const string& device, NodeExecStatsWrapper* stats) = 0;

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
  // Does not take ownership of `ss`.
  explicit StepStatsCollector(StepStats* ss);

  // BuildCostModel builds or updates a CostModel managed by cost_model_manager,
  // using the currently collected DeviceStats associated with the devices in
  // device_map.
  void BuildCostModel(
      CostModelManager* cost_model_manager,
      const std::unordered_map<string, const Graph*>& device_map);

  // Save saves nt to the DeviceStats object associated with device.
  // Should be called before Finalize.
  void Save(const string& device, NodeExecStats* nt);
  void Save(const string& device, NodeExecStatsWrapper* stats) override;

  string ReportAllocsOnResourceExhausted(const string& err) override;

  // The following 2 Finalize methods populate the StepStats passed
  // from the constructor. Calling it more than once won't have any effect.
  // User shouldn't call Save() methods after Finalize.
  void Finalize();
  // swaps the content of StepStats* from constructor with 'ss'.
  void FinalizeAndSwap(StepStats* ss);

 private:
  void FinalizeInternal() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  typedef std::vector<std::unique_ptr<NodeExecStatsWrapper>> NodeExecStatsVec;
  // TODO(suharshs): Make this configurable if its not possible to find a value
  //                 that works for all cases.
  const uint64 kMaxCollectedNodes = 1 << 20;
  mutex mu_;
  bool finalized_ GUARDED_BY(mu_);
  std::unordered_map<string, NodeExecStatsVec> dev_stats_ GUARDED_BY(mu_);
  StepStats* step_stats_ GUARDED_BY(mu_);
  uint64 collectedNodes GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
