/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_ASYNC_TRACKER_H_
#define XLA_SERVICE_ASYNC_TRACKER_H_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/latency_estimator.h"
#include "xla/service/scheduler_config.h"  // IWYU pragma: export
#include "xla/service/scheduler_core.h"  // IWYU pragma: export

namespace xla {

// Defined in latency_hiding_scheduler.h, which includes this header.
// `AsyncTracker` only uses them by pointer.
class HloGraphNode;
class HloScheduleGraph;

enum class ResourceType {
  kNoResource = 0,
  kAllToAll,
  kAllGather,
  kAllReduce,
  kCollectivePermute,
  kCopy,
  kReduceScatter,
  kSendRecv,
  kSendHost,
  kRecvHost,
  kCollectiveBroadcast,
  kNumResources,
  kRaggedAllToAll,
  kTargetDefinedResourceTypeBegin,
};

enum class ResourceUsageType {
  kNoResource,
  kResourceOccupy,
  kResourceRelease,
};

enum class ResourceHazardType {
  kShareable = 0,
  kSerial = 1,
  // The following hazard type represents the resources that are used by the
  // async ops and should be released right after the estimated time cost has
  // past. This hazard type is useful to prevent increasing such ops' overlaps
  // more than necessary.
  kNonextendable = 2,
  // Ops holding this resource can only have their latency/cost covered by
  // ops that are valuable for selective overlap.
  kSelective = 3,
  kUnshareable = 4,
  // An in-order resource (e.g. hardware FIFO command queue) where operations
  // are processed sequentially. Instructions calling a synchronous nested
  // computation using this resource cannot be scheduled while any outer
  // asynchronous operation on this resource is in flight.
  kInOrder = 5,
};

template <typename T, typename = typename std::enable_if_t<std::is_enum_v<T>>>
constexpr int64_t ResourceTypeToIndex(T resource_type) {
  return static_cast<int64_t>(resource_type);
}

constexpr int64_t ResourceUsageTypeToIndex(
    ResourceUsageType resource_usage_type) {
  return static_cast<int64_t>(resource_usage_type);
}

using ResourcePair = std::pair<int64_t, ResourceUsageType>;
using ResourcesVector = absl::InlinedVector<ResourcePair, 1>;

// Helper class to keep track of which instructions are to be supported and
// how many supported instructions per-type are contained in computations
// recursively.
class AsyncTracker {
 public:
  virtual ~AsyncTracker() = default;

  // Returns if this is an Async op done that the scheduler supports.
  virtual bool IsSupportedAsyncDone(const HloInstruction& hlo) const;

  // Returns if this is an Async op start that the scheduler supports.
  virtual bool IsSupportedAsyncStart(const HloInstruction& hlo) const;

  // Returns resources used (i.e., occupied or released) by this instruction
  virtual ResourcesVector GetResourcesFromInstructionImpl(
      const HloInstruction& hlo) const;

  // Gets the resource type associated with the given op.
  static ResourceType GetResourceTypeForOp(HloOpcode op);

  // Returns resources used (i.e., occupied or released) by this instruction
  absl::Span<const ResourcePair> GetResourcesFromInstruction(
      const HloInstruction& hlo) const;

  // Modifies the schedule graph passed as input to add dependencies that are
  // implicit based on the system we are running on.
  virtual void PostProcessScheduleGraph(
      HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator) const {}

  // Returns the number of resources (of type resource_type) that are used by
  // this instruction.
  virtual int64_t GetNumResourcesPerInstruction(
      ResourceType resource_type, const HloInstruction& instr) const;
  virtual int64_t GetNumResourcesPerInstruction(
      int64_t resource_type, const HloInstruction& instr) const;

  // Returns a map of number of resources used per resource type by this
  // instruction.
  virtual absl::flat_hash_map<int64_t, int64_t> GetNumResourcesPerInstruction(
      const HloInstruction& instr) const;

  // Sets the maximum allowed number of instances for each resource
  virtual void SetConcurrentResourceLimits(
      absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource) const;

  // Returns the name of the given resource
  virtual absl::string_view GetResourceName(int64_t resource_type) const;

  // Returns the name of the given resource usage
  absl::string_view GetResourceUsageName(int64_t resource_usage_type) const;
  absl::string_view GetResourceUsageName(
      ResourceUsageType resource_usage_type) const;

  // Returns the first target defined resource's id, regardless of if it exits
  static int64_t GetTargetDefinedResourceTypeBegin() {
    return ResourceTypeToIndex(ResourceType::kTargetDefinedResourceTypeBegin);
  }

  // Returns the number of target defined resources
  virtual int64_t GetNumTargetDefinedResources() const;

  // Returns how many instructions using the given resource_type we can overlap
  virtual int64_t GetNumAvailableResources(int64_t resource_type) const;

  // Returns the hazard type that describes how to resolve the conflicts when
  // multiple instructions attempt to use the given resource type concurrently.
  // Default resources have a hazard type of kUnshareable.
  virtual ResourceHazardType GetResourceHazardType(int64_t resource_type) const;

  // Returns the list of the released shareable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetReleasedShareableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the occupied shareable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetOccupiedShareableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the occupied serial resources filtered from the given
  // resources vector.
  virtual absl::InlinedVector<int64_t, 1> GetOccupiedSerialResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns the list of the released nonextendable resources filtered from the
  // given resources vector.
  virtual absl::InlinedVector<int64_t, 1>
  GetReleasedNonextendableResourcesFromVector(
      const ResourcesVector& resources) const;

  // Returns whether the provided node releases a selective resource.
  bool ReleasesSelectiveResource(const HloGraphNode* node) const;

  // Returns whether the provided node occupies a selective resource.
  bool OccupiesSelectiveResource(const HloGraphNode* node) const;

  // Returns whether the resource is an in-order resource (e.g. hardware FIFO
  // command queue) where operations are processed sequentially and cannot be
  // interleaved across nested computation boundaries.
  bool IsInorderResource(int64_t resource_type) const {
    return GetResourceHazardType(resource_type) == ResourceHazardType::kInOrder;
  }

  CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) const {
    return get_canonical_async_op_(hlo);
  }

  // Updates target defined states after scheduling a node.
  virtual void UpdateTargetDefinedStates(
      const HloInstruction& hlo, const HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator,
      LatencyEstimator::TimeCost current_time,
      SchedulerCore::SchedulingState* sched_state) {}

  // Updates target defined states after scheduling a computation.
  virtual void UpdateTargetDefinedStates(
      HloComputation* computation,
      SchedulerCore::SchedulingState* sched_state) {}

  // Resets target defined states after scheduling a computation.
  virtual void ResetTargetDefinedStates(
      SchedulerCore::SchedulingState* sched_state) {}

  const SchedulerConfig& GetConfig() const { return config_; }

  // Overrides SchedulerConfig::enable_cpd_for_sync_collective. Used by the
  // scheduler to reschedule with a different schedule-by-structure setting.
  void SetEnableCpdForSyncCollective(bool enable) {
    config_.enable_cpd_for_sync_collective = enable;
  }

  // Clears the cache of per-computation resource maps. This is needed when,
  // e.g., we modify the schedule of a computation, which could change the
  // resource usage of the computation.
  void InvalidateCache() {
    absl::MutexLock lock(&async_in_computation_cache_mu_);
    async_in_computation_cache_.clear();
  }

  // Similar to InvalidateCache(), but only invalidates the cache for the given
  // computation.
  void InvalidateCache(const HloComputation* computation) {
    absl::MutexLock lock(&async_in_computation_cache_mu_);
    async_in_computation_cache_.erase(computation);
  }

  // Returns whether the async tracker is using top down scheduling.
  bool IsTopDownScheduling() const { return config_.top_down_scheduling; }

  explicit AsyncTracker(
      const SchedulerConfig& config,
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : get_canonical_async_op_(std::move(func)), config_(config) {}

 private:
  // Returns the number of "occupy" type of resources used by the instructions
  // in the given computation. Uses the scheduling information if available to
  // obtain more accurate resource usage. If an instruction uses multiple
  // instances of the same "occupy" type of resource, that number is respected
  // and returned in the resulting map.
  const absl::flat_hash_map<int64_t, int64_t>& RecursivelyComputeResourceMap(
      const HloComputation* computation) const;
  // Similar as above, but uses scheduling information to obtain more accurate
  // resource usage. Useful for non-fusion computations.
  // REQUIRES: The computation must be scheduled.
  const absl::flat_hash_map<int64_t, int64_t>&
  RecursivelyComputeResourceMapForScheduledComputation(
      const HloComputation* computation) const;

  mutable absl::Mutex async_in_computation_cache_mu_;
  mutable absl::flat_hash_map<
      const HloComputation*,
      std::unique_ptr<absl::flat_hash_map<int64_t, int64_t>>>
      async_in_computation_cache_
          ABSL_GUARDED_BY(async_in_computation_cache_mu_);
  GetCanonicalAsyncOpFunc get_canonical_async_op_;

 protected:
  SchedulerConfig config_;
  mutable absl::Mutex resources_cache_mu_;
  mutable absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<ResourcesVector>>
      resources_cache_ ABSL_GUARDED_BY(resources_cache_mu_);
};

}  // namespace xla

#endif  // XLA_SERVICE_ASYNC_TRACKER_H_
