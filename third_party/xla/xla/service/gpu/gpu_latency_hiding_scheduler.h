/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_LATENCY_HIDING_SCHEDULER_H_
#define XLA_SERVICE_GPU_GPU_LATENCY_HIDING_SCHEDULER_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

// Breaks down higher level collectives into collective primitives.
// E.g. AllReduceStart is broken down into Reduce + AsyncStart.
CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo);

// The shape size function depending on the pointer size and
// memory space.
HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction(
    int64_t pointer_size, std::optional<int64_t> memory_space = std::nullopt);

// GPU overlap limit rule rule for scheduling candidate.
// On top of the default rule, we do not allow collectives with more than 1
// overlapping ranks to overlap. This is because the execution order of NCCL
// kernels is not deterministic and cannot be controlled by launch order at the
// moment. A cyclic dependency can be formed with at least 2 overlapping ranks.
bool GpuScheduleCrossesOverlapLimit(
    const DefaultSchedulerCore::SchedulingState& sched_state,
    const HloGraphNode* node);

// GPU specific resources for latency hiding scheduler.
//
// We use two different set of resources to model the scheduling of asynchronous
// collective operations and P2P Send and Recv operations. This corresponds to
// the fact that the runtime use a stream to run asynchronous collective
// operations and two other streams to run P2P Send and Recv operations.
enum class GpuResourceType {
  kGpuAsyncStreamCollectivesP2P = ResourceTypeToIndex(
      ResourceType::kTargetDefinedResourceTypeBegin),  // Resource for P2P
                                                       // collectives, which
                                                       // will be issued on a
                                                       // separate stream.
  kGpuAsyncStreamSend0,        // A resource for P2P Send operation.
  kGpuAsyncStreamSend1,        // Another resource for P2P Send operation.
  kGpuAsyncStreamRecv0,        // A resource for P2P Recv operation.
  kGpuAsyncStreamRecv1,        // Another resource for P2P Recv operation.
  kGpuAsyncStreamCollectives,  // The resource for collective operations.
  kGpuAsyncStreamComputes,     // The resource for async compute operations.
  kGpuAsyncStreamMemcpy,       // The resource for host offloading operations.
  kGpuResourceTypeEnd,
};

constexpr int32_t kP2pResourceCount = 4;

// Base GPU async tracker that enables async tracking only for async collectives
// that are marked for async execution.
class GpuAsyncTrackerBase : public AsyncTracker {
 public:
  explicit GpuAsyncTrackerBase(
      const SchedulerConfig& config,
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp);

  // Returns if this is an Async op done that the scheduler supports.
  bool IsSupportedAsyncDone(const HloInstruction& hlo) const override;

  // Returns if this is an Async op start that the scheduler supports.
  bool IsSupportedAsyncStart(const HloInstruction& hlo) const override;

  // Post processing the scheduling graph.
  void PostProcessScheduleGraph(
      HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator) const override;
};

// GPU async tracker maps all collectives onto an async stream resource.
class GpuAsyncTracker : public GpuAsyncTrackerBase {
 public:
  explicit GpuAsyncTracker(const SchedulerConfig& config);

  // Returns resources used (occupied or released) by `instr`.
  ResourcesVector GetResourcesFromInstructionImpl(
      const HloInstruction& instr) const override;

  // Returns the number of target defined resources
  int64_t GetNumTargetDefinedResources() const override;

  // Returns how many instructions using the given resource_type we can overlap
  int64_t GetNumAvailableResources(int64_t resource_type) const override;

  // Returns the name of the given resource
  absl::string_view GetResourceName(int64_t resource_type) const override;

  // Returns the hazard type that describes how to resolve the conflicts when
  // multiple instructions attempt to use the given resource type concurrently.
  // Default resources have a hazard type of kUnshareable.
  ResourceHazardType GetResourceHazardType(
      int64_t resource_type) const override;

  // Returns the number of resources (of type resource_type) that are used by
  // this instruction.
  int64_t GetNumResourcesPerInstruction(
      int64_t resource_type, const HloInstruction& instr) const override;
};

// GPU approximate latency estimator. It is a set of hardcoded heuristics
// for every instruction and async instruction pairs.
class GpuLatencyEstimator : public ApproximateLatencyEstimator {
 public:
  explicit GpuLatencyEstimator(
      int64_t pointer_size,
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp);

  // Uses the approximate node for an instruction `instr`.
  TimeCost NodeCost(const HloInstruction* instr) const override;

  // Returns a latency estimation between nodes `from` and `to`.
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& to) const override;

 private:
  int64_t pointer_size_;
};

// GPU PGLE statistics tracker.
class GPUProfileStatisticsAggregator : public ProfileStatisticsAggregator {
 public:
  // Counts `instruction` as missing if is not a NOP.
  void HandleMissingInstructionCost(const HloInstruction& instruction) override;

  // Counts `instruction` as found.
  void HandleFoundInstructionCost(const HloInstruction& instruction) override;

  // Counts `from` -> `to` pair as missing if it is an async pair.
  void HandleMissingInstructionLatency(const HloInstruction& from,
                                       const HloInstruction& to) override;

  // Counts `from` -> `to` pair as found.
  void HandleFoundInstructionLatency(const HloInstruction& from,
                                     const HloInstruction& to) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_LATENCY_HIDING_SCHEDULER_H_
