/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LATENCY_HIDING_SCHEDULER_H_
#define XLA_SERVICE_LATENCY_HIDING_SCHEDULER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/map_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/xla.pb.h"

namespace xla {

inline constexpr int64_t kInvalidAnnotation = -1;

struct CanonicalAsyncOp {
  HloOpcode outer;  // kAsyncStart or kAsyncDone
  HloOpcode inner;  // kAllReduce, kAllGather, kAllToAll, kCollectiveBroadcast,
                    // kCollectivePermute, or kReduceScatter
};

CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo);

using GetCanonicalAsyncOpFunc =
    std::function<CanonicalAsyncOp(const HloInstruction& hlo)>;

class HloGraphNode;
class ModulePressureState;

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

class HloGraphNode;
class HloScheduleGraph;

struct SchedulerConfig {
  int64_t collective_broadcast_overlap_limit = 1;
  int64_t collective_permute_overlap_limit = 1;
  int64_t all_to_all_overlap_limit = 1;
  int64_t ragged_all_to_all_overlap_limit = 1;
  int64_t all_gather_overlap_limit = 1;
  int64_t all_reduce_overlap_limit = 1;
  int64_t reduce_scatter_overlap_limit = 1;
  int64_t send_recv_overlap_limit = 1;
  int64_t send_recv_host_overlap_limit = 1;
  int64_t copy_overlap_limit = 1;
  uint64_t memory_limit = UINT64_MAX;
  int64_t max_hops_to_closest_selective_overlap = 0;
  int64_t rerun = 0;
  int64_t parallel_collective_overlap_limit = 1;
  bool schedule_send_recvs = false;
  bool deannotate_group_if_blocked = false;
  // Consider send recv as the same resource. Some platforms do not take well
  // overlapping the send/recv ops between themselves.
  bool force_send_recv_to_use_same_resource = false;
  bool use_real_cost_model = false;
  bool aggressive_scheduling_policies = false;
  bool prioritize_async_depth_over_stall = false;
  bool enable_release_start_policy = false;
  bool resource_sharing = false;
  bool resource_serializing = false;
  bool depth_based_memory_pressure_reduction = false;
  bool enable_selective_resources = false;

  // Freely schedule nodes at the start of a scheduling group, allowing the
  // scheduler to delay them to promote better overlap.
  bool flexible_scheduling_annotation_scheduling = false;
  // If the above flag is also set, force the scheduler to provide maximum delay
  // to nodes at the stat of a scheduling group.
  bool aggressive_flexible_annotation_scheduling = false;
  // Prioritize  flexible annotation scheduling over memory pressure; this is
  // useful when the memory pressure is high. Without this, under high memory
  // pressure, aggressive_flexible_annotation_scheduling is not respected.
  bool force_delay_over_memory_pressure = false;
  // If true, estimate the fragmentation size of the module by running the heap
  // simulator.
  bool estimate_fragmentation_size = false;
  // If true, track the resource usage of sync ops in latency hiding scheduler.
  bool track_sync_op_resource_usage = false;
  // If true, use top down scheduling.
  bool top_down_scheduling = false;
};

// Class used estimate latency between instructions and cost of HLOs.
class LatencyEstimator {
 public:
  using TimeCost = double;
  // Uses the approximate or cost model function for GetLatencyBetween based on
  // a flag.
  virtual TimeCost GetLatencyBetween(const HloGraphNode& from,
                                     const HloGraphNode& target) const = 0;
  // Uses the approximate or cost model function for NodeCost based on a flag.
  virtual TimeCost NodeCost(const HloInstruction* node) const = 0;
  // Returns the core frequency used in latency estimation.
  virtual int CyclesPerMicrosecond() const = 0;

  // Returns the latency cycles of the instruction if it is specified in the
  // frontend attributes. Otherwise, returns std::nullopt.
  std::optional<TimeCost> GetLatencyFromMetadata(
      const HloInstruction& instruction) const;

  virtual ~LatencyEstimator() = default;

  inline CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) const {
    return get_canonical_async_op_(hlo);
  }
  bool IsAsyncPair(const HloGraphNode& from, const HloGraphNode& target) const;
  bool IsP2pPair(const HloGraphNode& from, const HloGraphNode& target) const;
  explicit LatencyEstimator(
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : get_canonical_async_op_(func) {}

 private:
  GetCanonicalAsyncOpFunc get_canonical_async_op_;
};

// Implementation of LatencyEstimator using an approximate cost model.
class ApproximateLatencyEstimator : public LatencyEstimator {
 public:
  explicit ApproximateLatencyEstimator(
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : LatencyEstimator(func) {}

  // Returns a latency estimation between two instructions.
  // Currently this is in abstract units. When the real/accurate cost model is
  // implemented this will be in cycles.
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  // Uses the approximate or cost model function for NodeCost based on a flag.
  TimeCost NodeCost(const HloInstruction* instr) const override;
  // ApproximateLatencyEstimator uses abstract units so this returns 1.
  int CyclesPerMicrosecond() const override { return 1; }

 public:
  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kMediumCost = 1000.0;
  static constexpr TimeCost kHighCost = 5000.0;

 protected:
  // These values are empirically derived to obtain an overlap of one output
  // fusion/convolution with 1 async op or 5 loop fusions with an async op.
  static constexpr TimeCost kLowLatency = 1.0;
  static constexpr TimeCost kHighLatency = 5000.0;
};

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

  inline CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) const {
    return get_canonical_async_op_(hlo);
  }

  // Updates target defined states after scheduling a node.
  virtual void UpdateTargetDefinedStates(
      const HloInstruction& hlo, const HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator,
      LatencyEstimator::TimeCost current_time) {}

  // Updates target defined states after scheduling a computation.
  virtual void UpdateTargetDefinedStates(HloComputation* computation) {}

  // Resets target defined states after scheduling a computation.
  virtual void ResetTargetDefinedStates() {}

  const SchedulerConfig& GetConfig() const { return config_; }

  // Clears the cache of per-computation resource maps. This is needed when,
  // e.g., we modify the schedule of a computation, which could change the
  // resource usage of the computation.
  void InvalidateCache() { async_in_computation_cache_.clear(); }

  // Similar to InvalidateCache(), but only invalidates the cache for the given
  // computation.
  void InvalidateCache(const HloComputation* computation) {
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

  mutable absl::flat_hash_map<
      const HloComputation*,
      std::unique_ptr<absl::flat_hash_map<int64_t, int64_t>>>
      async_in_computation_cache_;
  GetCanonicalAsyncOpFunc get_canonical_async_op_;

 protected:
  const SchedulerConfig config_;
  mutable absl::flat_hash_map<const HloInstruction*, ResourcesVector>
      resources_cache_;
};

// Base class for the core scheduling algorithm.
class SchedulerCore {
 public:
  // Abstract base class for scheduling state.
  struct SchedulingState {
    virtual ~SchedulingState() = default;
  };

  // Hook function to modify scheduling graph before scheduler runs.
  using GraphProcessingHook = std::function<absl::Status(HloScheduleGraph*)>;

  virtual absl::Status InitializeScheduler(const HloModule* module) = 0;

  virtual absl::Status CaptureScheduleProto() = 0;

  virtual absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() = 0;

  virtual absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
  MakeSchedulingState(const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) {
    return absl::UnimplementedError("Not implemented.");
  }
  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation,
      std::shared_ptr<SchedulingState> sched_state) {
    return absl::UnimplementedError("Not implemented.");
  }

  virtual ~SchedulerCore() = default;
  virtual int64_t GetMemoryPeak() = 0;
  virtual void SetMemoryLimit(uint64_t new_limit) = 0;
  virtual uint64_t GetMemoryLimit() = 0;
  virtual int64_t GetRerunTimes() = 0;

  // Set a graph processing hook that will run before scheduling a computation.
  // Heuristics can use this to set scheduling preferences to the scheduling
  // graph nodes.
  virtual absl::Status SetGraphProcessingHook(const GraphProcessingHook& hook) {
    return absl::UnimplementedError("Unimplemented. ");
  }
};

class SchedulingContext {
  // Scheduling context for a single pass.  This object runs HloAliasAnalysis on
  // first use and should not be used across passes as the module may have
  // changed.
 public:
  SchedulingContext(const HloModule* module,
                    std::shared_ptr<const LatencyEstimator> latency_estimator,
                    std::shared_ptr<AsyncTracker> async_tracker,
                    const AliasInfo* alias_info,
                    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes =
                        HloCostAnalysis::DefaultShapeSize)
      : latency_estimator_(std::move(latency_estimator)),
        async_tracker_(std::move(async_tracker)),
        alias_info_(alias_info),
        shape_size_bytes_(shape_size_bytes),
        module_(module) {}

  std::shared_ptr<const HloAliasAnalysis> GetAliasAnalysis() const {
    // Lazy initialization of alias analysis on first use.
    if (alias_analysis_ == nullptr) {
      alias_analysis_ = HloAliasAnalysis::Run(module_, alias_info_).value();
    }
    return alias_analysis_;
  }

  std::shared_ptr<const LatencyEstimator> GetLatencyEstimator() const {
    return latency_estimator_;
  }
  std::shared_ptr<AsyncTracker> GetAsyncTracker() const {
    return async_tracker_;
  }

  const HloCostAnalysis::ShapeSizeFunction& GetShapeSizeBytes() const {
    return shape_size_bytes_;
  }

  const AliasInfo* GetAliasInfo() const { return alias_info_; }

 private:
  mutable std::shared_ptr<const HloAliasAnalysis> alias_analysis_;
  std::shared_ptr<const LatencyEstimator> latency_estimator_;
  std::shared_ptr<AsyncTracker> async_tracker_;
  const AliasInfo* alias_info_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  const HloModule* module_;
};

// Tracks user annotations for scheduling.
class AnnotationTracker {
 public:
  explicit AnnotationTracker(const HloModule* module) : module_(module) {
    for (const HloComputation* comp : module_->MakeNonfusionComputations()) {
      absl::flat_hash_set<int64_t> annotations;
      for (const HloInstruction* instr : comp->instructions()) {
        if (auto annotation = GetAnnotation(instr)) {
          if (annotations.insert(annotation.value()).second) {
            comp_annotation_map_[comp].push_back(annotation.value());
          }
          annotations_[annotation.value()][comp].push_back(instr);
        }
      }
    }
  }
  bool HasAnnotations(const HloComputation* comp) const {
    return ContainsKey(comp_annotation_map_, comp);
  }
  std::vector<int64_t> GetAnnotations(const HloComputation* comp) const {
    return comp_annotation_map_.at(comp);
  }
  std::optional<int64_t> GetAnnotation(const HloInstruction* instr) const {
    const auto& attrs = instr->frontend_attributes().map();
    if (attrs.contains(kXlaSchedulingGroupIdAttr)) {
      return std::stoi(attrs.at(kXlaSchedulingGroupIdAttr));
    }
    return std::nullopt;
  }
  std::vector<const HloInstruction*> GetInstructions(
      const HloComputation* comp, const int64_t annotation) const {
    if (annotation == kInvalidAnnotation) {
      return {};
    }
    return annotations_.at(annotation).at(comp);
  }
  int64_t GetNumInstructions(const HloComputation* comp,
                             const int64_t annotation) {
    return annotations_[annotation][comp].size();
  }
  void PrintAnnotationSets(int64_t level) const {
    for (const auto& [annotation, comp_instr_vector] : annotations_) {
      for (const auto& [comp, instrs] : comp_instr_vector) {
        VLOG(level) << "Annotation " << annotation << " has " << instrs.size()
                    << " instructions in computation " << comp->name();
        for (const HloInstruction* instr : instrs) {
          VLOG(level) << "  " << instr->name();
        }
      }
    }
  }

 private:
  const HloModule* module_;
  absl::flat_hash_map<const HloComputation*, std::vector<int64_t>>
      comp_annotation_map_;
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<const HloComputation*,
                                          std::vector<const HloInstruction*>>>
      annotations_;
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<const HloComputation*,
                                          std::vector<const HloInstruction*>>>
      annotation_successors_;
};

// Represents an edge between two nodes in the schedule graph.
class HloEdge {
 public:
  // Constructor used for array resizing
  HloEdge() : latency_(0), original_latency_(0), target_(nullptr) {}

  // Nullptr is not a valid value for 'target'.
  HloEdge(LatencyEstimator::TimeCost latency, HloGraphNode* target)
      : latency_(latency), original_latency_(latency), target_(target) {}
  LatencyEstimator::TimeCost Latency() const { return latency_; }
  LatencyEstimator::TimeCost OriginalLatency() const {
    return original_latency_;
  }
  void SetLatency(LatencyEstimator::TimeCost latency) { latency_ = latency; }
  void SetOriginalLatency(LatencyEstimator::TimeCost original_latency) {
    original_latency_ = original_latency;
  }
  const HloGraphNode& Target() const { return *target_; }
  HloGraphNode& Target() { return *target_; }
  HloGraphNode* TargetPtr() const { return target_; }
  std::string ToString() const;

  // Returns true iff SetSharableResources has been called
  bool SharableResourcesComputed() const {
    return sharable_resources_index_ >= 0;
  }
  // Returns the sharable resources vector for this edge
  // REQUIRES: SetSharableResources has been called
  const std::vector<int64_t>& GetSharableResources(
      const HloScheduleGraph* g) const;
  // Sets the sharable resources vector for this edge to "vals".  Subsequent
  // calls to GetSharableResources for this edge will return a vector
  // whose contents are identical to "vals".
  void SetSharableResources(HloScheduleGraph* g,
                            const std::vector<int64_t>& vals);

 private:
  // Latency between the two nodes connected by this edge. The other end of the
  // edge is the owner of the HloEdge object. This latency can get updated due
  // to various scheduling optimizations.
  LatencyEstimator::TimeCost latency_;
  // Original latency is the initial latency value (typically computed by a
  // latency estimator).
  LatencyEstimator::TimeCost original_latency_;
  // Target node of this edge.
  HloGraphNode* target_;

  // If -1, not initialized yet.  If >= 0, then this is an index
  // into a vector in the enclosing HloScheduleGraph.  0 is a special
  // value that points to an empty vector shared across all edges that
  // have an empty sharable_resources_ set (most edges).
  // > 0 points to an entry containing an edge-specific vector in the
  // HloScheduleGraph holding the sharable_resources_ for this edge.
  int sharable_resources_index_ = -1;
};

// Node in the schedule graph, plus information used for scheduling.
class HloGraphNode {
 public:
  using TimeCost = LatencyEstimator::TimeCost;

  // Constructor used temporarily for initializing a vector of HloGraphNodes
  HloGraphNode()
      : instr_(nullptr), opcode_(HloOpcode::kAdd), original_position_(0) {
    InitBitFields();
  }

  // Nullptr is not a valid value for 'i'.
  explicit HloGraphNode(const HloInstruction* i, int64_t original_position)
      : instr_(i), opcode_(i->opcode()), original_position_(original_position) {
    InitBitFields();
  }

  static void UpdateOrAddDependency(HloGraphNode* from, HloGraphNode* to,
                                    LatencyEstimator::TimeCost latency) {
    auto update_latency_if_edge_exists =
        [&](absl::Span<HloEdge> edges, HloGraphNode* to,
            LatencyEstimator::TimeCost latency) {
          auto it = absl::c_find_if(
              edges, [&](HloEdge edge) { return &edge.Target() == to; });
          if (it != edges.end()) {
            it->SetLatency(latency);
            return true;
          }
          return false;
        };
    if (!update_latency_if_edge_exists(from->GetSuccessors(), to, latency)) {
      from->AddSuccessor(HloEdge(latency, to));
      from->outdegree_++;
    }
    if (!update_latency_if_edge_exists(to->GetPredecessors(), from, latency)) {
      to->AddPredecessor(HloEdge(latency, from));
      to->indegree_++;
    }
  }

  static void UpdateOrAddDependency(HloGraphNode* from, HloGraphNode* to,
                                    const LatencyEstimator* latency_estimator) {
    UpdateOrAddDependency(from, to,
                          latency_estimator->GetLatencyBetween(*from, *to));
  }

  static void AddDependency(HloGraphNode* from, HloGraphNode* to,
                            LatencyEstimator::TimeCost latency) {
    to->AddPredecessor(HloEdge(latency, from));
    to->indegree_++;
    from->AddSuccessor(HloEdge(latency, to));
    from->outdegree_++;
  }

  static void AddDependency(HloGraphNode* from, HloGraphNode* to,
                            const LatencyEstimator* latency_estimator) {
    AddDependency(from, to, latency_estimator->GetLatencyBetween(*from, *to));
  }
  // Reset the node to a state where it's ready to be scheduled again.
  void ResetScheduling() {
    scheduled_ = false;
    indegree_ = predecessors_.size();
    outdegree_ = successors_.size();
    ready_time_ = std::numeric_limits<TimeCost>::max();
  }
  size_t GetReadyNodesIfScheduled() const { return ready_nodes_if_scheduled_; }
  void UpdateReadyNodesIfScheduled() {
    ready_nodes_if_scheduled_ = 0;
    for (auto& pred : GetPredecessors()) {
      if (pred.Target().GetOutdegree() == 1) {
        ++ready_nodes_if_scheduled_;
      }
    }
  }
  const HloInstruction& GetInstr() const { return *instr_; }
  HloOpcode GetOpcode() const { return opcode_; }
  bool IsScheduled() const { return scheduled_; }
  int32_t GetIndegree() const { return indegree_; }
  int32_t GetOutdegree() const { return outdegree_; }
  TimeCost GetReadyTime() const { return ready_time_; }
  void SetIndegree(int32_t indeg) { indegree_ = indeg; }
  void SetOutdegree(int32_t outdeg) {
    outdegree_ = outdeg;
    if (outdeg == 1) {
      for (HloEdge& succ : GetSuccessors()) {
        succ.Target().UpdateReadyNodesIfScheduled();
      }
    }
  }
  void SetScheduled() { scheduled_ = true; }
  void SetReadyTime(TimeCost ready_time) { ready_time_ = ready_time; }
  TimeCost GetCost() const { return cost_; }
  void SetCost(TimeCost cost) { cost_ = cost; }
  TimeCost GetAsyncDepth() const { return async_depth_; }
  TimeCost GetDepth() const { return depth_; }
  TimeCost GetGraphDepth() const { return graph_depth_; }
  void SetAsyncDepth(TimeCost async_depth) { async_depth_ = async_depth; }
  bool IsSupportedAsyncDone() const { return is_supported_async_done_; }
  bool IsSupportedAsyncStart() const { return is_supported_async_start_; }
  void SetDepth(TimeCost depth) { depth_ = depth; }
  void SetGraphDepth(TimeCost graph_depth) { graph_depth_ = graph_depth; }
  bool GetForceDelay() const { return force_delay_; }
  void SetForceDelay(bool force_delay) { force_delay_ = force_delay; }
  int GetForceDelayPriority() const { return force_delay_priority_; }
  void SetForceDelayPriority(int force_delay_priority) {
    force_delay_priority_ = force_delay_priority;
  }
  bool GetForceEarly() const { return force_early_; }
  void SetForceEarly(bool force_early) { force_early_ = force_early; }
  bool GetForceDelayAfterTarget() const { return force_delay_after_target_; }
  void SetForceDelayAfterTarget(bool force_delay_after_target) {
    force_delay_after_target_ = force_delay_after_target;
  }
  bool GetValuableForSelectiveOverlap() const {
    return valuable_for_selective_overlap_;
  }
  void SetValuableForSelectiveOverlap(bool valuable_for_selective_overlap) {
    valuable_for_selective_overlap_ = valuable_for_selective_overlap;
  }
  bool ReleasesSelectiveResource() const {
    return releases_selective_resource_;
  }
  bool OccupiesSelectiveResource() const {
    return occupies_selective_resource_;
  }
  void SetReleasesSelectiveResource(bool releases_selective_resource) {
    releases_selective_resource_ = releases_selective_resource;
  }
  void SetOccupiesSelectiveResource(bool occupies_selective_resource) {
    occupies_selective_resource_ = occupies_selective_resource;
  }
  int64_t GetNumHopsToClosestSelectiveResourceOccupier() const {
    return num_hops_to_closest_selective_resource_occupier_;
  }
  void SetNumHopsToClosestSelectiveResourceOccupier(
      int64_t num_hops_to_closest_selective_resource_occupier) {
    num_hops_to_closest_selective_resource_occupier_ =
        num_hops_to_closest_selective_resource_occupier;
  }
  void SetPreference(double preference) { preference_ = preference; }
  double GetPreference() const { return preference_; }
  const ResourcesVector& GetResources() const { return rare_->resources; }
  bool DoesOccupyAnyResource() const { return does_occupy_any_resource_; }
  bool DoesReleaseAnyResource() const { return does_release_any_resource_; }
  bool DoesOccupyShareableResource(int64_t resource) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_linear_search(rare_->occupied_shareable_resources, resource);
  }
  bool DoesReleaseResource(int64_t res) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_any_of(
        rare_->resources, [res](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceRelease &&
                 resource.first == res;
        });
  }
  bool DoesReleaseResource(ResourceType res) const {
    return DoesReleaseResource(ResourceTypeToIndex(res));
  }
  bool DoesOccupyResource(int64_t res) const {
    if (!has_rare_) {
      return false;
    }
    return absl::c_any_of(
        rare_->resources, [res](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceOccupy &&
                 resource.first == res;
        });
  }
  bool DoesOccupyResource(ResourceType res) const {
    return DoesOccupyResource(ResourceTypeToIndex(res));
  }
  // Returns the net resources used by the node. For a while loop, it computes
  // the net resources used by the instructions in the while body. Otherwise, it
  // returns the readily-available resources vector.
  ResourcesVector GetNetResources() const {
    if (GetOpcode() != HloOpcode::kWhile) {
      return rare_->resources;
    }
    ResourcesVector result;
    for (const auto& [resource, usage] : rare_->resources) {
      if (usage == ResourceUsageType::kResourceOccupy &&
          !DoesReleaseResource(resource)) {
        result.push_back(std::make_pair(resource, usage));
      }
      if (usage == ResourceUsageType::kResourceRelease &&
          !DoesOccupyResource(resource)) {
        result.push_back(std::make_pair(resource, usage));
      }
    }
    return result;
  }
  std::optional<ResourceUsageType> UsesResourceType(ResourceType res) const {
    if (has_rare_) {
      int64_t res_type = ResourceTypeToIndex(res);
      for (const auto& [resource_type, usage_type] : rare_->resources) {
        if (resource_type == res_type) {
          return usage_type;
        }
      }
    }
    return std::nullopt;
  }
  std::optional<ResourceUsageType> UsesResourceType(int64_t res) const {
    if (has_rare_) {
      for (const auto& [resource_type, usage_type] : rare_->resources) {
        if (resource_type == res) {
          return usage_type;
        }
      }
    }
    return std::nullopt;
  }
  const std::vector<int64_t>& GetShareableResourcesOnEdge(HloScheduleGraph* g,
                                                          HloEdge& edge) const {
    if (!edge.SharableResourcesComputed()) {
      HloGraphNode& to = edge.Target();
      std::vector<int64_t> resources;
      absl::c_for_each(rare_->released_shareable_resources,
                       [this, &to, &resources](const int64_t resource) {
                         if (to.DoesOccupyShareableResource(resource) &&
                             this->DoesReleaseResource(resource)) {
                           resources.push_back(resource);
                         }
                       });
      edge.SetSharableResources(g, resources);
    }
    return edge.GetSharableResources(g);
  }
  bool HasReleasedNonExtendableResources() const {
    return has_rare_ && !rare_->released_non_extendable_resources.empty();
  }
  const absl::InlinedVector<int64_t, 1>& GetReleasedNonExtendableResources()
      const {
    return rare_->released_non_extendable_resources;
  }

  absl::Span<HloEdge> GetPredecessors() { return predecessors_.GetSpan(); }
  absl::Span<const HloEdge> GetPredecessors() const {
    return predecessors_.GetConstSpan();
  }
  void AddPredecessor(const HloEdge& e) { predecessors_.AddEdge(e); }
  absl::Span<HloEdge> GetSuccessors() { return successors_.GetSpan(); }
  absl::Span<const HloEdge> GetSuccessors() const {
    return successors_.GetConstSpan();
  }
  void AddSuccessor(const HloEdge& e) { successors_.AddEdge(e); }
  int64_t GetOriginalPosition() const { return original_position_; }
  int64_t GetAnnotation() const { return annotation_; }
  absl::Status SetAnnotation(int64_t annotation) {
    TF_RET_CHECK(annotation_ == kInvalidAnnotation)
        << "Instruction " << instr_->name()
        << " has an existing annotation: " << annotation_;
    annotation_ = annotation;
    return absl::OkStatus();
  }
  void ClearAnnotation() { annotation_ = -1; }
  std::string ToString(const AsyncTracker* async_tracker = nullptr) const {
    std::string result;
    absl::StrAppend(&result, "Instr: ", instr_->ToShortString(), "\n");
    absl::StrAppend(&result, "ReadyTime: ", ready_time_, "\n");
    absl::StrAppend(&result, "Indegree: ", indegree_, "\n");
    absl::StrAppend(&result, "Outdegree: ", outdegree_, "\n");
    absl::StrAppend(&result, "Cost: ", cost_, "\n");
    absl::StrAppend(&result, "Async Depth: ", async_depth_, "\n");
    absl::StrAppend(&result, "Depth: ", depth_, "\n");
    absl::StrAppend(&result, "Graph Depth: ", graph_depth_, "\n");
    absl::StrAppend(&result, "Force Delay: ", force_delay_, "\n");
    absl::StrAppend(&result, "Force Early: ", force_early_, "\n");
    absl::StrAppend(
        &result, "Force Delay After Target: ", force_delay_after_target_, "\n");
    absl::StrAppend(&result, "Predecessors:\n");
    for (const HloEdge& e : GetPredecessors()) {
      absl::StrAppend(&result, e.ToString());
    }
    absl::StrAppend(&result, "Successors:\n");
    for (const HloEdge& e : GetSuccessors()) {
      absl::StrAppend(&result, e.ToString());
    }
    if (async_tracker != nullptr) {
      absl::StrAppend(&result, "Resources:\n");
      for (const auto& [resource, usage] : rare_->resources) {
        absl::StrAppend(
            &result, "\tResource: ", async_tracker->GetResourceName(resource),
            " usage: ", async_tracker->GetResourceUsageName(usage), "\n");
      }
    }
    return result;
  }
  bool HasRecursiveResources() const { return has_recursive_resources_; }
  const absl::flat_hash_map<int64_t, int64_t>& GetRecursiveResources() const {
    return rare_->recursive_resources;
  }
  bool HasOperandThatIsSupportedAsyncDone() const {
    return has_operand_that_is_supported_async_done_;
  }

  bool HasUserThatIsSupportedAsyncStart() const {
    return has_user_that_is_supported_async_start_;
  }

 private:
  friend class HloScheduleGraph;

  // Older c++ versions don't allow initializers for bitfields, so we initialize
  // these in this routine, invoked by the constructor
  void InitBitFields() {
    does_occupy_any_resource_ = false;
    does_release_any_resource_ = false;
    is_supported_async_done_ = false;
    is_supported_async_start_ = false;
    has_operand_that_is_supported_async_done_ = false;
    has_user_that_is_supported_async_start_ = false;
    scheduled_ = false;
    valuable_for_selective_overlap_ = true;
    releases_selective_resource_ = false;
    occupies_selective_resource_ = false;
    has_recursive_resources_ = false;
  }

  // Some of the fields in this are rarely non-empty (in one large compilation,
  // 98% of nodes had all the fields empty).  To make each HloGraphNode more
  // compact, we store the state for these fields in a vector of Rare objects in
  // the parent HloScheduleGraph.  All instructions that have all these fields
  // empty point to the same canonical entry with all empty fields.  Other
  // nodes point to their own storage allocated in a vector of Rare objects
  // in the HloScheduleGraph object.
  struct Rare {
    // Non-extendable resources released by this node.
    absl::InlinedVector<int64_t, 1> released_non_extendable_resources;
    // Shareable resources released by this node.
    absl::InlinedVector<int64_t, 1> released_shareable_resources;
    // Shareable resources occupied by this node.
    absl::InlinedVector<int64_t, 1> occupied_shareable_resources;
    // Recursive resources used by the node.
    absl::flat_hash_map<int64_t, int64_t> recursive_resources;
    // AsyncResources used by the node.
    ResourcesVector resources;
  };

  // Instruction this Graph node represents
  const HloInstruction* instr_;
  // Opcode of instr_, copied here for better cache behavior (so we can look at
  // the opcode without having to touch another cache line).
  HloOpcode opcode_;

  // Some of the booleans are looked at very often, so we avoid making them
  // bitfields
  // Force the scheduling of the nodes with attribute set as late as possible.
  bool force_delay_ = false;
  // If multiple nodes are there with force_delay_ = true, the one with the
  // lowest delay priority will be scheduled first.
  int force_delay_priority_ = 0;
  // Force the scheduling of the nodes with attribute set as early as possible.
  bool force_early_ = false;
  // If has_rare_ is false, then all the fields in rare can assumed to be
  // empty/default values
  bool has_rare_ = false;
  // Force the scheduling of the nodes with attribute as late as possible,
  // but do it after evaluating the early target scheduling rule.
  bool force_delay_after_target_ = false;

  // Preference value used for scheduling heuristics,
  // a graph node having a higher preference value means it's scheduled
  // earlier. See ReadySetLt::operator()
  float preference_ = 0.0;

  // Other boolean fields are less performance sensitive so can be stored in
  // bitfields.  These are initialized to default values in InitBitFields() (due
  // to older c++ versions not supporting initializers for bit fields).

  // Does the node occupy any resource.
  bool does_occupy_any_resource_ : 1;
  // Does the node release any resource.
  bool does_release_any_resource_ : 1;
  // Is the node a supported async done.
  bool is_supported_async_done_ : 1;
  // Is the node a supported async start.
  bool is_supported_async_start_ : 1;
  // Whether the instruction has an operand which is a supported async done.
  bool has_operand_that_is_supported_async_done_ : 1;
  // Whether the instruction has a user which is a supported async start.
  bool has_user_that_is_supported_async_start_ : 1;
  // Whether this node has been scheduled or not yet.
  bool scheduled_ : 1;
  // Whether this node can be overlapped with (can cover the latency/cost of)
  // edges occupying selective resources.
  bool valuable_for_selective_overlap_ : 1;
  // Whether this node releases a selective resource.
  bool releases_selective_resource_ : 1;
  // Whether this node occupies a selective resource.
  bool occupies_selective_resource_ : 1;
  // Whether recursive_resources_.size() > 0
  bool has_recursive_resources_ : 1;
  // The position of this node in the original order.
  int32_t original_position_;
  // Pointer to the HloGraphNode::Rare entry for this node in the parent object
  // (Actual storage is managed by rare_storage_ in parent object)
  Rare* rare_ = nullptr;
  // Estimated time at which this node is gonna be ready to be scheduled.
  // The node should be added to the ready to be scheduled set when ready_time_
  // is less or equal to the current time in the schedule.
  TimeCost ready_time_ = std::numeric_limits<TimeCost>::max();
  // Time cost of the execution of the operation of this nodes represent.
  TimeCost cost_ = 0.0;
  // Depth in latency terms of a node based on Async operation cost on the path.
  TimeCost async_depth_ = 0.0;
  // Depth in latency terms of node based on operation cost on the path to the
  // entry node.
  TimeCost depth_ = 0.0;
  int64_t annotation_ = kInvalidAnnotation;

  // Depth in latency terms of node based on distance to the entry node.
  int64_t graph_depth_ = 0;
  // Nums hops to closest selective resource occupier.
  int32_t num_hops_to_closest_selective_resource_occupier_ =
      std::numeric_limits<int32_t>::max();
  // Number of ready nodes if this node is scheduled.
  int32_t ready_nodes_if_scheduled_ = 0;
  // Number of predecessor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t indegree_ = 0;
  // Number of successor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t outdegree_ = 0;

  // EdgeStorage is used to manage incoming and outgoing edge arrays.
  // The storage is hybrid where most backing store for HloEdge objects
  // is in two vectors allocated in the HloScheduleGraph, and EdgeStorage
  // points into a span within those arrays.  However, sometimes in
  // the graph construction process, it is hard to know the total number
  // of edges needed for certain nodes in rare circumstances.  For handling
  // those cases when the allocated span in the shared arrays is not
  // sufficient, the storage is allocated and managed by EdgeStorage itself.
  class EdgeStorage {
   public:
    enum kOwnership { kNotOwned };
    EdgeStorage() : alloc_(0), owned_(false) {}

    ~EdgeStorage() {
      if (owned_) {
        delete[] edges_;
      }
    }
    // Set the edge storage to point to separately managed space for
    // an array of "alloc" edges pointed to by "ptr".  This memory must
    // remain live for as long as the edges are being used.
    void SetEmptyPointingToSharedSpace(HloEdge* ptr, int alloc) {
      edges_ = ptr;
      size_ = 0;
      alloc_ = alloc;
      owned_ = false;
    }

    int size() const { return size_; }
    absl::Span<HloEdge> GetSpan() { return absl::MakeSpan(edges_, size_); }
    absl::Span<const HloEdge> GetConstSpan() const {
      return absl::MakeConstSpan(edges_, size_);
    }

    void AddEdge(const HloEdge& e) {
      if (size_ >= alloc_) {
        // Grow: Make sure we always leave room for at least
        // one new edge since we're adding one right now, and do
        // doubling to avoid too much copying overhead
        int new_size = std::max(1, alloc_ * 2);
        HloEdge* new_edges = new HloEdge[new_size];
        for (int i = 0; i < size_; i++) {
          new_edges[i] = edges_[i];
        }
        if (owned_) {
          delete[] edges_;
        }
        alloc_ = new_size;
        edges_ = new_edges;
        owned_ = true;
      }
      DCHECK_LT(size_, alloc_);
      edges_[size_] = e;
      size_++;
    }

   private:
    // Points to edges.  Usually points into
    // HloScheduleGraph::predecessors_storage_ (for predecessors) or
    // HloScheduleGraph::successors_storage_ (if !owned).  If owned, then this
    // HloGraphNode object is responsible for the memory pointed to by edges
    HloEdge* edges_ = nullptr;
    // Number of valid edges
    int size_ = 0;
    // Allocated space in edges_ array
    int32_t alloc_ : 31;
    bool owned_ : 1;
  };
  EdgeStorage predecessors_;
  EdgeStorage successors_;
};

// Schedule graph that can be used to drive scheduling
// of HLO instructions.
class HloScheduleGraph {
 public:
  // Instructions in the list passed to the constructor shouldn't be
  // altered/deleted during the existence of the HloScheduleGraph.
  // Nullptr is not a valid value for 'post_order_instructions' and
  // 'alias_analysis'.
  HloScheduleGraph(const std::vector<HloInstruction*>* post_order_instructions,
                   std::shared_ptr<const SchedulingContext> scheduling_context);
  void PrintSizes() const {
    LOG(INFO) << "HloScheduleGraph sizes: node_storage: "
              << node_storage_.size()
              << " preds: " << predecessors_storage_.size()
              << " succs: " << successors_storage_.size()
              << " rare: " << rare_storage_.size()
              << " sharable_resources: " << sharable_resources_storage_.size()
              << " nodes: " << nodes_.size()
              << " original_order: " << original_order_.size();
  }

  std::string ToString() const;

  HloGraphNode& GetNode(const HloInstruction* instr) const;
  HloGraphNode* GetNodePtr(const HloInstruction* instr) const;

  std::vector<HloGraphNode*> FindBottomRoots() const;

  std::vector<HloGraphNode*> FindTopRoots() const;

  void InitializeGraphAnalysis();

  void AnnotateGraph(const AnnotationTracker* annotation_tracker);

  // List of instructions in the original scheduled order. (Before scheduling).
  absl::Span<const HloInstruction* const> GetOriginalInstrList() const {
    return absl::MakeConstSpan(original_order_);
  }
  // Returns what was the original instruction position in the original order.
  int64_t OriginalInstructionPosition(const HloInstruction* instr) const {
    auto it = nodes_.find(instr);
    CHECK(it != nodes_.end());
    return it->second;
  }

  // Return node preference values in original order.
  std::vector<double> GetPreferences() {
    std::vector<double> preferences;
    preferences.reserve(original_order_.size());
    for (const HloInstruction* instr : original_order_) {
      preferences.push_back(GetNodePtr(instr)->GetPreference());
    }
    return preferences;
  }

  // Set node preference values in original order.
  void SetPreferences(const std::vector<double>& preferences) {
    CHECK_EQ(preferences.size(), original_order_.size());
    for (int i = 0; i < original_order_.size(); ++i) {
      GetNodePtr(original_order_[i])->SetPreference(preferences[i]);
    }
  }
  void ResetScheduling() {
    for (auto& pair : nodes_) {
      node_storage_[pair.second].ResetScheduling();
    }
    for (auto& pair : nodes_) {
      node_storage_[pair.second].UpdateReadyNodesIfScheduled();
    }
  }

 private:
  friend class HloEdge;

  // Backing store for the nodes in the graph
  mutable std::vector<HloGraphNode> node_storage_;
  std::vector<HloEdge> predecessors_storage_;
  std::vector<HloEdge> successors_storage_;

  // Rare storage for HloGraphNode objects
  std::vector<std::unique_ptr<HloGraphNode::Rare>> rare_storage_;

  // Storage for HloEdge::sharable_resources_ vectors that are non-empty
  std::vector<std::vector<int64_t>> sharable_resources_storage_;

  // Map from instruction to the index in node_storage_ that holds the node
  absl::flat_hash_map<const HloInstruction*, int> nodes_;
  // List containing the original order (before scheduling) of the
  // instructions).
  std::vector<const HloInstruction*> original_order_;
  // Searches through node's predecessors to see if
  // possible_predecessor can be found.
  bool IsPredecessorTransitively(const HloGraphNode* node,
                                 const HloGraphNode* possible_predecessor);
  // Scheduling context for the graph.
  std::shared_ptr<const SchedulingContext> scheduling_context_;
};

// These HloEdge routines need to be defined after HloScheduleGraph, since
// they peek inside its representation for the shared storage.

inline const std::vector<int64_t>& HloEdge::GetSharableResources(
    const HloScheduleGraph* g) const {
  return g->sharable_resources_storage_[sharable_resources_index_];
}

// Sets the sharable resources vector for this edge to "vals".  Subsequent
// calls to GetSharableResources for this edge will return a vector
// whose contents are identical to "vals".
inline void HloEdge::SetSharableResources(HloScheduleGraph* g,
                                          const std::vector<int64_t>& vals) {
  CHECK_LT(sharable_resources_index_, 0);
  if (vals.empty()) {
    // Share the 0th entry, which is an empty vector
    CHECK(g->sharable_resources_storage_[0].empty());
    sharable_resources_index_ = 0;
  } else {
    // Non-empty: add the vector into g->sharable_resources_storage_ and
    // save the index where it is stored in sharable_resources_index_.
    sharable_resources_index_ = g->sharable_resources_storage_.size();
    g->sharable_resources_storage_.push_back(vals);
  }
}

// Tracks data about HloBuffers like where the first definition is in the
// original schedule and caches the buffer size (as Target::ShapeSize()) is
// expensive.
class BufferInfoTracker {
 public:
  struct ValueInfo {
    const HloBuffer* value = nullptr;
    const HloInstruction* first_definition = nullptr;
    const HloInstruction* last_use = nullptr;
    int64_t buffer_size = 0;

    // Precomputed value of
    //     value->values()[0]->shape().has_layout() &&
    //     (value->values()[0]->shape().layout().memory_space() !=
    //      kDefaultMemorySpace)
    // This expression is invoked repeatedly and is responsible for many cache
    // misses.
    bool non_default_memory_space_layout = false;
  };
  BufferInfoTracker(const HloModule* module,
                    const HloAliasAnalysis* alias_analysis,
                    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes);
  static ValueInfo CreateBufferInfo(
      const HloBuffer* value, const HloInstruction* first_definition,
      const HloInstruction* last_use,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
    const auto& shape = value->values()[0]->shape();
    const bool non_default_memory_space_layout =
        (shape.has_layout() &&
         (shape.layout().memory_space() != Layout::kDefaultMemorySpace));
    return ValueInfo{
        /*value=*/value,
        /*first_definition=*/first_definition,
        /*last_use=*/last_use,
        /*buffer_size=*/shape_size_bytes(shape),
        /*non_default_memory_space_layout=*/non_default_memory_space_layout,
    };
  }
  const ValueInfo& GetBufferInfo(HloBuffer::Id id) const {
    return buffer_infos_[id];
  }

 private:
  std::vector<ValueInfo> buffer_infos_;
};

// Used to track and maintain memory pressure during scheduling.
class MemoryPressureTracker {
 public:
  using LiveBufferSet = absl::flat_hash_set<HloBuffer::Id>;
  struct MemoryPressureState {
    int64_t memory_peak = 0;
    absl::flat_hash_set<HloBuffer::Id> live_ids_at_bottom;
  };
  MemoryPressureTracker(
      const HloAliasAnalysis* hlo_alias_analysis,
      const BufferInfoTracker& buffer_tracker,
      const absl::flat_hash_map<const HloComputation*, MemoryPressureState>&
          pressure_state_cache,
      bool top_down_scheduling = false)
      : hlo_alias_analysis_(hlo_alias_analysis),
        live_buffers_(hlo_alias_analysis->buffers().back().id() + 1),
        buffer_tracker_(buffer_tracker),
        pressure_state_cache_(pressure_state_cache),
        live_memory_usage_(0),
        initial_memory_pressure_(0),
        top_down_scheduling_(top_down_scheduling) {}
  // Initialize object to be ready to start tracking of computation.
  void Initialize(const HloComputation* computation,
                  const LiveBufferSet& initial_live_buffers);
  // Reset the memory pressure tracker to the initialized state.
  void Reset(const HloComputation* computation,
             const LiveBufferSet& initial_live_buffers);
  // After an instruction is scheduled, update the memory pressure effect on
  // other instructions.
  void UpdateBuffers(const HloInstruction* instruction);
  // Return the memory pressure difference estimation if this instruction was
  // scheduled.
  // Returns a pair of (increase, peak) values.
  // "increase" determines by how much the memory pressure increases or
  // decreases after this instruction is scheduled. "peak" determines what's the
  // peak usage of memory of the computation. The peak can be higher than the
  // total memory increase of the instruction (imagine a computation called by a
  // while loop, the body of the while could use quite some more memory than the
  // amount of memory at the interfaces of the while loop instruction).
  std::pair<int64_t, int64_t> MemoryPressureDifference(
      const HloInstruction* instruction) const;
  absl::flat_hash_set<HloBuffer::Id> live_buffers() const {
    return live_buffers_set_;
  }
  bool BufferIsLive(const HloValue* buffer) const {
    CHECK_LT(buffer->id(), live_buffers_.size());
    return live_buffers_[buffer->id()];
  }
  // Returns the actual memory usage at the current state. It is initial memory
  // + current memory usage inside of the computation.
  int64_t memory_usage() const {
    return live_memory_usage_ + initial_memory_pressure_;
  }
  // Returns the initial memory pressure at the bottom of the computation.
  int64_t initial_memory_pressure() const { return initial_memory_pressure_; }

  // Returns pressure state object for this MemoryPressureTracker object.
  const MemoryPressureState& pressure_state() const { return pressure_state_; }

  absl::Span<const HloBuffer::Id> allocated_buffer_ids(
      const HloInstruction* i) const {
    auto it = instruction_ids_.find(i);
    CHECK(it != instruction_ids_.end());
    NodeAllocReleaseSpan s = alloc_release_spans_[it->second];
    return absl::MakeSpan(alloc_release_ids_).subspan(s.start, s.num_alloc);
  }

  absl::Span<const HloBuffer::Id> released_buffer_ids(
      const HloInstruction* i) const {
    auto it = instruction_ids_.find(i);
    CHECK(it != instruction_ids_.end());
    NodeAllocReleaseSpan s = alloc_release_spans_[it->second];
    return absl::MakeSpan(alloc_release_ids_)
        .subspan(s.start + s.num_alloc, s.num_release);
  }

 private:
  // Append to *dst the list of buffer ids allocated by instruction whose
  // memory usage should be tracked. Returns number of ids added.
  int32_t ComputeBufferAllocations(const HloInstruction* instruction,
                                   std::vector<HloBuffer::Id>* dst);

  // Append to *dst the list of buffer ids released by instruction whose
  // memory usage should be tracked. Returns number of ids added.
  int32_t ComputeBufferReleases(const HloInstruction* instruction,
                                std::vector<HloBuffer::Id>* dst);

  static bool ShouldSkipBufferAllocations(
      const HloInstruction* instruction, const ShapeIndex& idx,
      const HloInstruction* first_definition) {
    // Make GetTupleElement/kBitcast make alive only the tuple pointer if not
    // array shape.
    if ((instruction->opcode() == HloOpcode::kGetTupleElement ||
         instruction->opcode() == HloOpcode::kBitcast) &&
        !idx.empty()) {
      return true;
    }
    // Skip entry computation parameters because their memory usage is already
    // accounted for.
    if (first_definition->opcode() == HloOpcode::kParameter &&
        first_definition->parent()->IsEntryComputation()) {
      return true;
    }
    return false;
  }
  static bool ShouldSkipBufferReleases(const HloInstruction* instruction) {
    // Do not release parameter buffers as they are still in use by the caller.
    if (instruction->opcode() == HloOpcode::kParameter) {
      return true;
    }
    return false;
  }
  const HloAliasAnalysis* hlo_alias_analysis_;

  // Mapping from instruction to dense id.
  absl::flat_hash_map<const HloInstruction*, int32_t> instruction_ids_;

  // Combined vector of buffers allocated/released by each node.
  // See allocated_buffer_ids() and released_buffer_ids().
  std::vector<HloBuffer::Id> alloc_release_ids_;

  // Information kept per node that identifies allocated/released buffers.
  struct NodeAllocReleaseSpan {
    // Allocated buffers in alloc_release_ids_[start,start+num_alloc).
    // Released buffers in
    // alloc_release_ids_[start+num_alloc,start+num_alloc+num_release)
    uint32_t start;
    uint32_t num_alloc;
    uint32_t num_release;
  };

  // Mapping from dense instruction id to span information within
  // alloc_release_ids_.
  std::vector<NodeAllocReleaseSpan> alloc_release_spans_;

  // Live buffer presence set. This is used to determine if a buffer is live or
  // not in a fast way. Because this is checked very often in the evaluation
  // function of the scheduler quering the live_buffer_set_ object is too slow.
  // This is much faster in a tight loop. Also we use int8_t explicitly rather
  // than "bool" as "bool" is optimized and bit-packed trading memory for bit
  // extract operations.
  std::vector<int8_t> live_buffers_;
  // Set of live buffer ids.
  LiveBufferSet live_buffers_set_;
  const BufferInfoTracker& buffer_tracker_;
  // Cache of buffer objects defined that are output of instructions.
  absl::flat_hash_map<
      HloInstruction*,
      std::vector<std::pair<BufferInfoTracker::ValueInfo, ShapeIndex>>>
      output_buffers_;
  // Cache of buffer objects defined that are defined by instructions.
  absl::flat_hash_map<HloInstruction*,
                      std::vector<BufferInfoTracker::ValueInfo>>
      defined_buffers_;
  // Map with pressure_state object for other computations. It's updated by
  // the user of this class.
  const absl::flat_hash_map<const HloComputation*, MemoryPressureState>&
      pressure_state_cache_;
  // Current memory usage delta from the initial memory of the computation.
  int64_t live_memory_usage_;
  // Initial memory pressure at the bottom of the computation.
  int64_t initial_memory_pressure_;
  MemoryPressureState pressure_state_;
  bool top_down_scheduling_;
};

// Module memory pressure state object. Handles and holds all the objects used
// to store information about memory pressure for computations.
// Computes initial pressure state.
class ModulePressureState {
 public:
  using PressureStateMap =
      absl::flat_hash_map<const HloComputation*,
                          MemoryPressureTracker::MemoryPressureState>;
  ModulePressureState(
      const HloModule* module, const HloAliasAnalysis* hlo_alias_analysis,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes,
      bool top_down_scheduling = false)
      : module_(module),
        hlo_alias_analysis_(hlo_alias_analysis),
        buffer_tracker_(module, hlo_alias_analysis, shape_size_bytes),
        top_down_scheduling_(top_down_scheduling) {}
  void InitializePressureStates();
  bool ComputationIsMemoryTracked(const HloComputation* computation) const {
    return ContainsKey(memory_pressure_states_, computation);
  }
  // Get memory pressure state for a certain computation stored in this class.
  const MemoryPressureTracker::MemoryPressureState&
  GetPressureStateForComputation(const HloComputation* comp) const {
    auto it = memory_pressure_states_.find(comp);
    CHECK(it != memory_pressure_states_.end())
        << "No state for " << comp->name();
    return it->second;
  }
  // Updates the memory pressure state cache.
  void UpdatePressureStateForComputation(
      const HloComputation* comp,
      MemoryPressureTracker::MemoryPressureState state) {
    auto [it, inserted] = memory_pressure_states_.insert_or_assign(comp, state);

    if (!inserted) {
      // Rescheduling computation that has already been scheduled
      // can only happen during preference/heuristic rescheduling.
      // Recalculate memory peak.
      memory_peak_ = 0;
      for (auto& memory_state : memory_pressure_states_) {
        memory_peak_ = std::max(memory_peak_, memory_state.second.memory_peak);
      }
    } else {
      memory_peak_ = state.memory_peak;
    }
  }
  // Returns the underlying pressure state cache object
  const PressureStateMap& pressure_state_cache() const {
    return memory_pressure_states_;
  }
  // Returns the buffer tracker object.
  const BufferInfoTracker& buffer_tracker() const { return buffer_tracker_; }
  int64_t GetMemoryPeak() { return memory_peak_; }
  void SetMemoryPeak(int64_t peak) { memory_peak_ = peak; }

 private:
  const HloModule* module_;
  const HloAliasAnalysis* hlo_alias_analysis_;
  absl::flat_hash_map<const HloComputation*,
                      MemoryPressureTracker::MemoryPressureState>
      memory_pressure_states_;
  BufferInfoTracker buffer_tracker_;
  int64_t memory_peak_ = 0;
  bool top_down_scheduling_ = false;
};

// Light data structure containing information on the schedule of a computation,
// can be used by a heuristic to evaluate the quality of the schedule.
struct ComputationScheduleInfo {
  double total_wasted_cycles;
  uint64_t peak_memory;
};

// Implementation of the default scheduling algorithm.
class DefaultSchedulerCore : public SchedulerCore {
 public:
  using ReadyQueueSet = std::vector<HloGraphNode*>;
  using ResourceMap = absl::flat_hash_map<int64_t, int64_t>;
  using ShouldSkipNodeFunction = std::function<bool(const HloGraphNode*)>;

  // Class used to cache expensive information. Currently memory pressure
  // changes are cached. The caching is invalidated at the end of the scheduling
  // process for this next candidate. The information shouldn't survive across
  // scheduling two different instructions.
  struct ScheduleCandidate {
    void set_pressure_change(std::pair<const int64_t, const int64_t> v) {
      pressure_change_first = v.first;
      pressure_change_second = v.second;
      has_pressure_change = true;
    }
    void set_estimated_connected_send_ready_time(HloGraphNode::TimeCost v) {
      estimated_connected_send_ready_time = v;
      has_estimated_connected_send_ready_time = true;
    }
    void set_resource_constrained(bool v) {
      resource_constrained = v;
      has_resource_constrained = true;
    }

    HloGraphNode* node = nullptr;

    // Fields below are valid if the corresponding has_... field is true

    bool has_pressure_change = false;
    bool has_estimated_connected_send_ready_time = false;
    bool has_resource_constrained = false;

    int64_t pressure_change_first;
    int64_t pressure_change_second;
    HloGraphNode::TimeCost estimated_connected_send_ready_time;
    bool resource_constrained;
  };

  struct CandidateResult {
    const ScheduleCandidate& result;
    const char* reason;
  };

  using TargetSchedulingRule = std::function<std::optional<CandidateResult>(
      ScheduleCandidate&, ScheduleCandidate&)>;

  // Returns nullopt if both conditions are equal, otherwise returns the
  // candidate corresponding to the true condition.
  static inline std::optional<CandidateResult> ChooseBestCandidate(
      bool first_cond, const ScheduleCandidate& first_candidate,
      bool second_cond, const ScheduleCandidate& second_candidate,
      const char* reason) {
    if (first_cond == second_cond) {
      return std::nullopt;
    }
    return CandidateResult{first_cond ? first_candidate : second_candidate,
                           reason};
  }

  absl::Status SetGraphProcessingHook(
      const SchedulerCore::GraphProcessingHook& hook) override {
    graph_processing_hook_ = hook;
    return absl::OkStatus();
  }

  // The scheduling state contains everything that is required for the
  // bookkeeping of the scheduling algorithm. Functions that perform operations
  // over the scheduling state can directly operate on the state contained into
  // this struct instead of having to pass many individual pointers to elements
  // of the state.
  struct SchedulingState : public SchedulerCore::SchedulingState {
    HloScheduleGraph sched_graph;
    // Ready set for the nodes. Its ordered by our heuristic defined in
    // ReadySetLt.
    ReadyQueueSet ready_set;
    // Maximum allowed number of overlapping instructions using the key resource
    // type.
    ResourceMap max_concurrent_resource;
    // New scheduling sequence produced by the scheduler. This is in reversed
    // order (because we schedule bottom up). This will be required to be
    // reversed before assigning to the HloSchedule.
    std::vector<HloInstruction*> new_sequence_reversed;

    // Memory pressure during and after an instruction in a schedule.
    // (memory_after, memory_peak)
    absl::flat_hash_map<const HloInstruction*, std::pair<int64_t, int64_t>>
        memory_trace;
    // Units of time passed in the schedule. To keep track of latency hiding.
    HloGraphNode::TimeCost current_time = 0;
    // Resources and corresponding occupiers in flight.
    absl::flat_hash_map<int64_t, absl::flat_hash_set<const HloInstruction*>>
        resource_occupiers_in_flight;
    // Number of instructions using the key resource type in the set waiting to
    // be scheduled.
    ResourceMap resource_users_in_queue;
    // Number of nodes scheduled.
    int64_t scheduled_count = 0;
    // Class returning information about instruction cost and latency between
    // instructions.
    const LatencyEstimator* latency_estimator;
    // Class used to track which instructions are async instructions and which
    // async instructions computations contain. It also tracks target defined
    // states related to the async instructions.
    const AsyncTracker* async_tracker;
    // Tracker of memory pressure for the computation.
    std::unique_ptr<MemoryPressureTracker> memory_pressure_tracker;
    // Vector containing a list of nodes that aren't ready to schedule yet in
    // order of time when they are going to become ready.
    std::vector<const HloGraphNode*> next_ready_stack;
    // List of the graph edges currently occupying the key shareable resource
    // with projected finish times.
    absl::flat_hash_map<
        int64_t, std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>>
        shareable_resource_occupiers;
    // List of the graph nodes that release selective resources.
    std::vector<HloGraphNode*> selective_resource_releasers;
    // Similar to ready set, but only contains the no-op instructions.
    ReadyQueueSet nop_set;
    // Number of {scheduled, all} nodes that are a successor for the given
    // annotation.
    struct NumSuccessorsForAnnotation {
      int64_t scheduled = 0;
      int64_t all = 0;
    };
    struct NumPredecessorsForAnnotation {
      int64_t scheduled = 0;
      int64_t all = 0;
    };
    absl::flat_hash_map<int64_t, NumSuccessorsForAnnotation>
        num_successors_for_annotation;
    absl::flat_hash_map<int64_t, NumPredecessorsForAnnotation>
        num_predecessors_for_annotation;
    // List of annotations that are ready to be scheduled.
    absl::InlinedVector<int64_t, 2> ready_annotations;
    // List of annotated nodes that are ready to be scheduled.
    ReadyQueueSet annotation_ready;
    // Annotation that is currently being scheduled.
    int64_t ongoing_annotation = kInvalidAnnotation;
    // If this set is not empty it means that we shouldn't schedule any more
    // annotated nodes until empty.
    absl::flat_hash_set<HloGraphNode*> nodes_holding_annotations;
    // Reference to this scheduler run configuration.
    const SchedulerConfig& config;
    SchedulingState(
        const HloInstructionSequence* instr_sequence,
        std::shared_ptr<const SchedulingContext>& scheduling_context,
        std::unique_ptr<MemoryPressureTracker> memory_pressure_tracker,
        const SchedulerConfig& config)
        : sched_graph(&instr_sequence->instructions(), scheduling_context),
          latency_estimator(scheduling_context->GetLatencyEstimator().get()),
          async_tracker(scheduling_context->GetAsyncTracker().get()),
          memory_pressure_tracker(std::move(memory_pressure_tracker)),
          config(config) {}
  };

  using OverlapLimitRule =
      std::function<bool(const SchedulingState&, const HloGraphNode*)>;
  using PostProcessingFn = std::function<void(SchedulingState&)>;

  DefaultSchedulerCore(
      std::shared_ptr<const SchedulingContext> scheduling_context,
      const SchedulerConfig& config,
      TargetSchedulingRule target_scheduling_rule = nullptr,
      TargetSchedulingRule early_target_scheduling_rule = nullptr,
      PostProcessingFn post_processing_fn = nullptr,
      OverlapLimitRule scheduling_instruction_crosses_overlap_limit = nullptr)
      : config_(config),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule),
        post_processing_fn_(post_processing_fn),
        scheduling_instruction_crosses_overlap_limit_(
            scheduling_instruction_crosses_overlap_limit),
        scheduling_context_(std::move(scheduling_context)),
        top_down_scheduling_(config.top_down_scheduling) {}

  absl::Status InitializeScheduler(const HloModule* module) override;

  absl::Status CaptureScheduleProto() override {
    schedule_proto_ = ScheduleProto();
    *schedule_proto_->mutable_hlo_module() = module_->ToProto();

    return absl::OkStatus();
  }

  absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() override {
    if (!schedule_proto_.has_value()) {
      return absl::FailedPreconditionError("Schedule proto not captured.");
    }
    return schedule_proto_.value();
  }

  absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
  MakeSchedulingState(const HloComputation* computation) override;
  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) override;
  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation,
      std::shared_ptr<SchedulerCore::SchedulingState> sched_state) override;
  static bool AddOccupierToResource(
      HloGraphNode::TimeCost current_time, HloEdge& new_edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers,
      bool top_down_scheduling = false);
  static bool DeleteOccupierFromResource(
      HloGraphNode::TimeCost current_time, HloEdge& edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  int64_t GetMemoryPeak() override {
    return module_pressure_state_->GetMemoryPeak();
  }
  int64_t GetMemoryPeakForComputation(const HloComputation* computation) const {
    return module_pressure_state_->GetPressureStateForComputation(computation)
        .memory_peak;
  }

  uint64_t GetMemoryLimit() override { return config_.memory_limit; }
  void SetMemoryLimit(uint64_t new_limit) override {
    this->config_.memory_limit = new_limit;
  }
  int64_t GetRerunTimes() override { return config_.rerun; }

  // Returns the amount of resources an annotation group needs. The amount of
  // resources needed is schedule-order dependent. This function returns the
  // minimum or the maximum amount of resources needed for the given annotation
  // group based on the value of get_max_resources.
  absl::flat_hash_map<int64_t, int64_t> GetNumResourcesNeededForAnnotation(
      const SchedulingState& sched_state, int64_t annotation,
      bool get_max_resources = false);

  // Returns true if the given annotation group crosses the overlap limit.
  // If use_max_resources is true, the maximum amount of resources needed for
  // the annotation group is used to compare against the overlap limit.
  // Otherwise, the minimum amount of resources needed for the annotation group
  // is used.
  bool SchedulingAnnotationCrossesOverlapLimit(
      const SchedulingState& sched_state, int64_t annotation,
      bool use_max_resources = false);

  int64_t GetNumPredecessorsForAnnotation(const SchedulingState& sched_state,
                                          int64_t annotation) const;

  int64_t GetNumSuccessorsForAnnotation(const SchedulingState& sched_state,
                                        int64_t annotation) const;

  // Tries to schedule any of the ready annotation groups using either the
  // maximum or minimum amount of resources needed for the annotation group
  // based on value of use_max_resources. Returns true if any annotation group
  // is scheduled, false otherwise.
  absl::StatusOr<bool> TryScheduleOneAnnotationGroup(
      DefaultSchedulerCore::SchedulingState* sched_state,
      const HloComputation* computation, bool use_max_resources);

  ScheduleProto::ComputationScheduleProto ComputationScheduleToProto(
      const HloComputation* computation, const SchedulingState& sched_state,
      const LatencyEstimator& estimator,
      const std::vector<HloInstruction*>& instructions);

 protected:
  virtual void LogInstruction(const HloInstruction* instr) const;
  // Schedules the given annotated node.
  absl::Status AnnotatedSchedulingStep(
      HloGraphNode* node,
      DefaultSchedulerCore::SchedulingState* sched_state) const;
  // Schedules all of the nodes with the given annotation in computation.
  absl::Status ScheduleAnnotation(
      const HloComputation* computation, int64_t annotation,
      DefaultSchedulerCore::SchedulingState* sched_state) const;
  // Update node that has been scheduled.
  virtual absl::StatusOr<HloGraphNode::TimeCost> ScheduleNode(
      HloGraphNode* n, SchedulingState* sched_state) const;
  // Perform the scheduling of one or more instructions. Called every time the
  // ready set is not empty.
  virtual absl::Status SchedulingStep(SchedulingState* sched_state);
  // Pick a node to schedule according to cost model.
  virtual absl::StatusOr<HloGraphNode*> FindAndExtractBestNodeAvailable(
      SchedulingState& sched_state,
      DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node);

  std::unique_ptr<ModulePressureState> module_pressure_state_;
  SchedulerConfig config_;
  TargetSchedulingRule target_scheduling_rule_ = nullptr;
  TargetSchedulingRule early_target_scheduling_rule_ = nullptr;
  PostProcessingFn post_processing_fn_ = nullptr;
  OverlapLimitRule scheduling_instruction_crosses_overlap_limit_ = nullptr;
  bool is_default_scheduling_instruction_crosses_overlap_limit_ = false;
  std::unique_ptr<AnnotationTracker> annotation_tracker_;
  std::optional<ScheduleProto> schedule_proto_;
  const HloModule* module_ = nullptr;
  SchedulerCore::GraphProcessingHook graph_processing_hook_;
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  bool top_down_scheduling_ = false;
};

// A scheduler oriented to hiding latencies of operations that can run in
// parallel with other operations.
class LatencyHidingScheduler : public HloModulePass {
 public:
  struct SchedulerStatistics {
    const HloComputation* computation = nullptr;
    double all_gather_wasted_cycles = 0;
    double all_reduce_wasted_cycles = 0;
    double collective_broadcast_wasted_cycles = 0;
    double collective_permute_wasted_cycles = 0;
    double all_to_all_wasted_cycles = 0;
    double ragged_all_to_all_wasted_cycles = 0;
    double reduce_scatter_wasted_cycles = 0;
    double send_wasted_cycles = 0;
    double recv_wasted_cycles = 0;
    double call_wasted_cycles = 0;
    double total_cycles = 0;
    int64_t memory_pressure_peak = 0;

    double GetTotalWastedCycles() const {
      return all_gather_wasted_cycles + all_reduce_wasted_cycles +
             collective_broadcast_wasted_cycles +
             collective_permute_wasted_cycles + all_to_all_wasted_cycles +
             ragged_all_to_all_wasted_cycles + reduce_scatter_wasted_cycles +
             send_wasted_cycles + recv_wasted_cycles + call_wasted_cycles;
    }

    ScheduleProto::SchedulerStatisticsProto ToProto() const;
    std::string ToString() const;
  };

  LatencyHidingScheduler(
      std::shared_ptr<const SchedulingContext> scheduling_context,
      std::shared_ptr<SchedulerCore> scheduler_core)
      : scheduling_context_(std::move(scheduling_context)),
        scheduler_core_(std::move(scheduler_core)) {}
  constexpr static absl::string_view kName = "latency-hiding-scheduler";
  absl::string_view name() const override { return kName; }

  // Returns some printable statistics about the latency hiding for
  // operations that can run in parallel to help evaluating the performance of
  // the scheduler and improve it.
  // Optionally the caller can pass in the alias analysis and module pressure
  // state to save the time to construct them within the function. This is
  // useful when we repeatedly call this function across computations within the
  // same module.
  static SchedulerStatistics LatencyHidingStatistics(
      const HloComputation* computation,
      std::shared_ptr<const SchedulingContext> scheduling_context,
      const ModulePressureState* pressure_state = nullptr,
      MemoryPressureTracker* memory_pressure_tracker = nullptr);

  // Even with random preferences this function will always return a schedule
  // that obeys overlap constraints.
  absl::StatusOr<
      std::pair<std::vector<HloInstruction*>, ComputationScheduleInfo>>
  ScheduleWithPreferences(HloModule* module,
                          const std::vector<double>& preferences,
                          const HloComputation* computation);

  virtual void LogScheduleStatistics(const HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  std::shared_ptr<SchedulerCore> scheduler_core_;
  std::vector<HloComputation*> computations_to_schedule_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LATENCY_HIDING_SCHEDULER_H_
