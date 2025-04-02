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
#include <cstdint>
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
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/ptrvec.h"
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
  bool schedule_send_recvs = false;
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
  int64_t max_hops_to_closest_selective_overlap = 0;
  int64_t rerun = 0;
  int64_t parallel_collective_overlap_limit = 1;
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

  explicit AsyncTracker(
      const SchedulerConfig& config,
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : get_canonical_async_op_(std::move(func)), config_(config) {}

 private:
  const absl::flat_hash_map<int64_t, int64_t>& RecursivelyComputeResourceMap(
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
  virtual absl::Status InitializeScheduler(const HloModule* module) = 0;

  virtual absl::Status CaptureScheduleProto() = 0;

  virtual absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() = 0;

  virtual absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) = 0;

  virtual ~SchedulerCore() = default;
  virtual int64_t GetMemoryPeak() = 0;
  virtual void SetMemoryLimit(uint64_t new_limit) = 0;
  virtual uint64_t GetMemoryLimit() = 0;
  virtual int64_t GetRerunTimes() = 0;
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
    return annotations_.at(annotation).at(comp);
  }
  int64_t GetNumInstructions(const HloComputation* comp,
                             const int64_t annotation) {
    return annotations_[annotation][comp].size();
  }
  void FindSuccessors(const HloComputation* comp, const int64_t annotation) {
    absl::flat_hash_set<const HloInstruction*> seen_instructions(
        annotations_[annotation][comp].begin(),
        annotations_[annotation][comp].end());
    for (const HloInstruction* instr : annotations_.at(annotation).at(comp)) {
      for (const PtrVec<HloInstruction*>& users :
           {instr->users(), instr->control_successors()}) {
        for (HloInstruction* user : users) {
          if (!seen_instructions.contains(user) &&
              (GetAnnotation(user) == std::nullopt ||
               GetAnnotation(user).value() != annotation)) {
            annotation_successors_[annotation][comp].push_back(user);
            VLOG(3) << "Annotation group: " << annotation
                    << ", successor: " << user->name();
          }
          seen_instructions.insert(user);
        }
      }
    }
  }
  int64_t GetNumSuccessors(const HloComputation* comp,
                           const int64_t annotation) {
    if (!annotation_successors_[annotation].contains(comp)) {
      FindSuccessors(comp, annotation);
    }
    return annotation_successors_[annotation][comp].size();
  }
  std::vector<const HloInstruction*> GetSuccessors(const HloComputation* comp,
                                                   const int64_t annotation) {
    if (!annotation_successors_[annotation].contains(comp)) {
      FindSuccessors(comp, annotation);
    }
    return annotation_successors_[annotation][comp];
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
  HloGraphNode* TargetPtr() { return target_; }
  std::string ToString() const;

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
};

// Node in the schedule graph, plus information used for scheduling.
class HloGraphNode {
 public:
  using TimeCost = LatencyEstimator::TimeCost;

  // Nullptr is not a valid value for 'i'.
  explicit HloGraphNode(const HloInstruction* i, int64_t original_position)
      : instr_(i), original_position_(original_position) {}

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
      from->successors_.push_back(HloEdge(latency, to));
      from->outdegree_++;
    }
    if (!update_latency_if_edge_exists(to->GetPredecessors(), from, latency)) {
      to->predecessors_.push_back(HloEdge(latency, from));
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
    to->predecessors_.push_back(HloEdge(latency, from));
    to->indegree_++;
    from->successors_.push_back(HloEdge(latency, to));
    from->outdegree_++;
  }

  static void AddDependency(HloGraphNode* from, HloGraphNode* to,
                            const LatencyEstimator* latency_estimator) {
    AddDependency(from, to, latency_estimator->GetLatencyBetween(*from, *to));
  }

  const HloInstruction& GetInstr() const { return *instr_; }
  bool IsScheduled() const { return scheduled_; }
  int32_t GetIndegree() const { return indegree_; }
  int32_t GetOutdegree() const { return outdegree_; }
  TimeCost GetReadyTime() const { return ready_time_; }
  void SetIndegree(int64_t indeg) { indegree_ = indeg; }
  void SetOutdegree(int64_t outdeg) { outdegree_ = outdeg; }
  void SetScheduled() { scheduled_ = true; }
  void SetReadyTime(TimeCost ready_time) { ready_time_ = ready_time; }
  TimeCost GetCost() const { return cost_; }
  void SetCost(TimeCost cost) { cost_ = cost; }
  TimeCost GetAsyncDepth() const { return async_depth_; }
  TimeCost GetDepth() const { return depth_; }
  TimeCost GetGraphDepth() const { return graph_depth_; }
  void SetAsyncDepth(TimeCost async_depth) { async_depth_ = async_depth; }
  void SetDepth(TimeCost depth) { depth_ = depth; }
  void SetGraphDepth(TimeCost graph_depth) { graph_depth_ = graph_depth; }
  bool GetForceDelay() const { return force_delay_; }
  void SetForceDelay(bool force_delay) { force_delay_ = force_delay; }
  bool GetForceEarly() const { return force_early_; }
  void SetForceEarly(bool force_early) { force_early_ = force_early; }
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
  int64_t GetNumHopsToClosestSelectiveResourceOccupier() const {
    return num_hops_to_closest_selective_resource_occupier_;
  }
  void SetNumHopsToClosestSelectiveResourceOccupier(
      int64_t num_hops_to_closest_selective_resource_occupier) {
    num_hops_to_closest_selective_resource_occupier_ =
        num_hops_to_closest_selective_resource_occupier;
  }
  ResourcesVector GetResources() const { return resources_; }
  bool DoesOccupyAnyResource() const {
    return absl::c_any_of(resources_, [](const ResourcePair& resource) {
      return resource.second == ResourceUsageType::kResourceOccupy;
    });
  }
  bool DoesReleaseAnyResource() const {
    return absl::c_any_of(resources_, [](const ResourcePair& resource) {
      return resource.second == ResourceUsageType::kResourceRelease;
    });
  }
  bool DoesOccupyShareableResource(int64_t resource) const {
    return absl::c_linear_search(occupied_shareable_resources_, resource);
  }
  bool DoesReleaseResource(int64_t res) const {
    return absl::c_any_of(resources_, [res](const ResourcePair& resource) {
      return resource.second == ResourceUsageType::kResourceRelease &&
             resource.first == res;
    });
  }
  bool DoesReleaseResource(ResourceType res) const {
    return DoesReleaseResource(ResourceTypeToIndex(res));
  }
  bool DoesOccupyResource(int64_t res) const {
    return absl::c_any_of(resources_, [res](const ResourcePair& resource) {
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
    if (GetInstr().opcode() != HloOpcode::kWhile) {
      return resources_;
    }
    ResourcesVector result;
    for (const auto& [resource, usage] : resources_) {
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
    int64_t res_type = ResourceTypeToIndex(res);
    for (const auto& [resource_type, usage_type] : resources_) {
      if (resource_type == res_type) {
        return usage_type;
      }
    }
    return std::nullopt;
  }
  std::optional<ResourceUsageType> UsesResourceType(int64_t res) const {
    for (const auto& [resource_type, usage_type] : resources_) {
      if (resource_type == res) {
        return usage_type;
      }
    }
    return std::nullopt;
  }
  std::vector<int64_t> GetShareableResourcesOnEdge(const HloEdge& edge) const {
    HloGraphNode to = edge.Target();
    std::vector<int64_t> resources;
    absl::c_for_each(released_shareable_resources_,
                     [this, &to, &resources](const int64_t resource) {
                       if (to.DoesOccupyShareableResource(resource) &&
                           this->DoesReleaseResource(resource)) {
                         resources.push_back(resource);
                       }
                     });
    return resources;
  }
  absl::Span<HloEdge> GetPredecessors() {
    return absl::MakeSpan(predecessors_);
  }
  absl::Span<const HloEdge> GetPredecessors() const {
    return absl::MakeConstSpan(predecessors_);
  }
  void AddPredecessor(const HloEdge& e) { predecessors_.push_back(e); }
  absl::Span<HloEdge> GetSuccessors() { return absl::MakeSpan(successors_); }
  absl::Span<const HloEdge> GetSuccessors() const {
    return absl::MakeConstSpan(successors_);
  }
  void AddSuccessor(const HloEdge& e) { successors_.push_back(e); }
  int64_t GetOriginalPosition() const { return original_position_; }
  int64_t GetAnnotation() const { return annotation_; }
  absl::Status SetAnnotation(int64_t annotation) {
    TF_RET_CHECK(annotation_ == -1)
        << "Instruction " << instr_->name()
        << " has an existing annotation: " << annotation_;
    annotation_ = annotation;
    return absl::OkStatus();
  }
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
    absl::StrAppend(&result, "Predecessors:\n");
    for (const HloEdge& e : predecessors_) {
      absl::StrAppend(&result, e.ToString());
    }
    absl::StrAppend(&result, "Successors:\n");
    for (const HloEdge& e : successors_) {
      absl::StrAppend(&result, e.ToString());
    }
    if (async_tracker != nullptr) {
      absl::StrAppend(&result, "Resources:\n");
      for (const auto& [resource, usage] : resources_) {
        absl::StrAppend(
            &result, "\tResource: ", async_tracker->GetResourceName(resource),
            " usage: ", async_tracker->GetResourceUsageName(usage), "\n");
      }
    }
    return result;
  }

 private:
  friend class HloScheduleGraph;
  // List of predecessor edges.
  std::vector<HloEdge> predecessors_;
  // List of successor edges.
  std::vector<HloEdge> successors_;
  // Instruction this Graph node represents
  const HloInstruction* instr_;
  // The prosition of this node in the original order.
  int64_t original_position_;
  // Estimated time at which this node is gonna be ready to be scheduled.
  // The node should be added to the ready to be scheduled set when ready_time_
  // is less or equal to the current time in the schedule.
  TimeCost ready_time_ = std::numeric_limits<TimeCost>::max();
  // Number of predecessor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t indegree_ = 0;
  // Number of successor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t outdegree_ = 0;
  // Time cost of the execution of the operation of this nodes represent.
  TimeCost cost_ = 0.0;
  // Depth in latency terms of a node based on Async operation cost on the path.
  TimeCost async_depth_ = 0.0;
  // Depth in latency terms of node based on operation cost on the path to the
  // entry node.
  TimeCost depth_ = 0.0;
  // Depth in latency terms of node based on distance to the entry node.
  int64_t graph_depth_ = 0;
  // AsyncResources used by the node.
  ResourcesVector resources_;
  // Force the scheduling of the nodes with attribute set as late as possible.
  bool force_delay_ = false;
  // Force the scheduling of the nodes with attribute set as early as possible.
  bool force_early_ = false;
  // Whether this node has been scheduled or not yet.
  bool scheduled_ = false;
  // Shareable resources released by this node.
  absl::InlinedVector<int64_t, 1> released_shareable_resources_;
  // Shareable resources occupied by this node.
  absl::InlinedVector<int64_t, 1> occupied_shareable_resources_;
  // Whether this node can be overlapped with (can cover the latency/cost of)
  // edges occupying selective resources.
  bool valuable_for_selective_overlap_ = true;
  // Whether this node releases a selective resource.
  bool releases_selective_resource_ = false;
  // Whether this node occupies a selective resource.
  bool occupies_selective_resource_ = false;
  // Nums hops to closest selective resource occupier.
  int64_t num_hops_to_closest_selective_resource_occupier_ =
      std::numeric_limits<int64_t>::max();
  int64_t annotation_ = -1;
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
                   HloAliasAnalysis* alias_analysis,
                   const LatencyEstimator* latency_estimator,
                   const AsyncTracker* async_tracker);

  std::string ToString(const AsyncTracker* async_tracker = nullptr) const;

  HloGraphNode& GetNode(const HloInstruction* instr) const;

  std::vector<HloGraphNode*> FindBottomRoots() const;

  std::vector<HloGraphNode*> FindTopRoots() const;

  void InitializeGraphAnalysis(const AsyncTracker* async_tracker);

  void AnnotateGraph(const AnnotationTracker* annotation_tracker);

  // List of instructions in the original scheduled order. (Before scheduling).
  absl::Span<const HloInstruction* const> GetOriginalInstrList() const {
    return absl::MakeConstSpan(original_order_);
  }
  // Returns what was the original instruction position in the original order.
  int64_t OriginalInstructionPosition(const HloInstruction* instr) const {
    auto it = instr_order_map_.find(instr);
    CHECK(it != instr_order_map_.end());
    return it->second;
  }

 private:
  // Map that allocates the nodes of the graph.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloGraphNode>>
      nodes_;
  // Map containing the ordinal value for each instruction.
  absl::flat_hash_map<const HloInstruction*, int64_t> instr_order_map_;
  // List containing the original order (before scheduling) of the
  // instructions).
  std::vector<const HloInstruction*> original_order_;
  // Searches through node's predecessors to see if
  // possible_predecessor can be found.
  bool IsPredecessorTransitively(const HloGraphNode* node,
                                 const HloGraphNode* possible_predecessor);
};

// Tracks data about HloBuffers like where the first definition is in the
// original schedule and caches the buffer size (as Target::ShapeSize()) is
// expensive.
class BufferInfoTracker {
 public:
  struct ValueInfo {
    const HloBuffer* value = nullptr;
    const HloInstruction* first_definition = nullptr;
    int64_t buffer_size = 0;
  };
  BufferInfoTracker(const HloModule* module,
                    const HloAliasAnalysis* alias_analysis,
                    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes);
  static ValueInfo CreateBufferInfo(
      const HloBuffer* value, const HloInstruction* first_definition,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
    return ValueInfo{
        /*value=*/value, /*first_definition=*/first_definition,
        /*buffer_size=*/shape_size_bytes(value->values()[0]->shape())};
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
          pressure_state_cache)
      : hlo_alias_analysis_(hlo_alias_analysis),
        live_buffers_(hlo_alias_analysis->buffers().back().id() + 1),
        buffer_tracker_(buffer_tracker),
        pressure_state_cache_(pressure_state_cache),
        live_memory_usage_(0),
        initial_memory_pressure_(0) {}
  // Intiialize object to be ready to start tracking of computation.
  void Initialize(const HloComputation* computation,
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

 private:
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
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes)
      : module_(module),
        hlo_alias_analysis_(hlo_alias_analysis),
        buffer_tracker_(module, hlo_alias_analysis, shape_size_bytes) {}
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
    memory_pressure_states_[comp] = state;
    memory_peak_ = std::max(memory_peak_, state.memory_peak);
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
    HloGraphNode* node = nullptr;
    std::optional<std::pair<int64_t, int64_t>> pressure_change;
    std::optional<HloGraphNode::TimeCost> estimated_connected_send_ready_time;
    std::optional<bool> resource_constrained;
  };

  struct CandidateResult {
    ScheduleCandidate result;
    const char* reason;
  };

  using TargetSchedulingRule = std::function<std::optional<CandidateResult>(
      ScheduleCandidate&, ScheduleCandidate&)>;

  // Returns nullopt if both parameters are equal, otherwise true if the first
  // parameter is true and false if the second is true
  static std::optional<bool> TrueForOneOnly(bool first, bool second) {
    if (first == second) {
      return std::nullopt;
    }
    return first;
  }

  static std::optional<CandidateResult> ChooseBestCandidate(
      bool first_cond, const ScheduleCandidate& first_candidate,
      bool second_cond, const ScheduleCandidate& second_candidate,
      const char* reason) {
    if (auto cond = TrueForOneOnly(first_cond, second_cond)) {
      return CandidateResult{*cond ? first_candidate : second_candidate,
                             reason};
    }
    return std::nullopt;
  }

  // The scheduling state contains everything that is required for the
  // bookkeeping of the scheduling algorithm. Functions that perform operations
  // over the scheduling state can directly operate on the state contained into
  // this struct instead of having to pass many individual pointers to elements
  // of the state.
  struct SchedulingState {
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
    // async instructions computations contain.
    const AsyncTracker* async_tracker;
    // Tracker of memory pressure for the computation.
    MemoryPressureTracker* memory_pressure_tracker;
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
    // Number of scheduled nodes that are a successor for the given annotation.
    absl::flat_hash_map<int64_t, int64_t>
        num_scheduled_successors_for_annotation;
    // List of annotations that are ready to be scheduled.
    absl::InlinedVector<int64_t, 2> ready_annotations;
    // List of annotated nodes that are ready to be scheduled.
    ReadyQueueSet annotation_ready;
    // Annotation that is currently being scheduled.
    int64_t ongoing_annotation = -1;
    // Reference to this scheduler run configuration.
    const SchedulerConfig& config;
    SchedulingState(const HloInstructionSequence* instr_sequence,
                    HloAliasAnalysis* alias_analysis,
                    const LatencyEstimator* latency_estimator,
                    const AsyncTracker* async_tracker,
                    MemoryPressureTracker* memory_pressure_tracker,
                    const SchedulerConfig& config)
        : sched_graph(&instr_sequence->instructions(), alias_analysis,
                      latency_estimator, async_tracker),
          latency_estimator(latency_estimator),
          async_tracker(async_tracker),
          memory_pressure_tracker(memory_pressure_tracker),
          config(config) {}
  };

  using OverlapLimitRule =
      std::function<bool(const SchedulingState&, const HloGraphNode*)>;
  using PostProcessingFn = std::function<void(SchedulingState&)>;

  DefaultSchedulerCore(
      HloCostAnalysis::ShapeSizeFunction shape_size_bytes,
      const AsyncTracker* async_tracker,
      const LatencyEstimator* latency_estimator, const SchedulerConfig& config,
      TargetSchedulingRule target_scheduling_rule = nullptr,
      TargetSchedulingRule early_target_scheduling_rule = nullptr,
      PostProcessingFn post_processing_fn = nullptr,
      OverlapLimitRule scheduling_instruction_crosses_overlap_limit = nullptr)
      : shape_size_bytes_(shape_size_bytes),
        async_tracker_(async_tracker),
        latency_estimator_(latency_estimator),
        config_(config),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule),
        post_processing_fn_(post_processing_fn),
        scheduling_instruction_crosses_overlap_limit_(
            scheduling_instruction_crosses_overlap_limit) {}

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

  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) override;
  static bool AddOccupierToResource(
      HloGraphNode::TimeCost current_time, HloEdge& new_edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  static bool DeleteOccupierFromResource(
      HloGraphNode::TimeCost current_time, HloEdge& edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  int64_t GetMemoryPeak() override {
    return module_pressure_state_->GetMemoryPeak();
  }
  uint64_t GetMemoryLimit() override { return config_.memory_limit; }
  void SetMemoryLimit(uint64_t new_limit) override {
    this->config_.memory_limit = new_limit;
  }
  int64_t GetRerunTimes() override { return config_.rerun; }
  bool SchedulingAnnotationCrossesOverlapLimit(
      const SchedulingState& sched_state, int64_t annotation);
  absl::flat_hash_map<int64_t, int64_t> GetNumResourcesNeededForAnnotation(
      const SchedulingState& sched_state, int64_t annotation);

  ScheduleProto::ComputationScheduleProto ComputationScheduleToProto(
      const HloComputation* computation, const HloScheduleGraph& schedule_graph,
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

  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  std::unique_ptr<ModulePressureState> module_pressure_state_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  const AsyncTracker* async_tracker_;
  const LatencyEstimator* latency_estimator_;
  SchedulerConfig config_;
  TargetSchedulingRule target_scheduling_rule_ = nullptr;
  TargetSchedulingRule early_target_scheduling_rule_ = nullptr;
  PostProcessingFn post_processing_fn_ = nullptr;
  OverlapLimitRule scheduling_instruction_crosses_overlap_limit_ = nullptr;
  std::unique_ptr<AnnotationTracker> annotation_tracker_;
  std::optional<ScheduleProto> schedule_proto_;
  const HloModule* module_ = nullptr;
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
    double total_cycles = 0;
    int64_t memory_pressure_peak = 0;

    double GetTotalWastedCycles() const {
      return all_gather_wasted_cycles + all_reduce_wasted_cycles +
             collective_broadcast_wasted_cycles +
             collective_permute_wasted_cycles + all_to_all_wasted_cycles +
             ragged_all_to_all_wasted_cycles + reduce_scatter_wasted_cycles +
             send_wasted_cycles + recv_wasted_cycles;
    }

    ScheduleProto::SchedulerStatisticsProto ToProto() const;
    std::string ToString() const;
  };

  LatencyHidingScheduler(
      std::unique_ptr<LatencyEstimator> latency_estimator,
      std::unique_ptr<AsyncTracker> async_tracker,
      std::unique_ptr<SchedulerCore> scheduler_core,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes)
      : latency_estimator_(std::move(latency_estimator)),
        async_tracker_(std::move(async_tracker)),
        scheduler_core_(std::move(scheduler_core)),
        shape_size_bytes_(shape_size_bytes) {}
  constexpr static absl::string_view kName = "latency-hiding-scheduler";
  absl::string_view name() const override { return kName; }

  // Returns some printable statistics about the latency hiding for
  // operations that can run in parallel to help evaluating the performance of
  // the scheduler and improve it.
  static SchedulerStatistics LatencyHidingStatistics(
      const HloComputation* computation,
      const LatencyEstimator* latency_estimator,
      const AsyncTracker* async_tracker,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  virtual void LogScheduleStatistics(const HloComputation* computation);

 private:
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  std::unique_ptr<AsyncTracker> async_tracker_;
  std::unique_ptr<SchedulerCore> scheduler_core_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  absl::flat_hash_set<HloComputation*> computations_to_schedule_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LATENCY_HIDING_SCHEDULER_H_
