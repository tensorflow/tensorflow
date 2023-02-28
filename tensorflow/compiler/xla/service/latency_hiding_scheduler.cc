/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/latency_hiding_scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {

namespace {
struct CanonicalAsyncOp {
  HloOpcode outer;  // kAsyncStart or kAsyncDone
  HloOpcode inner;  // kAllReduce, kAllGather, kAllToAll, kCollectivePermute
};

CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncDone:
      return {hlo.opcode(), hlo.async_wrapped_opcode()};
    case HloOpcode::kAllReduceStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCollectivePermute};
    case HloOpcode::kAllReduceDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCollectivePermute};
    default:
      return {hlo.opcode(), hlo.opcode()};
  }
}

}  // namespace

LatencyEstimator::TimeCost ApproximateLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  // These values are empirically derived to obtain an overlap of one output
  // fusion/convolution with 1 async op or 5 loop fusions with an async op.
  static constexpr TimeCost kLowLatency = 1.0;
  static constexpr TimeCost kHighLatency = 5000.0;
  CanonicalAsyncOp from_op = GetCanonicalAsyncOp(from.GetInstr());
  CanonicalAsyncOp target_op = GetCanonicalAsyncOp(target.GetInstr());
  if (from_op.outer == HloOpcode::kAsyncStart &&
      target_op.outer == HloOpcode::kAsyncDone &&
      from_op.inner == target_op.inner) {
    return kHighLatency;
  }
  // Every other instruction we consider synchronous, which means the
  // latency between each of them is always one unit.
  return kLowLatency;
}

// Uses the approximate function for NodeCost based on a flag.
LatencyEstimator::TimeCost ApproximateLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (instr->IsLoopFusion()) {
    return kMediumCost;
  }
  if (instr->IsOutputFusion() || instr->opcode() == HloOpcode::kConvolution) {
    return kHighCost;
  }
  return kLowCost;
}

// Returns if this is an Async done op that the scheduler supports.
bool AsyncTracker::IsSupportedAsyncDone(const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kSendDone || op.outer == HloOpcode::kRecvDone) {
    return config_.schedule_send_recvs;
  }

  if (op.outer == HloOpcode::kAsyncDone) {
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectivePermute:
        return true;
      default:
        return false;
    }
  }
  return false;
}

// Returns if this is an Async op start that the scheduler supports.
bool AsyncTracker::IsSupportedAsyncStart(const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kSend || op.outer == HloOpcode::kRecv) {
    return config_.schedule_send_recvs;
  }

  if (op.outer == HloOpcode::kAsyncStart) {
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectivePermute:
        return true;
      default:
        return false;
    }
  }
  return false;
}

ResourcesVector AsyncTracker::GetResourcesFromInstruction(
    const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  auto get_resource_for_op = [](HloOpcode op) -> ResourceType {
    switch (op) {
      case HloOpcode::kAllReduce:
        return ResourceType::kAllReduce;
      case HloOpcode::kAllGather:
        return ResourceType::kAllGather;
      case HloOpcode::kAllToAll:
        return ResourceType::kAllToAll;
      case HloOpcode::kCollectivePermute:
        return ResourceType::kCollectivePermute;
      default:
        return ResourceType::kNoResource;
    }
  };
  if (op.outer == HloOpcode::kAsyncStart || op.outer == HloOpcode::kAsyncDone) {
    ResourceType type = get_resource_for_op(op.inner);
    if (type == ResourceType::kNoResource) {
      return {};
    }
    ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                  ? ResourceUsageType::kResourceRelease
                                  : ResourceUsageType::kResourceOccupy;
    return {std::make_pair(ResourceTypeToIndex(type), usage)};
  }

  switch (hlo.opcode()) {
    case HloOpcode::kAfterAll:
      // TODO(maggioni): Understand why AfterAll need to not be overlapped.
      return ResourcesVector{
          std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                         ResourceUsageType::kNoResource)};
    case HloOpcode::kRecv:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(&hlo)->is_host_transfer()
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceRelease)};
    case HloOpcode::kSend:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(&hlo)->is_host_transfer()
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceRelease)};
    case HloOpcode::kRecvDone:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(hlo.operand(0))
                  ->is_host_transfer()
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    case HloOpcode::kSendDone:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(hlo.operand(0))
                  ->is_host_transfer()
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    default:
      return ResourcesVector{};
  }
}

int64_t AsyncTracker::GetNumResourcesPerInstruction(
    ResourceType resource_type, const HloInstruction& instr) const {
  return GetNumResourcesPerInstruction(ResourceTypeToIndex(resource_type),
                                       instr);
}

int64_t AsyncTracker::GetNumResourcesPerInstruction(
    int64_t resource_type, const HloInstruction& instr) const {
  // For instructions not calling a computation then return 1 if the instruction
  // has opcode equal to 'async_done'
  if (instr.called_computations().empty() ||
      instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    return absl::c_any_of(GetResourcesFromInstruction(instr),
                          [resource_type](const ResourcePair& resource) {
                            return resource.second ==
                                       ResourceUsageType::kResourceOccupy &&
                                   (resource_type == resource.first);
                          })
               ? 1
               : 0;
  }
  std::function<void(const HloComputation*)> recursively_compute_resource_map =
      [this,
       &recursively_compute_resource_map](const HloComputation* computation) {
        absl::flat_hash_map<int64_t, int64_t> per_opcode_map;
        for (HloInstruction* instr : computation->instructions()) {
          if (IsSupportedAsyncDone(*instr)) {
            for (auto& resource : GetResourcesFromInstruction(*instr)) {
              ++per_opcode_map[resource.first];
            }
          }
          for (const HloComputation* called_comp :
               instr->called_computations()) {
            auto it = async_in_computation_cache_.find(called_comp);
            if (it == async_in_computation_cache_.end()) {
              recursively_compute_resource_map(called_comp);
              it = async_in_computation_cache_.find(called_comp);
              CHECK(it != async_in_computation_cache_.end());
            }
            for (auto& called_per_opcode_pair : it->second) {
              per_opcode_map[called_per_opcode_pair.first] +=
                  called_per_opcode_pair.second;
            }
          }
        }
        async_in_computation_cache_[computation] = std::move(per_opcode_map);
      };
  int64_t num_resources = 0;
  for (const HloComputation* computation : instr.called_computations()) {
    auto it = async_in_computation_cache_.find(computation);
    if (it == async_in_computation_cache_.end()) {
      recursively_compute_resource_map(computation);
      it = async_in_computation_cache_.find(computation);
      CHECK(it != async_in_computation_cache_.end());
    }
    auto opcode_it = it->second.find(resource_type);
    if (opcode_it == it->second.end()) {
      continue;
    }
    num_resources += opcode_it->second;
  }
  return num_resources;
}

void AsyncTracker::SetConcurrentResourceLimits(
    absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource) const {
  // Set the limits for default resources
  max_concurrent_resource[ResourceTypeToIndex(
      ResourceType::kCollectivePermute)] =
      config_.collective_permute_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllToAll)] =
      config_.all_to_all_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllGather)] =
      config_.all_gather_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllReduce)] =
      config_.all_reduce_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendRecv)] =
      config_.send_recv_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendHost)] =
      config_.send_recv_host_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kRecvHost)] =
      config_.send_recv_host_overlap_limit;
  // Set the limits for target-defined resources
  const int64_t first_target_resource =
      AsyncTracker::GetFirstTargetDefinedResource();
  for (int64_t i = 0; i < GetNumTargetDefinedResources(); ++i) {
    max_concurrent_resource[first_target_resource + i] =
        GetNumAvailableResources(first_target_resource + i);
  }
}

absl::string_view AsyncTracker::GetResourceName(int64_t resource_type) const {
  switch (resource_type) {
    case ResourceTypeToIndex(ResourceType::kNoResource):
      return "kNoResource";
    case ResourceTypeToIndex(ResourceType::kAllToAll):
      return "kAllToAll";
    case ResourceTypeToIndex(ResourceType::kAllGather):
      return "kAllGather";
    case ResourceTypeToIndex(ResourceType::kAllReduce):
      return "kAllReduce";
    case ResourceTypeToIndex(ResourceType::kCollectivePermute):
      return "kCollectivePermute";
    case ResourceTypeToIndex(ResourceType::kSendRecv):
      return "kSendRecv";
    case ResourceTypeToIndex(ResourceType::kSendHost):
      return "kSendHost";
    case ResourceTypeToIndex(ResourceType::kRecvHost):
      return "kRecvHost";
    default:
      return "not a default resource";
  }
}

int64_t AsyncTracker::GetNumTargetDefinedResources() const { return 0; }

int64_t AsyncTracker::GetNumAvailableResources(int64_t resource_type) const {
  return 0;
}

BufferInfoTracker::BufferInfoTracker(
    const HloModule* module, const HloAliasAnalysis* alias_analysis,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
  // Resize buffer_infos_ vector. The buffers in AliasAnalysis are sorted by
  // id.
  buffer_infos_.resize(alias_analysis->buffers().back().id() + 1);
  // Recursively walk the HLO graph looking inside every computation and
  // collecting buffer information. Effectively flattening the HLO schedule
  // across called computations.
  std::function<void(const HloComputation*)> process_computation =
      [&process_computation, module, alias_analysis, this,
       &shape_size_bytes](const HloComputation* computation) {
        const HloInstructionSequence& sequence =
            module->schedule().sequence(computation);
        for (int idx = 0; idx < sequence.size(); ++idx) {
          const HloInstruction* instruction = sequence.instructions()[idx];
          for (auto* called_computation : instruction->called_computations()) {
            if (called_computation->IsFusionComputation()) {
              continue;
            }
            process_computation(called_computation);
          }
          ShapeUtil::ForEachSubshape(
              instruction->shape(),
              [&](const Shape& subshape, const ShapeIndex& index) {
                for (const HloBuffer* buffer :
                     alias_analysis->ComputeBuffersAt(instruction, index)) {
                  if (buffer_infos_[buffer->id()].value == nullptr) {
                    buffer_infos_[buffer->id()] =
                        CreateBufferInfo(buffer, instruction, shape_size_bytes);
                  }
                }
              });
        }
      };
  process_computation(module->entry_computation());
}

void ModulePressureState::InitializePressureStates() {
  memory_pressure_states_.clear();
  std::function<void(HloComputation*,
                     const MemoryPressureTracker::LiveBufferSet&)>
      process_computation = [this, &process_computation](
                                HloComputation* computation,
                                const MemoryPressureTracker::LiveBufferSet&
                                    initial_live_buffers) {
        const HloInstructionSequence& sequence =
            module_->schedule().sequence(computation);
        MemoryPressureTracker tracker(hlo_alias_analysis_, buffer_tracker_,
                                      memory_pressure_states_);
        tracker.Initialize(computation, initial_live_buffers);
        VLOG(6) << "Pressure at bottom for " << computation->name() << ": "
                << tracker.memory_usage();
        for (int idx = sequence.size() - 1; idx >= 0; --idx) {
          const HloInstruction* instruction = sequence.instructions()[idx];
          if (!instruction->called_computations().empty()) {
            for (auto* called_computation :
                 instruction->called_computations()) {
              if (called_computation->IsFusionComputation()) {
                continue;
              }
              process_computation(called_computation, tracker.live_buffers());
            }
          }
          VLOG(10) << "Instruction: " << instruction->ToString();
          VLOG(10) << "Pressure change: "
                   << tracker.MemoryPressureDifference(instruction).first;
          VLOG(10) << "Current usage: " << tracker.memory_usage();
          tracker.UpdateBuffers(instruction);
          VLOG(10) << "Current usage after update: " << tracker.memory_usage();
          VLOG(10) << "Current peak after update: "
                   << tracker.pressure_state().memory_peak;
        }
        VLOG(6) << "Pressure peak for " << computation->name() << ": "
                << tracker.pressure_state().memory_peak;
        UpdatePressureStateForComputation(computation,
                                          tracker.pressure_state());
      };
  process_computation(module_->entry_computation(), {});
}

void MemoryPressureTracker::Initialize(
    const HloComputation* computation,
    const LiveBufferSet& initial_live_buffers) {
  live_memory_usage_ = 0;
  initial_memory_pressure_ = 0;
  pressure_state_ = MemoryPressureState{};
  output_buffers_.clear();
  defined_buffers_.clear();
  live_buffers_set_.clear();
  for (auto* instruction : computation->instructions()) {
    auto& output_values = this->output_buffers_[instruction];
    auto& defined_values = this->defined_buffers_[instruction];
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          for (const HloBuffer* buffer :
               hlo_alias_analysis_->ComputeBuffersAt(instruction, index)) {
            output_values.push_back(std::make_pair(
                buffer_tracker_.GetBufferInfo(buffer->id()), index));
            if (absl::c_any_of(buffer->values(), [&](const HloValue* value) {
                  return value->defining_instruction() == instruction;
                })) {
              defined_values.push_back(
                  buffer_tracker_.GetBufferInfo(buffer->id()));
            }
          }
        });
  }
  if (!initial_live_buffers.empty()) {
    for (HloBuffer::Id id : initial_live_buffers) {
      auto& buffer = buffer_tracker_.GetBufferInfo(id);
      live_buffers_[buffer.value->id()] = 1;
      initial_memory_pressure_ += buffer.buffer_size;
    }
    live_buffers_set_ = initial_live_buffers;
  } else {
    absl::c_fill(live_buffers_, 0);
  }
  pressure_state_.live_ids_at_bottom = live_buffers_set_;
}

void MemoryPressureTracker::UpdateBuffers(const HloInstruction* instruction) {
  int64_t computations_peak = 0;
  for (auto* called_comp : instruction->called_computations()) {
    if (called_comp->IsFusionComputation()) {
      continue;
    }
    auto it = pressure_state_cache_.find(called_comp);
    CHECK(it != pressure_state_cache_.end());
    computations_peak = std::max(computations_peak, it->second.memory_peak);
  }
  if (pressure_state_.memory_peak < live_memory_usage_ + computations_peak) {
    pressure_state_.memory_peak = live_memory_usage_ + computations_peak;
  }
  for (auto* op : instruction->operands()) {
    auto& output_values = output_buffers_[op];
    for (auto& info : output_values) {
      if (ShouldSkipBufferAllocations(instruction, info.second)) {
        continue;
      }
      if (live_buffers_[info.first.value->id()] == 0) {
        live_buffers_[info.first.value->id()] = 1;
        live_buffers_set_.insert(info.first.value->id());
        live_memory_usage_ += info.first.buffer_size;
      }
    }
  }
  pressure_state_.memory_peak =
      std::max(live_memory_usage_, pressure_state_.memory_peak);
  auto it = defined_buffers_.find(instruction);
  CHECK(it != defined_buffers_.end());
  if (!ShouldSkipBufferReleases(instruction)) {
    for (auto& b : it->second) {
      if (live_buffers_[b.value->id()] != 0) {
        if (b.first_definition == instruction) {
          live_memory_usage_ -= b.buffer_size;
          live_buffers_set_.erase(b.value->id());
        }
      }
    }
  }
}

// Return the memory pressure difference estimation if this instruction was
// scheduled.
std::pair<int64_t, int64_t> MemoryPressureTracker::MemoryPressureDifference(
    const HloInstruction* instruction) const {
  int64_t increase = 0;
  int64_t peak = 0;
  // Compute peak increase produced by called computations.
  if (!instruction->called_computations().empty()) {
    int64_t called_comp_peak = 0;
    for (auto* called_comp : instruction->called_computations()) {
      if (called_comp->IsFusionComputation()) {
        continue;
      }
      auto it = pressure_state_cache_.find(called_comp);
      CHECK(it != pressure_state_cache_.end());
      // Take max increase of the called computations.
      peak = called_comp_peak =
          std::max(called_comp_peak, it->second.memory_peak);
    }
  }
  // Allocate memory increase from the operand and record increase in peak.
  for (auto* op : instruction->operands()) {
    auto it = output_buffers_.find(op);
    CHECK(it != output_buffers_.end());
    for (auto& b : it->second) {
      if (ShouldSkipBufferAllocations(instruction, b.second)) {
        continue;
      }
      if (!live_buffers_[b.first.value->id()]) {
        increase += b.first.buffer_size;
      }
    }
  }
  peak = std::max(increase, peak);
  auto it = defined_buffers_.find(instruction);
  CHECK(it != defined_buffers_.end());
  // Decrease memory pressure if some buffers are released.
  if (!ShouldSkipBufferReleases(instruction)) {
    for (auto& b : it->second) {
      if (live_buffers_[b.value->id()]) {
        if (b.first_definition == instruction) {
          increase -= b.buffer_size;
        }
      }
    }
  }
  return std::make_pair(increase, peak);
}

DefaultSchedulerCore::ScheduleCandidate InitializeCandidate(
    HloGraphNode* node,
    const DefaultSchedulerCore::SchedulingState& sched_state) {
  DefaultSchedulerCore::ScheduleCandidate cand;
  cand.node = node;
  return cand;
}

namespace {

// Comparator for the ready set. This class represents the priority policies
// for the nodes in the ready set. The policy can be whatever is appropriate to
// reduce the execution time of the graph or achieve interesting properties
// (best CMEM/VMEM allocations, latency hiding, memory pressure ... etc).
class ReadySetLt {
 public:
  // Nullptr is not a valid value for 'sched_graph'. It needs to be a valid
  // schedule graph containing the nodes this comparator is meant to compare.
  // It needs to outlive the comparator object.
  explicit ReadySetLt(
      const DefaultSchedulerCore::SchedulingState* sched_state,
      DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule)
      : sched_state_(*sched_state),
        target_scheduling_rule_(target_scheduling_rule) {}
  // The comparison here implements the priority for the nodes in the ready set.
  DefaultSchedulerCore::CandidateResult operator()(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b) const {
    // Schedule according to ForceDelay first.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            !a.node->GetForceDelay(), a, !b.node->GetForceDelay(), b,
            "kForceDelay")) {
      return *value;
    }
    // Prioritize instructions that are NOPs as they have no memory pressure
    // issue and unlock different operations for being scheduled.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            IsNop(*a.node), a, IsNop(*b.node), b, "kIsNop")) {
      return *value;
    }
    std::pair<int64_t, int64_t> a_increase = std::make_pair(0LL, 0LL);
    std::pair<int64_t, int64_t> b_increase = std::make_pair(0LL, 0LL);
    // Check if memory pressure tracking is enabled. Even if it evaluate memory
    // pressure.
    if (sched_state_.config.memory_limit != UINT64_MAX &&
        sched_state_.memory_pressure_tracker->memory_usage() >
            (sched_state_.config.memory_limit / 2)) {
      a_increase = GetMemoryPressureChanges(a);
      b_increase = GetMemoryPressureChanges(b);
      // If out of memory reduce memory at all costs. Choose the instruction
      // that causes the most decrease (or least further increase) of memory
      // pressure.
      if (sched_state_.memory_pressure_tracker->memory_usage() >=
          sched_state_.config.memory_limit) {
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_increase.first < b_increase.first, a,
                b_increase.first < a_increase.first, b,
                "kDecreaseMemoryOverLimit")) {
          return *value;
        }
      }
      // Avoid to bring peak beyond limit. Choose instruction that doesn't do
      // so.
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_increase.second +
                      sched_state_.memory_pressure_tracker->memory_usage() <=
                  sched_state_.config.memory_limit,
              a,
              b_increase.second +
                      sched_state_.memory_pressure_tracker->memory_usage() <=
                  sched_state_.config.memory_limit,
              b, "kMemoryPeakOverLimit")) {
        return *value;
      }
    }
    // Some heuristic that try to prioritize unlocking "done" instructions
    // so that we can perform overlap. More fancy heuristics can be used by
    // discovering the closest "done" to every instruction and prioritize
    // those that are closer rather than ones that are further away.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            ShouldScheduleAsyncDone(*a.node), a,
            ShouldScheduleAsyncDone(*b.node), b, "kScheduleDone")) {
      return *value;
    }
    const ApproximateLatencyEstimator::TimeCost a_ready_interval =
        std::max(a.node->GetReadyTime() - sched_state_.current_time, 0.0);
    const ApproximateLatencyEstimator::TimeCost b_ready_interval =
        std::max(b.node->GetReadyTime() - sched_state_.current_time, 0.0);
    // Make sure that between two instructions that are not ready we first emit
    // the one that causes less stall. This allows to potentially expose more
    // opportunities for the other to overlap.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_ready_interval < b_ready_interval, a,
            b_ready_interval < a_ready_interval, b, "kLessStall")) {
      return *value;
    }
    if (sched_state_.config.aggressive_scheduling_policies) {
      // If an instruction releasing a resource is not resource constrained and
      // has an async depth of 0, delay it as much as possible to avoid
      // potential cost model inefficiencies.
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              /*first_cond=*/!(a.node->DoesReleaseAnyResource() &&
                               a.node->GetAsyncDepth() == 0 &&
                               !IsResourceConstrained(a)),
              a,
              /*second_cond=*/
              !(b.node->DoesReleaseAnyResource() &&
                b.node->GetAsyncDepth() == 0 && !IsResourceConstrained(b)),
              b, "kStartAtZeroDepth")) {
        return *value;
      }
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->DoesReleaseAnyResource() && IsResourceConstrained(a), a,
            b.node->DoesReleaseAnyResource() && IsResourceConstrained(b), b,
            "kFreeBackedupResource")) {
      return *value;
    }
    if (sched_state_.config.aggressive_scheduling_policies) {
      // Try to favor paths that are dependent of chains of async operations
      // with long latency as we want to get to them as soon as possible to
      // overlap them with computation.
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a.node->GetAsyncDepth() > b.node->GetAsyncDepth(), a,
              b.node->GetAsyncDepth() > a.node->GetAsyncDepth(), b,
              "kAsyncDepth")) {
        return *value;
      }
      // Favor nodes that are the closest in amount of latency they hide with
      // the highest amount of latency that needs to be hidden to avoid
      // wasting / big nodes over small async operations.
      if (!sched_state_.next_ready_stack.empty()) {
        HloGraphNode::TimeCost latest_ready =
            sched_state_.next_ready_stack.front()->GetReadyTime();
        HloGraphNode::TimeCost a_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - a.node->GetCost());
        HloGraphNode::TimeCost b_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - b.node->GetCost());
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_cost_diff < b_cost_diff, a, b_cost_diff < a_cost_diff, b,
                "kAvoidWaste")) {
          return *value;
        }
      }
    }
    //  Check if any operand is an async done operation of the two ops to be
    //  compared. Prioritize those to unlock async dones to be scheduled.
    //  TODO(maggioni): Develop a more complete analysis of the graph to
    //  prioritize candidates that would more likely unlock more async dones
    //  to be scheduled.
    bool a_operands = absl::c_any_of(
        a.node->GetInstr().operands(),
        [async_tracker = sched_state_.async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    bool b_operands = absl::c_any_of(
        b.node->GetInstr().operands(),
        [async_tracker = sched_state_.async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_operands, a, b_operands, b, "kUnlockDone")) {
      return *value;
    }
    if (target_scheduling_rule_) {
      if (auto value = target_scheduling_rule_(a, b)) {
        return *value;
      }
    }
    if (sched_state_.config.aggressive_scheduling_policies) {
      // Favor nodes that unlock other nodes to be scheduled if possible.
      // This makes us more flexible in what we can use in scheduling.
      int ready_if_a_scheduled = ReadyIfScheduled(*a.node);
      int ready_if_b_scheduled = ReadyIfScheduled(*b.node);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              ready_if_a_scheduled > ready_if_b_scheduled, a,
              ready_if_b_scheduled > ready_if_a_scheduled, b,
              "kCreatesMoreReadyNodes")) {
        return *value;
      }
    }
    // If we computed memory pressure increase of instructions when we don't
    // have a better choice let's just choose the one that decreases or
    // increases less memory pressure.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_increase.first < b_increase.first, a,
            b_increase.first < a_increase.first, b, "kDecreaseMemory")) {
      return *value;
    }
    // If none of the heuristics above triggers then prefer to schedule
    // according the original order so that we don't impact memory pressure.
    if (sched_state_.sched_graph.OriginalInstructionPosition(
            &a.node->GetInstr()) >
        sched_state_.sched_graph.OriginalInstructionPosition(
            &b.node->GetInstr())) {
      return {a, "kOriginalOrder"};
    }
    return {b, "kOriginalOrder"};
  }

 private:
  const DefaultSchedulerCore::SchedulingState& sched_state_;
  DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule_;

  int ReadyIfScheduled(const HloGraphNode& gn) const {
    int ready_nodes_if_scheduled = 0;
    for (auto& pred : gn.GetPredecessors()) {
      if (pred.Target().GetOutdegree() == 1) {
        ++ready_nodes_if_scheduled;
      }
    }
    return ready_nodes_if_scheduled;
  }
  static bool IsNop(const HloGraphNode& gn) {
    return gn.GetInstr().opcode() == HloOpcode::kGetTupleElement ||
           gn.GetInstr().opcode() == HloOpcode::kBitcast ||
           gn.GetInstr().IsEffectiveBitcast();
  }
  bool IsResourceConstrained(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (cand.resource_constrained) {
      return *cand.resource_constrained;
    }
    if (cand.node->GetResources().empty()) {
      cand.resource_constrained = false;
      return *(cand.resource_constrained);
    }
    cand.resource_constrained = false;
    for (const auto& [resource_type, usage_type] : cand.node->GetResources()) {
      auto max_it = sched_state_.max_concurrent_resource.find(resource_type);
      auto res_it = sched_state_.resource_users_in_queue.find(resource_type);
      cand.resource_constrained =
          max_it != sched_state_.max_concurrent_resource.end() &&
          max_it->second == 0 &&
          res_it != sched_state_.resource_users_in_queue.end() &&
          res_it->second > 0;
      if (*cand.resource_constrained) {
        return *cand.resource_constrained;
      }
    }
    return *cand.resource_constrained;
  }
  bool ShouldScheduleAsyncDone(const HloGraphNode& gn) const {
    if (!gn.DoesOccupyAnyResource()) {
      return false;
    }
    return !ShouldDelaySendHostDone(gn);
  }
  bool ShouldDelaySendHostDone(const HloGraphNode& gn) const {
    if (!gn.UsesResourceType(ResourceType::kSendHost).has_value() ||
        gn.GetInstr().opcode() != HloOpcode::kSendDone) {
      return false;
    }
    // Try to delay the send-done for host based operations like outside
    // compilation to avoid allocating memory unnecessarily.
    const HloGraphNode& start =
        sched_state_.sched_graph.GetNode(gn.GetInstr().operand(0));
    const LatencyEstimator::TimeCost latency =
        sched_state_.latency_estimator->GetLatencyBetween(start, gn);
    if (start.GetReadyTime() - sched_state_.current_time <= latency) {
      return false;
    }
    return true;
  }
  // Compute and cache memory pressure change computation for candidiate.
  std::pair<int64_t, int64_t> GetMemoryPressureChanges(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (cand.pressure_change) {
      return *cand.pressure_change;
    }
    cand.pressure_change =
        sched_state_.memory_pressure_tracker->MemoryPressureDifference(
            &cand.node->GetInstr());
    return *cand.pressure_change;
  }
};

}  // namespace

// Helper function to find the best node from the queues of scheduling state for
// scheduling.
HloGraphNode* DefaultSchedulerCore::FindAndExtractBestNodeAvailable(
    DefaultSchedulerCore::SchedulingState& sched_state,
    DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node) {
  auto scheduling_instruction_crosses_overlap_limit =
      [&sched_state](const HloInstruction& instr) {
        for (const auto& [resource, limit] :
             sched_state.max_concurrent_resource) {
          // No resources in flight of this kind. Continue.
          auto it = sched_state.resources_in_flight.find(resource);
          if (it == sched_state.resources_in_flight.end() || it->second == 0) {
            continue;
          }
          // Number of instances of 'resource' needed if this instruction was to
          // be scheduled.
          const int64_t num_resources_needed =
              sched_state.async_tracker->GetNumResourcesPerInstruction(resource,
                                                                       instr);
          if (limit < num_resources_needed) {
            return true;
          }
        }
        return false;
      };
  VLOG(6) << "Current time: " << sched_state.current_time;
  ReadySetLt ready_lt{&sched_state, target_scheduling_rule_};
  // Construct a schedule candidate for caching.
  ScheduleCandidate ready_chosen;
  auto chosen_it = sched_state.ready_set.end();
  // Try to pick nodes from the ready set first as are the ones that cause the
  // most latency hiding.
  for (auto ready_node_it = sched_state.ready_set.begin(),
            e = sched_state.ready_set.end();
       ready_node_it != e; ++ready_node_it) {
    if (should_skip_node && should_skip_node(*ready_node_it)) {
      continue;
    }
    // If this node would cause the max_concurrent_resource count to go beyond
    // the limit do not schedule it and pass to the next node.
    if (scheduling_instruction_crosses_overlap_limit(
            (*ready_node_it)->GetInstr())) {
      continue;
    }
    ScheduleCandidate ready_candidate =
        InitializeCandidate(*ready_node_it, sched_state);
    if (ready_chosen.node == nullptr) {
      ready_chosen = ready_candidate;
      chosen_it = ready_node_it;
      VLOG(6) << "Choosing from ready (" << ready_chosen.node->GetInstr().name()
              << ") Reason: First Candidate";
      continue;
    }
    // Compare the current candidate with the previous candidate.
    CandidateResult cand_result = ready_lt(ready_candidate, ready_chosen);
    const bool new_candidate_selected =
        cand_result.result.node == *ready_node_it;
    VLOG(6) << "Choosing from ready ("
            << (new_candidate_selected ? ready_candidate.node->GetInstr().name()
                                       : ready_chosen.node->GetInstr().name())
            << ") vs ("
            << (new_candidate_selected
                    ? ready_chosen.node->GetInstr().name()
                    : ready_candidate.node->GetInstr().name())
            << ") Reason: " << cand_result.reason;
    if (new_candidate_selected) {
      ready_chosen = cand_result.result;
      chosen_it = ready_node_it;
    }
  }
  if (ready_chosen.node == nullptr) {
    return nullptr;
  }
  CHECK(chosen_it != sched_state.ready_set.end());
  std::swap(*chosen_it, sched_state.ready_set.back());
  sched_state.ready_set.pop_back();
  return ready_chosen.node;
}

void DefaultSchedulerCore::LogInstruction(const HloInstruction* instr) const {
  VLOG(5) << instr->ToString();
}

StatusOr<HloGraphNode::TimeCost> DefaultSchedulerCore::ScheduleNode(
    HloGraphNode* n, DefaultSchedulerCore::SchedulingState* sched_state) const {
  // Insert the node into the sequence and mark it as scheduled.
  sched_state->new_sequence_reversed.push_back(
      const_cast<HloInstruction*>(&n->GetInstr()));
  n->SetScheduled();
  // If this node is an async start/done handle the increase/decrease the number
  // of outstanding async ops.
  for (auto& resource :
       sched_state->async_tracker->GetResourcesFromInstruction(n->GetInstr())) {
    if (resource.second == ResourceUsageType::kResourceRelease) {
      ++(sched_state->max_concurrent_resource[resource.first]);
    } else if (resource.second == ResourceUsageType::kResourceOccupy) {
      --(sched_state->max_concurrent_resource[resource.first]);
      --(sched_state->resource_users_in_queue[resource.first]);
    }
  }
  // Compute the new current time after scheduling this node. It is computed
  // as the highest time computed as the sum of the time a successor node has
  // been scheduled and the latency of the edge connecting this node to that
  // node.
  HloGraphNode::TimeCost schedule_time = sched_state->current_time;
  for (const HloEdge& pred : n->GetSuccessors()) {
    const HloGraphNode::TimeCost time_from_edge =
        pred.Target().GetReadyTime() + pred.Latency();
    if (time_from_edge > schedule_time) {
      schedule_time = time_from_edge;
    }
  }
  // Set the ready time to the scheduled time for scheduled nodes.
  n->SetReadyTime(schedule_time);
  HloGraphNode::TimeCost current_time = schedule_time + n->GetCost();
  auto ready_time_cmp = [](const HloGraphNode* a, const HloGraphNode* b) {
    return a->GetReadyTime() > b->GetReadyTime();
  };
  while (!sched_state->next_ready_stack.empty()) {
    const HloGraphNode* node = sched_state->next_ready_stack.front();
    if (node->GetReadyTime() < current_time) {
      std::pop_heap(sched_state->next_ready_stack.begin(),
                    sched_state->next_ready_stack.end(), ready_time_cmp);
      sched_state->next_ready_stack.pop_back();
      continue;
    }
    break;
  }

  // After scheduling the node we decided to schedule, release the nodes that
  // don't have any more successors unscheduled by putting them in the
  // ready_set. If a released node ready time is higher than the current time we
  // put it also in the next_ready_stack, which is used in the ReadySetLt class
  // for nodes cost comparison.
  for (HloEdge& edge : n->GetPredecessors()) {
    const int64_t current_outdegree = edge.Target().GetOutdegree();
    // Node is not ready yet. Decrease the outdegree and continue.
    if (current_outdegree != 1) {
      edge.Target().SetOutdegree(current_outdegree - 1);
      continue;
    }
    // This node is now ready to schedule. Set the outdegree to 0 and compute
    // the time at which it is gonna be ready to be scheduled. If the time is
    // not what the current time is we put it also in next_ready_stack.
    edge.Target().SetOutdegree(0);
    LatencyEstimator::TimeCost ready_time = current_time;
    for (const HloEdge& pred : edge.Target().GetSuccessors()) {
      const LatencyEstimator::TimeCost edge_time =
          pred.Target().GetReadyTime() + pred.Latency();
      if (edge_time > ready_time) {
        ready_time = edge_time;
      }
    }
    for (auto& resource :
         sched_state->async_tracker->GetResourcesFromInstruction(
             edge.Target().GetInstr())) {
      if (resource.second == ResourceUsageType::kResourceOccupy) {
        ++(sched_state->resource_users_in_queue[resource.first]);
      }
    }
    edge.Target().SetReadyTime(ready_time);
    sched_state->ready_set.push_back(&edge.Target());
    if (edge.Target().GetReadyTime() > current_time) {
      sched_state->next_ready_stack.push_back(&edge.Target());
      std::push_heap(sched_state->next_ready_stack.begin(),
                     sched_state->next_ready_stack.end(), ready_time_cmp);
    }
  }
  ++sched_state->scheduled_count;
  for (auto& resource :
       sched_state->async_tracker->GetResourcesFromInstruction(n->GetInstr())) {
    if (resource.second == ResourceUsageType::kResourceRelease) {
      --sched_state->resources_in_flight[resource.first];
    } else if (resource.second == ResourceUsageType::kResourceOccupy) {
      ++sched_state->resources_in_flight[resource.first];
    }
  }
  VLOG(10) << "Memory pressure before schedule: "
           << sched_state->memory_pressure_tracker->memory_usage();
  VLOG(10)
      << "Memory peak before schedule: "
      << sched_state->memory_pressure_tracker->pressure_state().memory_peak;
  sched_state->memory_pressure_tracker->UpdateBuffers(&n->GetInstr());
  VLOG(10) << "Memory pressure after schedule: "
           << sched_state->memory_pressure_tracker->memory_usage();
  VLOG(10)
      << "Memory peak after schedule: "
      << sched_state->memory_pressure_tracker->pressure_state().memory_peak;
  return current_time;
}

std::string HloEdge::ToString() const {
  return absl::StrCat("\tEdge: ", target_->GetInstr().name(),
                      " latency: ", Latency(), "\n");
}

HloScheduleGraph::HloScheduleGraph(
    const std::vector<HloInstruction*>* post_order_instructions,
    HloAliasAnalysis* alias_analysis, const LatencyEstimator* latency_estimator,
    const AsyncTracker* async_tracker)
    : original_order_(post_order_instructions->begin(),
                      post_order_instructions->end()) {
  HloComputation* comp = (*post_order_instructions)[0]->parent();
  auto reachability = HloReachabilityMap::Build(comp);
  int64_t current_pos = 0;
  // Allocating the graph nodes. One for each of the instructions in the
  // original instructions order.
  for (HloInstruction* instr : *post_order_instructions) {
    auto [new_node_it, inserted] = nodes_.try_emplace(
        instr, std::make_unique<HloGraphNode>(instr, current_pos));
    CHECK(inserted) << "Expected the value to not be already in the map";
    instr_order_map_[instr] = current_pos++;
    new_node_it->second->predecessors_.reserve(instr->operand_count());
    new_node_it->second->successors_.reserve(instr->user_count());
    new_node_it->second->cost_ = latency_estimator->NodeCost(instr);
    new_node_it->second->resources_ =
        async_tracker->GetResourcesFromInstruction(*instr);
  }
  // Cache used to detect if we already added a dependency between two nodes
  // to avoid duplicates in the predecessors/successors lists.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      dependencies_set;
  auto add_dependency_helper = [&dependencies_set, latency_estimator,
                                async_tracker](HloGraphNode* from,
                                               HloGraphNode* to) {
    // Get the latency between these two instructions for this edge.
    const LatencyEstimator::TimeCost latency =
        latency_estimator->GetLatencyBetween(*from, *to);
    // Adding dependencies as successors for the instruction we are
    // considering now (instr) and as predecessor for the user.
    from->successors_.push_back(HloEdge(latency, to));
    to->predecessors_.push_back(HloEdge(latency, from));
    ++to->indegree_;
    ++from->outdegree_;
    if (async_tracker->IsSupportedAsyncStart(to->GetInstr())) {
      dependencies_set[&to->GetInstr()].insert(&from->GetInstr());
    }
  };
  // Add dependencies edges between each of the graph nodes.
  for (const HloInstruction* instr : *post_order_instructions) {
    auto node_it = nodes_.find(instr);
    CHECK(node_it != nodes_.end()) << "We should have just allocated a node";
    HloGraphNode* instr_node = node_it->second.get();
    VLOG(10) << "Adding users for " << instr_node->GetInstr().ToString();
    // Add edges that derive from def->use relationships of the HLO graph.
    for (const HloInstruction* user : instr->users()) {
      VLOG(10) << "\tUser: " << user->ToString();
      auto user_node_it = nodes_.find(user);
      CHECK(user_node_it != nodes_.end());
      HloGraphNode* user_node = user_node_it->second.get();
      add_dependency_helper(instr_node, user_node);
    }
    for (const HloInstruction* ctrl_succ : instr->control_successors()) {
      VLOG(10) << "\tCtrl Successor: " << ctrl_succ->ToString();
      auto ctrl_succ_node_it = nodes_.find(ctrl_succ);
      CHECK(ctrl_succ_node_it != nodes_.end());
      HloGraphNode* ctrl_succ_node = ctrl_succ_node_it->second.get();
      add_dependency_helper(instr_node, ctrl_succ_node);
    }
    // To make sure an instruction that aliases with the buffer produced
    // by the async-done operation is not scheduled in between the start and the
    // done instruction as that buffer is in flux when the start happens.
    // Add an edge between this instruction and the start in this case.
    if (async_tracker->IsSupportedAsyncDone(*instr)) {
      const HloInstruction* async_start = instr->operand(0);
      if (alias_analysis != nullptr) {
        for (const HloBuffer* buffer :
             alias_analysis->ComputeBuffersAt(instr, {})) {
          for (const HloValue* value : buffer->values()) {
            if (value->defining_instruction() == instr) {
              continue;
            }
            for (const HloUse& use : value->GetUses()) {
              if (ContainsKey(instr_order_map_, use.instruction)) {
                // The instruction itself and later ones might be
                // identified as use.instruction. Add checks here to avoid
                // adding dependencies for these instructions.
                // Also don't add the dependency if it has been already added.
                auto dep_it = dependencies_set.find(async_start);
                if (use.instruction == async_start ||
                    reachability->IsReachable(instr, use.instruction) ||
                    dep_it->second.contains(use.instruction)) {
                  continue;
                }
                auto it = nodes_.find(use.instruction);
                CHECK(it != nodes_.end());
                HloGraphNode* pred_node = it->second.get();
                it = nodes_.find(async_start);
                CHECK(it != nodes_.end());
                HloGraphNode* start_node = it->second.get();
                pred_node->successors_.push_back(HloEdge(1, start_node));
                start_node->predecessors_.push_back(HloEdge(1, pred_node));
                ++pred_node->outdegree_;
                ++start_node->indegree_;
              }
            }
          }
        }
      }
    }
  }
}

std::string HloScheduleGraph::ToString() const {
  std::string result;
  std::vector<std::pair<const HloGraphNode*, int>> stack;
  for (const auto& node : nodes_) {
    if (node.second->predecessors_.empty()) {
      stack.push_back(std::make_pair(node.second.get(), 0));
    }
  }
  std::vector<const HloGraphNode*> order;
  absl::flat_hash_set<const HloGraphNode*> visited;
  while (!stack.empty()) {
    auto& val = stack.back();
    if (val.second == val.first->successors_.size()) {
      order.push_back(val.first);
      stack.pop_back();
      continue;
    }
    const int64_t next_child = val.second++;
    if (visited.insert(&val.first->successors_[next_child].Target()).second) {
      stack.push_back(
          std::make_pair(&val.first->successors_[next_child].Target(), 0));
    }
  }
  for (auto it = order.rbegin(), e = order.rend(); it != e; ++it) {
    absl::StrAppend(&result, (*it)->ToString());
  }
  return result;
}

HloGraphNode& HloScheduleGraph::GetNode(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return *it->second;
}

std::vector<HloGraphNode*> HloScheduleGraph::FindBottomRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetOutdegree() == 0) {
      roots.push_back(&node);
    }
  }
  return roots;
}

void HloScheduleGraph::InitializeGraphAnalysis(
    const AsyncTracker* async_tracker) {
  absl::flat_hash_map<HloGraphNode*, int> current_rank;
  std::vector<HloGraphNode*> stack;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    current_rank[&node] = node.GetIndegree();
    node.SetAsyncDepth(0.0);
    if (node.GetIndegree() == 0) {
      stack.push_back(&node);
    }
  }
  while (!stack.empty()) {
    auto* node = stack.back();
    stack.pop_back();
    if (async_tracker->IsSupportedAsyncDone(node->GetInstr())) {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth() + pred.Latency(),
                     node->GetAsyncDepth()));
      }
    } else {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth(), node->GetAsyncDepth()));
      }
    }
    for (auto& succ : node->GetSuccessors()) {
      if (--current_rank[&succ.Target()] == 0) {
        stack.push_back(&succ.Target());
      }
    }
  }
}

Status DefaultSchedulerCore::InitializeScheduler(const HloModule* module) {
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
  module_pressure_state_ = std::make_unique<ModulePressureState>(
      module, alias_analysis_.get(), shape_size_bytes_);
  module_pressure_state_->InitializePressureStates();
  return OkStatus();
}

Status DefaultSchedulerCore::SchedulingStep(SchedulingState* sched_state) {
  // Get the first available node for scheduling that is the node that
  // satisfies our ready heuristic the best.
  HloGraphNode* node = FindAndExtractBestNodeAvailable(
      *sched_state, /*should_skip_node=*/nullptr);
  CHECK(node != nullptr);
  TF_ASSIGN_OR_RETURN(sched_state->current_time,
                      ScheduleNode(node, sched_state));
  VLOG(5) << "Scheduled: ";
  XLA_VLOG_LINES(5, node->ToString());
  return OkStatus();
}

StatusOr<std::vector<HloInstruction*>>
DefaultSchedulerCore::ScheduleComputation(const HloComputation* computation) {
  const HloSchedule& module_schedule = computation->parent()->schedule();
  MemoryPressureTracker memory_pressure_tracker(
      alias_analysis_.get(), module_pressure_state_->buffer_tracker(),
      module_pressure_state_->pressure_state_cache());
  memory_pressure_tracker.Initialize(
      computation,
      module_pressure_state_->GetPressureStateForComputation(computation)
          .live_ids_at_bottom);

  SchedulingState sched_state(
      &module_schedule.sequence(computation), alias_analysis_.get(),
      latency_estimator_, async_tracker_, &memory_pressure_tracker, config_);
  async_tracker_->PostProcessScheduleGraph(&sched_state.sched_graph,
                                           latency_estimator_);
  sched_state.sched_graph.InitializeGraphAnalysis(async_tracker_);
  VLOG(5) << "Just built graph:";
  XLA_VLOG_LINES(5, sched_state.sched_graph.ToString());
  async_tracker_->SetConcurrentResourceLimits(
      sched_state.max_concurrent_resource);
  // Collect the bottom roots of the graph (nodes that don't have any
  // successor)
  // We are going to use them as starting point for scheduling.
  auto roots = sched_state.sched_graph.FindBottomRoots();
  for (HloGraphNode* root : roots) {
    // Set ready time for the roots 0.
    root->SetReadyTime(0.0);
  }
  VLOG(5) << "Initial memory pressure for " << computation->name() << ": "
          << memory_pressure_tracker.memory_usage();
  sched_state.ready_set.insert(sched_state.ready_set.end(), roots.begin(),
                               roots.end());
  // Schedule in order bottom up.
  while (!sched_state.ready_set.empty()) {
    VLOG(10) << "Current ready queue:";
    XLA_VLOG_LINES(10, [&sched_state]() {
      struct LogFormatter {
        void operator()(std::string* out, const HloGraphNode* n) const {
          out->append(absl::StrCat("\t", n->GetInstr().ToString(),
                                   " Ready time: ", n->GetReadyTime()));
        }
      };
      return absl::StrJoin(sched_state.ready_set, "\n", LogFormatter());
    }());

    TF_RETURN_IF_ERROR(SchedulingStep(&sched_state));
  }
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "New order";
    for (auto r_it = sched_state.new_sequence_reversed.rbegin(),
              e_it = sched_state.new_sequence_reversed.rend();
         r_it != e_it; ++r_it) {
      LogInstruction(*r_it);
    }
  }
  module_pressure_state_->UpdatePressureStateForComputation(
      computation, memory_pressure_tracker.pressure_state());
  CHECK_EQ(sched_state.new_sequence_reversed.size(),
           sched_state.sched_graph.GetOriginalInstrList().size())
      << "Not all instructions have been scheduled "
      << sched_state.new_sequence_reversed.size() << " vs "
      << sched_state.sched_graph.GetOriginalInstrList().size();
  VLOG(1) << "Total time: "
          << sched_state.sched_graph
                 .GetNode(sched_state.new_sequence_reversed.back())
                 .GetReadyTime();
  absl::c_reverse(sched_state.new_sequence_reversed);

  const auto& debug_options = xla::GetDebugOptionsFromFlags();
  if (debug_options.xla_dump_latency_hiding_schedule() &&
      computation->IsEntryComputation()) {
    int core_freq = latency_estimator_->CyclesPerMicrosecond();
    DumpLatencyHidingSchedule(computation, sched_state.sched_graph,
                              sched_state.new_sequence_reversed, core_freq,
                              debug_options);
  }

  return std::move(sched_state.new_sequence_reversed);
}

void DefaultSchedulerCore::DumpLatencyHidingSchedule(
    const HloComputation* computation, const HloScheduleGraph& schedule_graph,
    const std::vector<HloInstruction*>& instructions,
    const int cycles_per_microsecond, const DebugOptions& debug_options) {
  ScheduleProto proto;
  proto.set_computation_id(computation->unique_id());
  proto.set_cycles_per_microsecond(cycles_per_microsecond);

  const HloGraphNode& first_node = schedule_graph.GetNode(instructions.front());
  const double total_time = first_node.GetReadyTime() + first_node.GetCost();
  for (const HloInstruction* instr : instructions) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    const double start_time =
        total_time - (instr_node.GetReadyTime() + instr_node.GetCost());
    const double end_time = start_time + instr_node.GetCost();

    ScheduleProto::Instruction* instr_msg = proto.add_instructions();
    instr_msg->set_id(instr->unique_id());
    instr_msg->set_start_timestamp_cycles(start_time);
    instr_msg->set_end_timestamp_cycles(end_time);
  }
  *proto.mutable_hlo_module() = computation->parent()->ToProto();

  const std::string fn = absl::StrFormat("%s.schedule", computation->name());
  DumpProtobufToFile(proto, debug_options, fn);
}

LatencyHidingScheduler::SchedulerStatistics
LatencyHidingScheduler::LatencyHidingStatistics(
    const HloComputation* computation,
    const LatencyEstimator* latency_estimator,
    const AsyncTracker* async_tracker,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
  const HloModule* module = computation->parent();
  // A map keyed by outstanding collective op's opcode, with value of a tuple
  // including {instruction, scheduled_time, position in the original order}.
  absl::flat_hash_map<
      HloOpcode,
      std::vector<std::tuple<const HloInstruction*, int64_t, int64_t>>>
      outstanding_collectives;
  double current_time = 0;
  enum class AsyncKind {
    kNotAsync,
    kAllGather,
    kAllReduce,
    kCollectivePermute,
    kSend,
    kRecv,
  };
  auto opcode_to_async_kind = [](HloOpcode opcode) {
    switch (opcode) {
      case HloOpcode::kAllGatherStart:
        return AsyncKind::kAllGather;
      case HloOpcode::kAllReduceStart:
        return AsyncKind::kAllReduce;
      case HloOpcode::kCollectivePermuteStart:
        return AsyncKind::kCollectivePermute;
      case HloOpcode::kSend:
        return AsyncKind::kSend;
      case HloOpcode::kRecv:
        return AsyncKind::kRecv;
      default:
        return AsyncKind::kNotAsync;
    }
  };
  auto find_node_successor_edge = [](const HloGraphNode& graph_node,
                                     const HloGraphNode& successor_node) {
    auto edge_it = std::find_if(graph_node.GetSuccessors().begin(),
                                graph_node.GetSuccessors().end(),
                                [&successor_node](const HloEdge& edge) {
                                  return &edge.Target() == &successor_node;
                                });
    CHECK(edge_it != graph_node.GetSuccessors().end());
    return edge_it;
  };
  auto find_outstanding_async = [&outstanding_collectives](
                                    const HloInstruction* instr) {
    const auto& collective_vec = outstanding_collectives[instr->opcode()];
    auto it = absl::c_find_if(
        collective_vec,
        [instr](const std::tuple<const HloInstruction*, int64_t, int64_t>& p) {
          return instr == std::get<0>(p);
        });
    CHECK(it != collective_vec.end());
    return it;
  };
  absl::flat_hash_map<AsyncKind, double> wasted_time_per_collective;
  SchedulerConfig config;
  config.schedule_send_recvs = true;
  config.use_real_cost_model = true;
  std::unique_ptr<HloAliasAnalysis> hlo_alias_analysis =
      HloAliasAnalysis::Run(module).value();
  auto instructions_post_order = computation->MakeInstructionPostOrder();
  HloScheduleGraph schedule_graph(&instructions_post_order,
                                  /*alias_analysis=*/nullptr, latency_estimator,
                                  async_tracker);
  async_tracker->PostProcessScheduleGraph(&schedule_graph, latency_estimator);
  int64_t curr_pos = 0;
  for (const HloInstruction* instr :
       module->schedule().sequence(computation).instructions()) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    current_time += instr_node.GetCost();
    if (async_tracker->IsSupportedAsyncStart(*instr)) {
      outstanding_collectives[instr->opcode()].push_back(
          {instr, current_time, curr_pos});
    } else if (async_tracker->IsSupportedAsyncDone(*instr)) {
      const HloInstruction* start_instr = instr->operand(0);
      auto it = find_outstanding_async(start_instr);
      const HloGraphNode& start_node = schedule_graph.GetNode(std::get<0>(*it));
      auto edge_it = find_node_successor_edge(start_node, instr_node);
      const double async_wasted_cycles =
          std::max(0.0, edge_it->Latency() - (current_time - std::get<1>(*it)));
      wasted_time_per_collective[opcode_to_async_kind(start_instr->opcode())] +=
          async_wasted_cycles;
      current_time += async_wasted_cycles;
    }
    curr_pos++;
  }
  ModulePressureState module_pressure_state(module, hlo_alias_analysis.get(),
                                            shape_size_bytes);
  module_pressure_state.InitializePressureStates();
  const MemoryPressureTracker::MemoryPressureState* memory_pressure_state =
      module_pressure_state.ComputationIsMemoryTracked(computation)
          ? &module_pressure_state.GetPressureStateForComputation(computation)
          : nullptr;
  MemoryPressureTracker mem_pressure_tracker(
      hlo_alias_analysis.get(), module_pressure_state.buffer_tracker(),
      module_pressure_state.pressure_state_cache());
  if (memory_pressure_state != nullptr) {
    mem_pressure_tracker.Initialize(computation,
                                    memory_pressure_state->live_ids_at_bottom);
  }
  return LatencyHidingScheduler::SchedulerStatistics{
      /*computation=*/computation,
      /*all_gather_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kAllGather],
      /*all_reduce_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kAllReduce],
      /*collective_permute_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kCollectivePermute],
      /*send_wasted_cycles=*/wasted_time_per_collective[AsyncKind::kSend],
      /*recv_wasted_cycles=*/wasted_time_per_collective[AsyncKind::kRecv],
      /*total_cycles=*/current_time,
      /*memory_pressure_peak=*/
      memory_pressure_state ? mem_pressure_tracker.initial_memory_pressure() +
                                  memory_pressure_state->memory_peak
                            : 0};
}

// Prints a SchedulerStatistics object.
std::string LatencyHidingScheduler::SchedulerStatisticsString(
    const SchedulerStatistics& sched_stats) {
  std::string result;
  if (sched_stats.computation != nullptr) {
    absl::StrAppend(&result,
                    "For computation: ", sched_stats.computation->name(), "\n");
  }
  absl::StrAppend(&result, "Total wasted cycles: ",
                  sched_stats.all_gather_wasted_cycles +
                      sched_stats.all_reduce_wasted_cycles +
                      sched_stats.collective_permute_wasted_cycles +
                      sched_stats.send_wasted_cycles +
                      sched_stats.recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-permute: ",
                  sched_stats.collective_permute_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-gather: ",
                  sched_stats.all_gather_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-reduce: ",
                  sched_stats.all_reduce_wasted_cycles, "\n");
  absl::StrAppend(&result,
                  "Wasted cycles for send: ", sched_stats.send_wasted_cycles,
                  "\n");
  absl::StrAppend(&result,
                  "Wasted cycles for recv: ", sched_stats.recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Total cycles: ", sched_stats.total_cycles, "\n");
  absl::StrAppend(&result, "Memory pressure peak (bytes): ",
                  sched_stats.memory_pressure_peak, "\n");
  return result;
}

void LatencyHidingScheduler::LogScheduleStatistics(
    const HloComputation* computation) {
  XLA_VLOG_LINES(1, SchedulerStatisticsString(LatencyHidingStatistics(
                        computation, latency_estimator_.get(),
                        async_tracker_.get(), shape_size_bytes_)));
}

StatusOr<bool> LatencyHidingScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "Original module:";
  XLA_VLOG_LINES(5, module->ToString());
  // Currently we expect that a schedule that minimizes memory pressure is
  // provided as a base. It's not necessary for the algorithm itself but it
  // allows us to not having to think for now about memory pressure.
  std::vector<HloComputation*> computations_to_schedule;
  computations_to_schedule.reserve(module->computation_count());
  // Collect which computations have latency hiding opportunities.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : computation->instructions()) {
      if (async_tracker_->IsSupportedAsyncStart(*instr) ||
          async_tracker_->IsSupportedAsyncDone(*instr)) {
        computations_to_schedule.push_back(computation);
        break;
      }
    }
  }

  if (computations_to_schedule.empty()) {
    return false;
  }

  TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
  for (HloComputation* computation : computations_to_schedule) {
    VLOG(1) << "Statistics before scheduling:";
    LogScheduleStatistics(computation);
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        scheduler_core_->ScheduleComputation(computation));
    module->schedule().set_sequence(computation,
                                    absl::MakeConstSpan(new_schedule));
    VLOG(1) << "Statistics after scheduling:";
    LogScheduleStatistics(computation);
  }
  return true;
}

}  // namespace xla
