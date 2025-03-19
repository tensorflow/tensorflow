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

#include "xla/service/latency_hiding_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/map_util.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

const int64_t kDefaultMemorySpace = 0;

bool IsNopInstruction(const HloInstruction& hlo) {
  HloOpcode op = hlo.opcode();
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kBroadcast || op == HloOpcode::kIota ||
         hlo.IsEffectiveBitcast() ||
         (op == HloOpcode::kTuple && hlo.user_count() == 1 &&
          hlo.users().front()->opcode() == HloOpcode::kWhile);
}

bool InstructionDefinesValue(const HloInstruction* instruction,
                             const HloValue* value) {
  if (value->defining_instruction() == instruction) {
    return true;
  }
  if (value->shape().has_layout() &&
      value->shape().layout().memory_space() != kDefaultMemorySpace) {
    return false;
  }
  // Also check if the instruction is a call to a computation that defines the
  // value. This is needed in cases, e.g., where we wrap a value-defining
  // instruction in a async call for offloading, and the async start itself will
  // effectively define the value in the current scope that the scheduler is
  // running in.
  if (instruction->opcode() == HloOpcode::kAsyncStart) {
    if (instruction->async_wrapped_opcode() == HloOpcode::kCall) {
      return instruction->async_wrapped_instruction()
                 ->called_computations()[0]
                 ->root_instruction() == value->defining_instruction();
    }
    return instruction->async_wrapped_instruction() ==
           value->defining_instruction();
  }
  return false;
}

bool InstructionFirstDefinesBuffer(
    const HloInstruction* instruction,
    const BufferInfoTracker::ValueInfo& buffer_value_info) {
  if (buffer_value_info.first_definition == instruction) {
    return true;
  }
  if (buffer_value_info.value->values()[0]->shape().has_layout() &&
      buffer_value_info.value->values()[0]->shape().layout().memory_space() !=
          kDefaultMemorySpace) {
    return false;
  }
  // Similar to logic above, also check if the instruction is a call to a
  // computation that defines the value.
  if (instruction->opcode() == HloOpcode::kAsyncStart) {
    if (instruction->async_wrapped_opcode() == HloOpcode::kCall) {
      return instruction->async_wrapped_instruction()
                 ->called_computations()[0]
                 ->root_instruction() == buffer_value_info.first_definition;
    }
    return instruction->async_wrapped_instruction() ==
           buffer_value_info.first_definition;
  }
  return false;
}

}  // namespace

CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncDone:
      if (hlo.async_wrapped_opcode() == HloOpcode::kCall) {
        return {hlo.opcode(), hlo.async_wrapped_instruction()
                                  ->called_computations()[0]
                                  ->root_instruction()
                                  ->opcode()};
      }
      return {hlo.opcode(), hlo.async_wrapped_opcode()};
    case HloOpcode::kAllReduceStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCollectivePermute};
    case HloOpcode::kCopyStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCopy};
    case HloOpcode::kCopyDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCopy};
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

bool LatencyEstimator::IsAsyncPair(const HloGraphNode& from,
                                   const HloGraphNode& target) const {
  CanonicalAsyncOp from_op = GetCanonicalAsyncOp(from.GetInstr());
  CanonicalAsyncOp target_op = GetCanonicalAsyncOp(target.GetInstr());
  return from_op.outer == HloOpcode::kAsyncStart &&
         target_op.outer == HloOpcode::kAsyncDone &&
         from_op.inner == target_op.inner;
}

bool LatencyEstimator::IsP2pPair(const HloGraphNode& from,
                                 const HloGraphNode& target) const {
  return (from.GetInstr().opcode() == HloOpcode::kSend &&
          target.GetInstr().opcode() == HloOpcode::kSendDone) ||
         (from.GetInstr().opcode() == HloOpcode::kRecv &&
          target.GetInstr().opcode() == HloOpcode::kRecvDone);
}

LatencyEstimator::TimeCost ApproximateLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  if (IsAsyncPair(from, target)) {
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
    // If there are parallel thread computations, always schedule.
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return true;
    }
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kRaggedAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kCopy:
      case HloOpcode::kReduceScatter:
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
    // If there are parallel thread computations, always schedule.
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return true;
    }
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kRaggedAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kCopy:
      case HloOpcode::kReduceScatter:
        return true;
      default:
        return false;
    }
  }
  return false;
}

ResourcesVector AsyncTracker::GetResourcesFromInstructionImpl(
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
      case HloOpcode::kRaggedAllToAll:
        return ResourceType::kRaggedAllToAll;
      case HloOpcode::kCollectiveBroadcast:
        return ResourceType::kCollectiveBroadcast;
      case HloOpcode::kCollectivePermute:
        return ResourceType::kCollectivePermute;
      case HloOpcode::kCopy:
        return ResourceType::kCopy;
      case HloOpcode::kReduceScatter:
        return ResourceType::kReduceScatter;
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
    case HloOpcode::kRecvDone: {
      const HloSendRecvInstruction* recv =
          DynCast<HloSendRecvInstruction>(hlo.operand(0));
      return ResourcesVector{
          (recv != nullptr && recv->is_host_transfer())
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    }
    case HloOpcode::kSendDone: {
      const HloSendRecvInstruction* send =
          DynCast<HloSendRecvInstruction>(hlo.operand(0));
      return ResourcesVector{
          (send != nullptr && send->is_host_transfer())
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    }
    default:
      return ResourcesVector{};
  }
}

absl::Span<const ResourcePair> AsyncTracker::GetResourcesFromInstruction(
    const HloInstruction& hlo) const {
  auto [it, inserted] = resources_cache_.emplace(&hlo, ResourcesVector{});
  if (inserted) {
    it->second = GetResourcesFromInstructionImpl(hlo);
  }
  return it->second;
}

int64_t AsyncTracker::GetNumResourcesPerInstruction(
    ResourceType resource_type, const HloInstruction& instr) const {
  return GetNumResourcesPerInstruction(ResourceTypeToIndex(resource_type),
                                       instr);
}

// Returns the number of "occupy" type of resources used by the instructions in
// the given computation. If there are multiple instructions in the computation
// that have the exact same resource usages, it only counts one of them. For
// example, if there are two non-overlapping async all-gathers in a while loop,
// this will have 1 for all-gather in the returned map for the while
// instruction. This is because there is no proof that those all-gathers will
// overlap each other and over- counting such resources causes the while not
// being scheduled due to the resource limits (checked in
// scheduling_node_crosses_overlap_limit).
//
// If an instruction uses multiple instances of the same "occupy" type of
// resource, that number is respected and returned in the resulting map.
const absl::flat_hash_map<int64_t, int64_t>&
AsyncTracker::RecursivelyComputeResourceMap(
    const HloComputation* computation) const {
  auto& per_opcode_map = async_in_computation_cache_[computation];
  if (per_opcode_map != nullptr) {
    return *per_opcode_map;
  }
  per_opcode_map = std::make_unique<absl::flat_hash_map<int64_t, int64_t>>();
  auto* m = per_opcode_map.get();
  absl::flat_hash_set<int64_t> seen_resources_per_comp;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsSupportedAsyncDone(*instr)) {
      absl::flat_hash_set<int64_t> seen_resources_per_inst;
      for (const auto& resource : GetResourcesFromInstruction(*instr)) {
        if (seen_resources_per_comp.contains(resource.first)) {
          continue;
        }
        ++(*m)[resource.first];
        seen_resources_per_inst.insert(resource.first);
      }
      seen_resources_per_comp.insert(seen_resources_per_inst.begin(),
                                     seen_resources_per_inst.end());
    }
    for (const HloComputation* called_comp : instr->called_computations()) {
      for (auto& called_per_opcode_pair :
           RecursivelyComputeResourceMap(called_comp)) {
        if (seen_resources_per_comp.contains(called_per_opcode_pair.first)) {
          continue;
        }
        (*m)[called_per_opcode_pair.first] += called_per_opcode_pair.second;
        seen_resources_per_comp.insert(called_per_opcode_pair.first);
      }
    }
  }
  return *m;
}

int64_t AsyncTracker::GetNumResourcesPerInstruction(
    int64_t resource_type, const HloInstruction& instr) const {
  // For instructions not calling a computation, or async start/done
  // instructions, we directly check the resources from the instruction.
  if (instr.called_computations().empty() ||
      instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    return absl::c_count_if(GetResourcesFromInstruction(instr),
                            [resource_type](const ResourcePair& resource) {
                              return resource.second ==
                                         ResourceUsageType::kResourceOccupy &&
                                     (resource_type == resource.first);
                            });
  }
  int64_t num_resources = 0;
  for (const HloComputation* computation : instr.called_computations()) {
    const auto& map = RecursivelyComputeResourceMap(computation);
    auto opcode_it = map.find(resource_type);
    if (opcode_it != map.end()) {
      num_resources += opcode_it->second;
      // We can return early if we have found the resource we are looking for.
      // There is no need to check each called computation.
      break;
    }
  }
  return num_resources;
}

void AsyncTracker::SetConcurrentResourceLimits(
    absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource) const {
  // Set the limits for default resources
  max_concurrent_resource[ResourceTypeToIndex(
      ResourceType::kCollectiveBroadcast)] =
      config_.collective_broadcast_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(
      ResourceType::kCollectivePermute)] =
      config_.collective_permute_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kCopy)] =
      config_.copy_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllToAll)] =
      config_.all_to_all_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kRaggedAllToAll)] =
      config_.ragged_all_to_all_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllGather)] =
      config_.all_gather_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllReduce)] =
      config_.all_reduce_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kReduceScatter)] =
      config_.reduce_scatter_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendRecv)] =
      config_.send_recv_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendHost)] =
      config_.send_recv_host_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kRecvHost)] =
      config_.send_recv_host_overlap_limit;
  // Set the limits for target-defined resources.
  for (int64_t resource_type = GetTargetDefinedResourceTypeBegin();
       resource_type <
       GetTargetDefinedResourceTypeBegin() + GetNumTargetDefinedResources();
       ++resource_type) {
    CHECK_GT(GetNumAvailableResources(resource_type), 0)
        << "Target-defined resource with id " << resource_type
        << " has a concurrency limit of 0. Please set it to a positive value "
           "by making sure GetNumTargetDefinedResources returns the correct "
           "limit.";
    max_concurrent_resource[resource_type] =
        GetNumAvailableResources(resource_type);
  }
}

absl::string_view AsyncTracker::GetResourceName(int64_t resource_type) const {
  switch (resource_type) {
    case ResourceTypeToIndex(ResourceType::kNoResource):
      return "kNoResource";
    case ResourceTypeToIndex(ResourceType::kAllToAll):
      return "kAllToAll";
    case ResourceTypeToIndex(ResourceType::kRaggedAllToAll):
      return "kRaggedAllToAll";
    case ResourceTypeToIndex(ResourceType::kAllGather):
      return "kAllGather";
    case ResourceTypeToIndex(ResourceType::kAllReduce):
      return "kAllReduce";
    case ResourceTypeToIndex(ResourceType::kCollectiveBroadcast):
      return "kCollectiveBroadcast";
    case ResourceTypeToIndex(ResourceType::kCollectivePermute):
      return "kCollectivePermute";
    case ResourceTypeToIndex(ResourceType::kCopy):
      return "kCopy";
    case ResourceTypeToIndex(ResourceType::kSendRecv):
      return "kSendRecv";
    case ResourceTypeToIndex(ResourceType::kSendHost):
      return "kSendHost";
    case ResourceTypeToIndex(ResourceType::kRecvHost):
      return "kRecvHost";
    case ResourceTypeToIndex(ResourceType::kReduceScatter):
      return "kReduceScatter";
    default:
      return "Not a valid default resource";
  }
}

absl::string_view AsyncTracker::GetResourceUsageName(
    ResourceUsageType resource_usage_type) const {
  return GetResourceUsageName(ResourceUsageTypeToIndex(resource_usage_type));
}

ResourceHazardType AsyncTracker::GetResourceHazardType(
    int64_t resource_type) const {
  if (resource_type == ResourceTypeToIndex(ResourceType::kCopy)) {
    return ResourceHazardType::kShareable;
  }
  return ResourceHazardType::kUnshareable;
}

absl::string_view AsyncTracker::GetResourceUsageName(
    int64_t resource_usage_type) const {
  switch (resource_usage_type) {
    case ResourceUsageTypeToIndex(ResourceUsageType::kNoResource):
      return "kNoResource";
    case ResourceUsageTypeToIndex(ResourceUsageType::kResourceOccupy):
      return "kResourceOccupy";
    case ResourceUsageTypeToIndex(ResourceUsageType::kResourceRelease):
      return "kResourceRelease";
    default:
      return "Not a valid resource usage type";
  }
}

int64_t AsyncTracker::GetNumTargetDefinedResources() const { return 0; }

int64_t AsyncTracker::GetNumAvailableResources(int64_t resource_type) const {
  return 0;
}

// For now, only the target-defined resources have shareable hazard type, so
// this async tracker does not know which resources are shareable.
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetReleasedShareableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}

// For now, only the target-defined resources have shareable hazard type, so
// this async tracker does not know which resources are shareable.
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetOccupiedShareableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}

// For now, only the target-defined resources have serial hazard type, so
// this async tracker does not know which resources are serial.
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetOccupiedSerialResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}

// For now, only the target-defined resources have nonextendable hazard type, so
// this async tracker does not know which resources are nonextendable.
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetReleasedNonextendableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}

bool AsyncTracker::ReleasesSelectiveResource(const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceRelease &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
}

bool AsyncTracker::OccupiesSelectiveResource(const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceOccupy &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
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
                  return InstructionDefinesValue(instruction, value);
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
      if (buffer.value->values()[0]->shape().has_layout() &&
          buffer.value->values()[0]->shape().layout().memory_space() != 0) {
        continue;
      }
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
      if (ShouldSkipBufferAllocations(instruction, info.second,
                                      info.first.first_definition) ||
          (info.first.value->values()[0]->shape().has_layout() &&
           info.first.value->values()[0]->shape().layout().memory_space() !=
               kDefaultMemorySpace)) {
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
      if (b.value->values()[0]->shape().has_layout() &&
          b.value->values()[0]->shape().layout().memory_space() !=
              kDefaultMemorySpace) {
        continue;
      }
      if (live_buffers_[b.value->id()] != 0) {
        if (InstructionFirstDefinesBuffer(instruction, b)) {
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
      if (ShouldSkipBufferAllocations(instruction, b.second,
                                      b.first.first_definition) ||
          (b.first.value->values()[0]->shape().has_layout() &&
           b.first.value->values()[0]->shape().layout().memory_space() !=
               kDefaultMemorySpace)) {
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
      if (b.value->values()[0]->shape().has_layout() &&
          b.value->values()[0]->shape().layout().memory_space() !=
              kDefaultMemorySpace) {
        continue;
      }
      if (live_buffers_[b.value->id()]) {
        if (InstructionFirstDefinesBuffer(instruction, b)) {
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

// Find the num hops to the closest selective resource overlap in ready set that
// provided node can be scheduled in between.
int64_t GetNumHopsToClosestSelectiveOverlap(
    const DefaultSchedulerCore::ReadyQueueSet& ready_set,
    const HloGraphNode* node) {
  int64_t num_hops_to_closest_selective_resource_occupier =
      std::numeric_limits<int64_t>::max();
  for (const HloGraphNode* n : ready_set) {
    // Skip the node itself.
    if (n == node) {
      continue;
    }
    num_hops_to_closest_selective_resource_occupier =
        std::min(num_hops_to_closest_selective_resource_occupier,
                 n->GetNumHopsToClosestSelectiveResourceOccupier());
  }
  return num_hops_to_closest_selective_resource_occupier;
}

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
      DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule,
      DefaultSchedulerCore::TargetSchedulingRule early_target_scheduling_rule)
      : sched_state_(*sched_state),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule) {}
  // The comparison here implements the priority for the nodes in the ready set.
  DefaultSchedulerCore::CandidateResult operator()(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b) const {
    // Schedule according to ForceEarly.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->GetForceEarly(), a, b.node->GetForceEarly(), b,
            "kForceEarly")) {
      return *value;
    }
    // Schedule according to ForceDelay first.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            !a.node->GetForceDelay(), a, !b.node->GetForceDelay(), b,
            "kForceDelay")) {
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
        if (sched_state_.config.depth_based_memory_pressure_reduction) {
          // Try to pick a node that actually reduces memory pressure first.
          if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                  a_increase.first < 0 && a_increase.first < b_increase.first,
                  a,
                  b_increase.first < 0 && b_increase.first < a_increase.first,
                  b, "kOnlyDecreaseMemoryOverLimit")) {
            return *value;
          }
          // If there's none than prefer a node that is the deepest. That
          // matches well with unlocking pressure-reducing nodes for typical ML
          // graphs.
          if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                  a.node->GetGraphDepth() > b.node->GetGraphDepth(), a,
                  b.node->GetGraphDepth() > a.node->GetGraphDepth(), b,
                  "kDepthOverLimit")) {
            return *value;
          }
        }
        // Otherwise pick a node that increases the pressure the least from the
        // list.
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
    if (early_target_scheduling_rule_) {
      if (auto value = early_target_scheduling_rule_(a, b)) {
        return *value;
      }
    }
    // Some heuristic that try to prioritize unlocking "done" instructions
    // so that we can perform overlap. More fancy heuristics can be used by
    // discovering the closest "done" to every instruction and prioritize
    // those that are closer rather than ones that are further away.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            ShouldScheduleAsyncDone(a), a, ShouldScheduleAsyncDone(b), b,
            "kScheduleDone")) {
      return *value;
    }

    // The following rule targets the async ops using resources that should be
    // released right after the op's estimated time cost has past. It prevents
    // increasing the overlaps of such async ops more than necessary.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            PastDueCyclesForNonextendableResource(a) >
                PastDueCyclesForNonextendableResource(b),
            a,
            PastDueCyclesForNonextendableResource(b) >
                PastDueCyclesForNonextendableResource(a),
            b, "kReleaseNonextendable")) {
      return *value;
    }

    if (sched_state_.config.enable_release_start_policy) {
      // Prioritise scheduling ready "start" ops, to avoid useless extension of
      // start-done latencies. This benefits future latency ops, as ops
      // postponed here may be used to hide not-yet-scheduled latency ops.
      const ApproximateLatencyEstimator::TimeCost a_ready_interval =
          a.node->GetReadyTime() - sched_state_.current_time;
      const ApproximateLatencyEstimator::TimeCost b_ready_interval =
          b.node->GetReadyTime() - sched_state_.current_time;
      bool a_ready_and_release =
          a_ready_interval <= 0 &&
          a.node->DoesReleaseResource(ResourceType::kCollectivePermute);
      bool b_ready_and_release =
          b_ready_interval <= 0 &&
          b.node->DoesReleaseResource(ResourceType::kCollectivePermute);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_ready_and_release, a, b_ready_and_release, b,
              "kScheduleStart")) {
        return *value;
      }
      if (a_ready_and_release && b_ready_and_release) {
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_ready_interval < b_ready_interval, a,
                b_ready_interval < a_ready_interval, b, "kScheduleStart")) {
          return *value;
        }
      }
    }

    auto async_depth_0_candidate =
        [this](DefaultSchedulerCore::ScheduleCandidate& a,
               DefaultSchedulerCore::ScheduleCandidate& b)
        -> std::optional<DefaultSchedulerCore::CandidateResult> {
      // If an instruction releasing a resource is not resource constrained and
      // has an async depth of 0, delay it as much as possible to avoid
      // potential cost model inefficiencies. For example, if a pair of
      // async-start and async-done have no dependencies on other ops inside a
      // loop, the async-start will be pushed to the beginning of the loop.
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              /*first_cond=*/!(a.node->DoesReleaseAnyResource() &&
                               a.node->GetAsyncDepth() == 0 &&
                               !IsResourceConstrained(a)),
              a,
              /*second_cond=*/
              !(b.node->DoesReleaseAnyResource() &&
                b.node->GetAsyncDepth() == 0 && !IsResourceConstrained(b)),
              b, "kStartAtZeroDepth")) {
        return value;
      }
      return std::nullopt;
    };

    if (sched_state_.config.aggressive_scheduling_policies &&
        sched_state_.config.prioritize_async_depth_over_stall) {
      if (auto value = async_depth_0_candidate(a, b)) {
        return *value;
      }
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
    if (sched_state_.config.resource_serializing) {
      // Prioritize scheduling the instruction which has less serial-resource
      // conflicts with the resources in flight.
      const int64_t a_num_conflicting_resources =
          GetNumConflictingSerialResources(a);
      const int64_t b_num_conflicting_resources =
          GetNumConflictingSerialResources(b);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_num_conflicting_resources < b_num_conflicting_resources, a,
              b_num_conflicting_resources < a_num_conflicting_resources, b,
              "kLessSerialResourceConflict")) {
        return *value;
      }
    }
    if (sched_state_.config.aggressive_scheduling_policies &&
        !sched_state_.config.prioritize_async_depth_over_stall) {
      if (auto value = async_depth_0_candidate(a, b)) {
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
                !a.node->DoesReleaseAnyResource() && a_cost_diff < b_cost_diff,
                a,
                !b.node->DoesReleaseAnyResource() && b_cost_diff < a_cost_diff,
                b, "kAvoidWaste")) {
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
    // If there are no selective overlaps open currently and there will be
    // overlaps opened in the near future, hold off scheduling instructions
    // that are valuable for selective overlaps.
    if (sched_state_.config.enable_selective_resources &&
        sched_state_.selective_resource_releasers.empty()) {
      int64_t distance_to_selective_overlap_for_a =
          GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, a.node);
      int64_t distance_to_selective_overlap_for_b =
          GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, b.node);
      // If a is valuable for selective overlap and there is a selective
      // overlap in the near future a can be scheduled inside, hold off
      // scheduling a and schedule b instead. Same logic applies in reverse.
      int64_t max_distance =
          sched_state_.config.max_hops_to_closest_selective_overlap;
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              (a.node->GetValuableForSelectiveOverlap() &&
               distance_to_selective_overlap_for_a <= max_distance),
              b,
              (b.node->GetValuableForSelectiveOverlap() &&
               distance_to_selective_overlap_for_b <= max_distance),
              a, "kNotValuableForSelectiveOverlap")) {
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
    // have a better choice let's just choose the one that decreases the memory
    // pressure.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_increase.first < 0, a, b_increase.first < 0, b,
            "kDecreaseMemory")) {
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
  DefaultSchedulerCore::TargetSchedulingRule early_target_scheduling_rule_;
  DefaultSchedulerCore::OverlapLimitRule
      scheduling_instruction_crosses_overlap_limit_;

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
    return IsNopInstruction(gn.GetInstr());
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
  bool ShouldScheduleAsyncDone(
      DefaultSchedulerCore::ScheduleCandidate& gn_cand) const {
    if (!gn_cand.node->DoesOccupyAnyResource()) {
      return false;
    }
    return !ShouldDelaySendHostDone(gn_cand);
  }

  HloGraphNode::TimeCost PastDueCyclesForNonextendableResource(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (sched_state_.async_tracker
            ->GetReleasedNonextendableResourcesFromVector(
                cand.node->GetResources())
            .empty()) {
      return 0.0;
    }
    return std::max(sched_state_.current_time - cand.node->GetReadyTime(), 0.0);
  }
  bool ShouldDelaySendHostDone(
      DefaultSchedulerCore::ScheduleCandidate& gn_cand) const {
    const HloGraphNode& gn = *gn_cand.node;
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
    if (!gn_cand.estimated_connected_send_ready_time.has_value()) {
      HloGraphNode::TimeCost start_ready_time = 0;
      for (const auto& succ : start.GetSuccessors()) {
        // If any successor is not ready skip this logic. We detect this by
        // checking that ready time is set to max. This should never happen
        // because sends always have 1 or 2 successors that should be scheduled
        // or ready already, but in case somebody comes up with different
        // patterns lets keep this check here.
        if (succ.Target().GetReadyTime() >=
            std::numeric_limits<HloGraphNode::TimeCost>::max()) {
          return false;
        }
        start_ready_time = std::max(
            start_ready_time, succ.Latency() + succ.Target().GetReadyTime());
      }
      gn_cand.estimated_connected_send_ready_time = start_ready_time;
    }
    if (*gn_cand.estimated_connected_send_ready_time -
            sched_state_.current_time <=
        latency) {
      return false;
    }
    return true;
  }
  // Compute and cache memory pressure change computation for candidate.
  std::pair<int64_t, int64_t> GetMemoryPressureChanges(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (cand.pressure_change) {
      return *cand.pressure_change;
    }
    std::optional<std::pair<int64_t, int64_t>> start_result;
    // In case of async-done instruction they can increase the memory pressure
    // but its always a possible move to schedule the start immediately after,
    // so for memory pressure purpose in the scheduling heuristic actually use
    // the memory pressure change of the start rather than the -done.
    if (this->sched_state_.async_tracker->IsSupportedAsyncDone(
            cand.node->GetInstr())) {
      const HloInstruction* start = cand.node->GetInstr().operand_count() > 0
                                        ? cand.node->GetInstr().operand(0)
                                        : nullptr;
      if (start != nullptr &&
          this->sched_state_.async_tracker->IsSupportedAsyncStart(*start)) {
        start_result =
            sched_state_.memory_pressure_tracker->MemoryPressureDifference(
                start);
      }
    }
    cand.pressure_change =
        sched_state_.memory_pressure_tracker->MemoryPressureDifference(
            &cand.node->GetInstr());
    if (start_result.has_value()) {
      cand.pressure_change->first =
          std::min(start_result->first, cand.pressure_change->first);
      cand.pressure_change->second =
          std::max(start_result->second, cand.pressure_change->second);
    }
    return *cand.pressure_change;
  }
  int64_t GetNumConflictingSerialResources(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    auto resources =
        sched_state_.async_tracker->GetOccupiedSerialResourcesFromVector(
            cand.node->GetResources());
    int64_t num_conflicting_resources = 0;
    for (int64_t resource : resources) {
      if (!sched_state_.resource_occupiers_in_flight.count(resource)) continue;
      num_conflicting_resources +=
          sched_state_.resource_occupiers_in_flight.at(resource).size();
    }
    return num_conflicting_resources;
  }
};

enum SkipNodeReason {
  kShouldSkipNodeFunction,
  kExceedsOverlapLimit,
  kAnnotationGroupNotReady,
};

absl::string_view SkipNodeReasonString(SkipNodeReason reason) {
  switch (reason) {
    case SkipNodeReason::kShouldSkipNodeFunction:
      return "Skipped due to kShouldSkipNodeFunction.";
    case SkipNodeReason::kExceedsOverlapLimit:
      return "Skipped due to kExceedsOverlapLimit.";
    case SkipNodeReason::kAnnotationGroupNotReady:
      return "Skipped due to kAnnotationNotReady.";
  }
}

}  // namespace

// Helper function to find the best node from the queues of scheduling state for
// scheduling.
absl::StatusOr<HloGraphNode*>
DefaultSchedulerCore::FindAndExtractBestNodeAvailable(
    DefaultSchedulerCore::SchedulingState& sched_state,
    DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node) {
  // Schedule a nop instruction if available.
  if (!sched_state.nop_set.empty()) {
    HloGraphNode* node = sched_state.nop_set.back();
    sched_state.nop_set.pop_back();
    return node;
  }
  absl::InlinedVector<std::pair<HloGraphNode*, SkipNodeReason>, 2>
      skipped_nodes_and_reasons;
  VLOG(2) << "Current time: " << sched_state.current_time;
  ReadySetLt ready_lt{&sched_state, target_scheduling_rule_,
                      early_target_scheduling_rule_};
  // Construct a schedule candidate for caching.
  ScheduleCandidate ready_chosen;
  auto chosen_it = sched_state.ready_set.end();
  // Try to pick nodes from the ready set first as are the ones that cause the
  // most latency hiding.
  for (auto ready_node_it = sched_state.ready_set.begin(),
            e = sched_state.ready_set.end();
       ready_node_it != e; ++ready_node_it) {
    if (should_skip_node && should_skip_node(*ready_node_it)) {
      if (ready_chosen.node == nullptr) {
        skipped_nodes_and_reasons.push_back(
            {*ready_node_it, SkipNodeReason::kShouldSkipNodeFunction});
      }
      continue;
    }
    // These ifs will be true when the iterator points to an annotated node, but
    // the chosen node is nullptr because the annotation group is not ready to
    // be scheduled yet (because of the annotation roots' successors not being
    // scheduled yet). So we skip this node and continue to the next one.
    if ((*ready_node_it)->GetAnnotation() != -1) {
      if (ready_chosen.node == nullptr) {
        skipped_nodes_and_reasons.push_back(
            {*ready_node_it, SkipNodeReason::kAnnotationGroupNotReady});
      }
      continue;
    }
    // If this node would cause the max_concurrent_resource count to go beyond
    // the limit do not schedule it and pass to the next node.
    if (scheduling_instruction_crosses_overlap_limit_(sched_state,
                                                      *ready_node_it)) {
      if (ready_chosen.node == nullptr) {
        skipped_nodes_and_reasons.push_back(
            {*ready_node_it, SkipNodeReason::kExceedsOverlapLimit});
      }
      continue;
    }
    ScheduleCandidate ready_candidate =
        InitializeCandidate(*ready_node_it, sched_state);
    if (ready_chosen.node == nullptr) {
      ready_chosen = ready_candidate;
      chosen_it = ready_node_it;
      VLOG(2) << "Choosing from ready (" << ready_chosen.node->GetInstr().name()
              << ") Reason: First Candidate";
      continue;
    }
    // Compare the current candidate with the previous candidate.
    CandidateResult cand_result = ready_lt(ready_candidate, ready_chosen);
    const bool new_candidate_selected =
        cand_result.result.node == *ready_node_it;
    auto print_pressure_change =
        [](const std::optional<std::pair<int64_t, int64_t>>& p) {
          if (p.has_value()) {
            return std::to_string(p.value().first);
          }
          return std::string("N/A");
        };
    VLOG(2) << "Choosing from ready ("
            << (new_candidate_selected ? ready_candidate.node->GetInstr().name()
                                       : ready_chosen.node->GetInstr().name())
            << ") vs ("
            << (new_candidate_selected
                    ? ready_chosen.node->GetInstr().name()
                    : ready_candidate.node->GetInstr().name())
            << ") Reason: " << cand_result.reason << " mem pressure chosen "
            << print_pressure_change(
                   (new_candidate_selected ? ready_candidate : ready_chosen)
                       .pressure_change)
            << " mem pressure other "
            << print_pressure_change(
                   (new_candidate_selected ? ready_chosen : ready_candidate)
                       .pressure_change);
    if (new_candidate_selected) {
      ready_chosen = cand_result.result;
      chosen_it = ready_node_it;
    }
  }
  if (ready_chosen.node == nullptr) {
    if (!sched_state.ready_annotations.empty()) {
      std::string error_message = absl::StrCat(
          "There is a scheduling group which exceeds the overlap limits. "
          "Annotation id: ",
          sched_state.ready_annotations.front(), ". ");
      absl::flat_hash_map<int64_t, int64_t> num_resources_needed =
          GetNumResourcesNeededForAnnotation(
              sched_state, sched_state.ready_annotations.front());
      for (const auto& [resource, num_needed] : num_resources_needed) {
        int64_t limit = sched_state.max_concurrent_resource.at(resource);
        if (num_needed > limit) {
          absl::StrAppend(&error_message, "It needs ", num_needed, " ",
                          sched_state.async_tracker->GetResourceName(resource),
                          " resources, but the limit is ", limit, ". ");
        }
      }
      return absl::InternalError(error_message);
    }
    return absl::InternalError(absl::StrCat(
        "FindAndExtractBestNodeAvailable failed to find a node to "
        "schedule, skipped nodes: ",
        absl::StrJoin(skipped_nodes_and_reasons, "; ",
                      [](std::string* out, const auto& pair) {
                        absl::StrAppend(out, pair.first->GetInstr().name(),
                                        ": ",
                                        SkipNodeReasonString(pair.second));
                      })));
  }
  CHECK(chosen_it != sched_state.ready_set.end());
  std::swap(*chosen_it, sched_state.ready_set.back());
  sched_state.ready_set.pop_back();
  return ready_chosen.node;
}

void DefaultSchedulerCore::LogInstruction(const HloInstruction* instr) const {
  VLOG(5) << instr->ToString();
}

void PrintOccupierList(
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  for (int64_t i = 0; i < occupiers.size(); i++) {
    VLOG(3) << "\tOccupier " << i << ": "
            << occupiers[i].first->Target().GetInstr().name()
            << ", projected finish time: " << occupiers[i].second
            << " original latency: " << occupiers[i].first->OriginalLatency()
            << " latency: " << occupiers[i].first->Latency();
  }
}

bool DefaultSchedulerCore::DeleteOccupierFromResource(
    HloGraphNode::TimeCost current_time, HloEdge& edge,
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  // If this edge does not exist in the list, return false
  if (absl::c_any_of(
          occupiers,
          [&edge](const std::pair<HloEdge*, HloGraphNode::TimeCost>& element) {
            return element.first == &edge;
          }) == false) {
    return false;
  }
  std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>::iterator it =
      occupiers.begin();
  int64_t num_occupiers = occupiers.size();
  HloGraphNode::TimeCost prev_time = current_time;
  HloGraphNode::TimeCost accumulated_saved_time = 0;
  while (it != occupiers.end() && it->first != &edge) {
    if (it->second <= current_time) {
      num_occupiers--;
      it++;
      continue;
    }
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    prev_time = it->second;
    CHECK_GT(num_occupiers, 0);
    HloGraphNode::TimeCost current_saved_time =
        remaining_time_of_edge / num_occupiers;
    accumulated_saved_time += current_saved_time;
    CHECK_GE(it->second, accumulated_saved_time);
    it->second -= accumulated_saved_time;
    num_occupiers--;
    it++;
  }
  CHECK(it != occupiers.end());  // The edge has to exist
  // If the edge has not finished yet, shorten the remaining pfts
  if (it->second > current_time) {
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    HloGraphNode::TimeCost current_saved_time =
        remaining_time_of_edge / num_occupiers;
    accumulated_saved_time += current_saved_time;
  }
  it = occupiers.erase(it);
  for (; it != occupiers.end(); it++) {
    it->second -= accumulated_saved_time;
  }
  return true;
}

// This function assumes the existing occupiers' latencies are already adjusted
// and sorted by their projected finish time. WARNING: Do not add an edge with a
// current time smaller than the current times when the existing edges were
// inserted.
bool DefaultSchedulerCore::AddOccupierToResource(
    HloGraphNode::TimeCost current_time, HloEdge& new_edge,
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  CHECK(new_edge.OriginalLatency() > 0 && current_time >= 0);
  auto new_edge_remaining = new_edge.OriginalLatency();
  std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>::iterator it =
      occupiers.begin();
  int64_t num_occupiers = occupiers.size();
  HloGraphNode::TimeCost prev_time = current_time;
  HloGraphNode::TimeCost accumulated_delay = 0;
  while (it != occupiers.end() &&
         it->second - prev_time <= new_edge_remaining * num_occupiers) {
    // This edge has already finished, so it shouldn't affect the delays.
    if (it->second <= current_time) {
      num_occupiers--;
      it++;
      continue;
    }
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    prev_time = it->second;
    CHECK_GT(num_occupiers, 0);
    HloGraphNode::TimeCost current_delay =
        remaining_time_of_edge / num_occupiers;
    new_edge_remaining -= current_delay;
    accumulated_delay += current_delay;
    it->second += accumulated_delay;
    num_occupiers--;
    it++;
  }
  // Add the new edge
  num_occupiers++;
  HloGraphNode::TimeCost adjusted_remaining_time =
      new_edge_remaining * num_occupiers;
  it = occupiers.insert(
      it, std::make_pair(&new_edge, prev_time + accumulated_delay +
                                        adjusted_remaining_time));
  // Since it points to the newly inserted element, increment it
  it++;
  accumulated_delay += new_edge_remaining;
  CHECK(new_edge.OriginalLatency() - 0.0001 < accumulated_delay &&
        accumulated_delay < new_edge.OriginalLatency() + 0.0001);
  for (; it != occupiers.end(); it++) {
    it->second += accumulated_delay;
  }

  // Update the ready time of the occupiers.
  for (auto it = occupiers.begin(); it != occupiers.end(); it++) {
    HloGraphNode* done_node = it->first->TargetPtr();
    if (done_node == nullptr) {
      continue;
    }
    if (done_node->GetReadyTime() < it->second) {
      for (HloEdge& start_edge : done_node->GetPredecessors()) {
        if (start_edge.Target().GetReadyTime() < it->second) {
          start_edge.Target().SetReadyTime(it->second);
        }
      }
    }
  }
  return true;
}

// Comparator for the annotated ready set. This class represents the priority
// policies for the nodes in the annotated ready set. The policies are currently
// very minimal (recall that the scheduling is done in the reverse order):
//  1. Async done nodes are scheduled before any other nodes.
//  2. Among other nodes, async start nodes are scheduled after other nodes.
class AnnotationReadySetLt {
 public:
  explicit AnnotationReadySetLt() = default;
  // Implements the priority for the nodes in the annotated ready set.
  DefaultSchedulerCore::CandidateResult operator()(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b) const {
    // Schedule an async done.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->DoesOccupyAnyResource(), a, b.node->DoesOccupyAnyResource(),
            b, "kAnnotatedAsyncDone")) {
      return *value;
    }
    // Schedule anything but an async start.
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->DoesReleaseAnyResource(), b,
            b.node->DoesReleaseAnyResource(), a, "kAnnotatedNotAsyncStart")) {
      return *value;
    }
    return {a, "kAnnotatedNoReason"};
  }
};
absl::StatusOr<HloGraphNode*> FindAndExtractBestAnnotatedNode(
    DefaultSchedulerCore::SchedulingState& sched_state) {
  using ScheduleCandidate = DefaultSchedulerCore::ScheduleCandidate;
  using CandidateResult = DefaultSchedulerCore::CandidateResult;
  AnnotationReadySetLt ready_lt;
  // Construct a schedule candidate for caching.
  ScheduleCandidate ready_chosen;
  auto& annotation_ready = sched_state.annotation_ready;
  auto chosen_it = annotation_ready.end();
  // Try to pick nodes from the ready set first as are the ones that cause the
  // most latency hiding.
  for (auto ready_node_it = annotation_ready.begin(),
            e = annotation_ready.end();
       ready_node_it != e; ++ready_node_it) {
    ScheduleCandidate ready_candidate;
    ready_candidate.node = *ready_node_it;
    if (ready_chosen.node == nullptr) {
      ready_chosen = ready_candidate;
      chosen_it = ready_node_it;
      VLOG(2) << "Choosing from ready (" << ready_chosen.node->GetInstr().name()
              << ") Reason: First Candidate";
      continue;
    }
    // Compare the current candidate with the previous candidate.
    CandidateResult cand_result = ready_lt(ready_candidate, ready_chosen);
    const bool new_candidate_selected =
        cand_result.result.node == *ready_node_it;
    auto print_pressure_change =
        [](const std::optional<std::pair<int64_t, int64_t>>& p) {
          if (p.has_value()) {
            return std::to_string(p.value().first);
          }
          return std::string("N/A");
        };
    VLOG(2) << "Choosing from ready ("
            << (new_candidate_selected ? ready_candidate.node->GetInstr().name()
                                       : ready_chosen.node->GetInstr().name())
            << ") vs ("
            << (new_candidate_selected
                    ? ready_chosen.node->GetInstr().name()
                    : ready_candidate.node->GetInstr().name())
            << ") Reason: " << cand_result.reason << " mem pressure chosen "
            << print_pressure_change(
                   (new_candidate_selected ? ready_candidate : ready_chosen)
                       .pressure_change)
            << " mem pressure other "
            << print_pressure_change(
                   (new_candidate_selected ? ready_chosen : ready_candidate)
                       .pressure_change);
    if (new_candidate_selected) {
      ready_chosen = cand_result.result;
      chosen_it = ready_node_it;
    }
  }
  // Delete the node from the annotation ready set.
  std::swap(*chosen_it, annotation_ready.back());
  annotation_ready.pop_back();
  return ready_chosen.node;
}

absl::Status DefaultSchedulerCore::ScheduleAnnotation(
    const HloComputation* computation, int64_t annotation,
    DefaultSchedulerCore::SchedulingState* sched_state) const {
  // Filter the ready nodes with the annotation.
  TF_RET_CHECK(sched_state->annotation_ready.empty());
  for (HloGraphNode* node : sched_state->ready_set) {
    if (node->GetAnnotation() == annotation) {
      sched_state->annotation_ready.push_back(node);
    }
  }
  int64_t num_scheduled = 0;
  int64_t annotation_size =
      annotation_tracker_->GetNumInstructions(computation, annotation);
  while (!sched_state->annotation_ready.empty()) {
    // Print the current annotation ready queue.
    VLOG(2) << "Current annotation ready queue:";
    XLA_VLOG_LINES(2, [&]() {
      struct LogFormatter {
        void operator()(std::string* out, const HloGraphNode* n) const {
          absl::StrAppend(out, "\t", n->GetInstr().name(),
                          " Ready time: ", n->GetReadyTime());
        }
      };
      return absl::StrJoin(sched_state->annotation_ready, "\n", LogFormatter());
    }());
    VLOG(2) << "Current time: " << sched_state->current_time;
    // Find the best annotated node to schedule.
    TF_ASSIGN_OR_RETURN(HloGraphNode * node,
                        FindAndExtractBestAnnotatedNode(*sched_state));

    TF_RET_CHECK(node != nullptr)
        << "Couldn't find an annotated node to schedule.";
    // Delete the node from the ready set.
    auto node_it = std::find(sched_state->ready_set.begin(),
                             sched_state->ready_set.end(), node);
    TF_RET_CHECK(node_it != sched_state->ready_set.end())
        << "Couldn't find the annotated node in ready set: "
        << node->GetInstr().name();
    std::swap(*node_it, sched_state->ready_set.back());
    sched_state->ready_set.pop_back();

    // Schedule the node.
    TF_ASSIGN_OR_RETURN(sched_state->current_time,
                        ScheduleNode(node, sched_state));
    num_scheduled++;
    VLOG(2) << "Scheduled annotated node (" << num_scheduled << "/"
            << annotation_size << "): " << node->GetInstr().name();
  }
  // Check that we scheduled all the nodes in the annotation.
  TF_RET_CHECK(num_scheduled == annotation_size)
      << "Couldn't schedule all annotated nodes in one go.";
  return absl::OkStatus();
}

// Returns the vector of annotations that the given node is a successor of, but
// is not included in that annotation group itself.
std::vector<int64_t> GetPredecessorAnnotations(const HloGraphNode* node) {
  int64_t cur_annotation = node->GetAnnotation();
  std::vector<int64_t> predecessor_annotations;
  absl::flat_hash_set<int64_t> seen_annotations;
  for (const HloEdge& edge : node->GetPredecessors()) {
    int64_t pred_annotation = edge.Target().GetAnnotation();
    if (pred_annotation != cur_annotation &&
        seen_annotations.insert(pred_annotation).second) {
      predecessor_annotations.push_back(pred_annotation);
    }
  }
  return predecessor_annotations;
}

absl::StatusOr<HloGraphNode::TimeCost> DefaultSchedulerCore::ScheduleNode(
    HloGraphNode* n, DefaultSchedulerCore::SchedulingState* sched_state) const {
  // Insert the node into the sequence and mark it as scheduled.
  sched_state->new_sequence_reversed.push_back(
      const_cast<HloInstruction*>(&n->GetInstr()));
  n->SetScheduled();

  // If this node was a successor to one or more scheduling groups, update the
  // number of scheduled successors for each of those groups and add the group
  // the ready_annotations set if all of its successors have been scheduled.
  std::vector<int64_t> annotations = GetPredecessorAnnotations(n);
  if (!annotations.empty()) {
    VLOG(2) << "Scheduled node is a frontier: " << n->GetInstr().name();
    for (int64_t annotation : annotations) {
      sched_state->num_scheduled_successors_for_annotation[annotation]++;
      VLOG(2)
          << "Annotation: " << annotation << " scheduled num successors: "
          << sched_state->num_scheduled_successors_for_annotation[annotation]
          << " total num successors: "
          << annotation_tracker_->GetNumSuccessors(n->GetInstr().parent(),
                                                   annotation);
      // LegalizeSchedulingAnnotations pass should have made sure that we will
      // eventually reach a state where all successors of the annotation are
      // scheduled.
      if (annotation_tracker_->GetNumSuccessors(n->GetInstr().parent(),
                                                annotation) ==
          sched_state->num_scheduled_successors_for_annotation[annotation]) {
        sched_state->ready_annotations.push_back(annotation);
      }
    }
  }
  // Remove scheduled node from selective_resources_releasers if it
  // was there.
  if (sched_state->config.enable_selective_resources &&
      n->ReleasesSelectiveResource()) {
    auto it = std::find(sched_state->selective_resource_releasers.begin(),
                        sched_state->selective_resource_releasers.end(), n);
    // Perform sanity check node was in selective_resources_releasers.
    if (it == sched_state->selective_resource_releasers.end()) {
      LOG(WARNING) << "Selective resource releasers list does not contain node "
                      "that releases a selective resource: "
                   << n->ToString();
    } else {
      sched_state->selective_resource_releasers.erase(it);
    }
  }

  // If scheduled node cannot overlap with nodes that hold selective resources,
  // we increment the ready time of all nodes that release a selective resource
  // with the cost of the scheduled node.
  if (sched_state->config.enable_selective_resources &&
      !n->GetValuableForSelectiveOverlap()) {
    for (HloGraphNode* node : sched_state->selective_resource_releasers) {
      node->SetReadyTime(node->GetReadyTime() + n->GetCost());
    }
  }

  // If this node is an async start/done handle the increase/decrease the number
  // of outstanding async ops.
  for (auto& resource : n->GetResources()) {
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
    schedule_time = std::max(schedule_time, time_from_edge);
    if (sched_state->config.resource_sharing) {
      // Adjust the ready time if this edge uses shareable resources
      auto occupied_resources = n->GetShareableResourcesOnEdge(pred);
      for (const int64_t resource : occupied_resources) {
        auto occupiers = sched_state->shareable_resource_occupiers[resource];
        for (auto [occupier_edge, edge_pft] : occupiers) {
          if (occupier_edge == &pred) {
            VLOG(3) << "Ready time of scheduled node " << n->GetInstr().name()
                    << " before update with pft: " << edge_pft
                    << ", ready_time: " << schedule_time;
            schedule_time = std::max(schedule_time, edge_pft);
            VLOG(3) << "Ready time of scheduled node " << n->GetInstr().name()
                    << " after update with pft: " << edge_pft
                    << ", ready_time: " << schedule_time;
          }
        }
      }
    }
  }
  // Set the ready time to the scheduled time for scheduled nodes.
  n->SetReadyTime(schedule_time);
  HloGraphNode::TimeCost current_time = schedule_time + n->GetCost();
  if (sched_state->config.resource_sharing) {
    // If a shareable resource is released by scheduling this node, delete the
    // corresponding edge from the respective occupier(s) list.
    for (HloEdge& edge : n->GetSuccessors()) {
      auto released_resources = n->GetShareableResourcesOnEdge(edge);
      for (const int64_t resource : released_resources) {
        CHECK(DeleteOccupierFromResource(
            schedule_time, edge,
            sched_state->shareable_resource_occupiers[resource]));
        if (VLOG_IS_ON(2)) {
          VLOG(3) << "Occupier list for "
                  << sched_state->async_tracker->GetResourceName(resource)
                  << ": ";
          PrintOccupierList(
              sched_state->shareable_resource_occupiers[resource]);
        }
      }
    }
    // If a shareable resource is occupied by scheduling this node, insert the
    // corresponding edge to the respective occupier(s) list.
    for (HloEdge& edge : n->GetPredecessors()) {
      for (HloEdge& inverse_edge : edge.Target().GetSuccessors()) {
        if (&(inverse_edge.Target()) == n) {
          auto occupied_resources =
              edge.Target().GetShareableResourcesOnEdge(inverse_edge);
          for (const int64_t resource : occupied_resources) {
            VLOG(3) << "Adding edge from" << edge.Target().GetInstr().name()
                    << " to " << inverse_edge.Target().GetInstr().name()
                    << " for resource"
                    << sched_state->async_tracker->GetResourceName(resource);
            CHECK(AddOccupierToResource(
                current_time, inverse_edge,
                sched_state->shareable_resource_occupiers[resource]));
            if (VLOG_IS_ON(2)) {
              VLOG(3) << "Occupier list for "
                      << sched_state->async_tracker->GetResourceName(resource)
                      << ": ";
              PrintOccupierList(
                  sched_state->shareable_resource_occupiers[resource]);
            }
          }
          break;
        }
      }
    }
  }
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
      ready_time = std::max(ready_time, edge_time);
      if (sched_state->config.resource_sharing) {
        // Adjust the ready time if this edge uses shareable resources
        auto occupied_resources =
            edge.Target().GetShareableResourcesOnEdge(pred);
        for (const int64_t resource : occupied_resources) {
          auto occupiers = sched_state->shareable_resource_occupiers[resource];
          for (auto [occupier_edge, edge_pft] : occupiers) {
            if (occupier_edge == &pred) {
              VLOG(3) << "Ready time of predecessor "
                      << edge.Target().GetInstr().name()
                      << " before update with pft: " << edge_pft
                      << ", ready_time: " << ready_time;
              ready_time = std::max(ready_time, edge_pft);
              VLOG(3) << "Ready time of predecessor "
                      << edge.Target().GetInstr().name()
                      << " after update with pft: " << edge_pft
                      << ", ready_time: " << ready_time;
            }
          }
        }
      }
    }
    for (auto& resource : edge.Target().GetResources()) {
      if (resource.second == ResourceUsageType::kResourceOccupy) {
        ++(sched_state->resource_users_in_queue[resource.first]);
      }
    }
    edge.Target().SetReadyTime(ready_time);
    int64_t annotation = edge.Target().GetAnnotation();
    // We are adding the no-op instructions to a separate set so that we can
    // immediately schedule them when they are ready.
    if (IsNopInstruction(edge.Target().GetInstr()) && annotation == -1) {
      sched_state->nop_set.push_back(&edge.Target());
      continue;
    }
    sched_state->ready_set.push_back(&edge.Target());
    if (annotation != -1 && annotation == sched_state->ongoing_annotation) {
      sched_state->annotation_ready.push_back(&edge.Target());
    }
    if (edge.Target().GetReadyTime() > current_time) {
      sched_state->next_ready_stack.push_back(&edge.Target());
      std::push_heap(sched_state->next_ready_stack.begin(),
                     sched_state->next_ready_stack.end(), ready_time_cmp);
    }

    // If the node we added to ready set releases a selective resource, add
    // it to selective_resources_releasers.
    if (sched_state->config.enable_selective_resources &&
        edge.Target().ReleasesSelectiveResource()) {
      sched_state->selective_resource_releasers.push_back(&edge.Target());
    }
  }
  ++sched_state->scheduled_count;
  for (auto& resource : n->GetResources()) {
    if (resource.second == ResourceUsageType::kResourceRelease) {
      // Some recv-dones exist without a corresponding recv op in the same
      // computation. In this case, we cannot find the corresponding start op
      // and thus cannot erase the start op from the map.
      if (sched_state->resource_occupiers_in_flight.contains(resource.first)) {
        sched_state->resource_occupiers_in_flight.at(resource.first)
            .erase(&n->GetInstr());
      }
    } else if (resource.second == ResourceUsageType::kResourceOccupy) {
      // For supported async collective done ops, save their corresponding start
      // ops in the map
      if (async_tracker_->IsSupportedAsyncDone(n->GetInstr()) &&
          async_tracker_->IsSupportedAsyncStart(*n->GetInstr().operand(0))) {
        sched_state->resource_occupiers_in_flight[resource.first].insert(
            n->GetInstr().operand(0));
      } else {
        sched_state->resource_occupiers_in_flight[resource.first].insert(
            &n->GetInstr());
      }
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

bool HloScheduleGraph::IsPredecessorTransitively(
    const HloGraphNode* node, const HloGraphNode* possible_predecessor) {
  absl::flat_hash_set<const HloGraphNode*> visited = {possible_predecessor};
  std::vector<const HloGraphNode*> to_visit_queue = {node};
  while (!to_visit_queue.empty()) {
    const HloGraphNode* curr = to_visit_queue.back();
    to_visit_queue.pop_back();
    if (curr == possible_predecessor) {
      return true;
    }
    if (visited.contains(curr)) {
      continue;
    }
    visited.insert(curr);
    for (const auto& edge : curr->GetPredecessors()) {
      auto user_node_it = nodes_.find(&edge.Target().GetInstr());
      to_visit_queue.push_back(user_node_it->second.get());
    }
  }
  return false;
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
  std::vector<const HloInstruction*> while_instrs;
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
    auto resources = async_tracker->GetResourcesFromInstruction(*instr);
    new_node_it->second->resources_ =
        ResourcesVector(resources.begin(), resources.end());
    new_node_it->second->released_shareable_resources_ =
        async_tracker->GetReleasedShareableResourcesFromVector(
            new_node_it->second->GetResources());
    new_node_it->second->occupied_shareable_resources_ =
        async_tracker->GetOccupiedShareableResourcesFromVector(
            new_node_it->second->GetResources());
    new_node_it->second->releases_selective_resource_ =
        async_tracker->ReleasesSelectiveResource(new_node_it->second.get());
    new_node_it->second->occupies_selective_resource_ =
        async_tracker->OccupiesSelectiveResource(new_node_it->second.get());
    // Gather while instructions for subsequent send-done dependency checks.
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instrs.push_back(instr);
    }
  }
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
      HloGraphNode::AddDependency(instr_node, user_node, latency_estimator);
    }
    for (const HloInstruction* ctrl_succ : instr->control_successors()) {
      VLOG(10) << "\tCtrl Successor: " << ctrl_succ->ToString();
      auto ctrl_succ_node_it = nodes_.find(ctrl_succ);
      CHECK(ctrl_succ_node_it != nodes_.end());
      HloGraphNode* ctrl_succ_node = ctrl_succ_node_it->second.get();
      HloGraphNode::AddDependency(instr_node, ctrl_succ_node,
                                  latency_estimator);
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
                if (use.instruction == async_start ||
                    reachability->IsReachable(instr, use.instruction)) {
                  continue;
                }
                auto it = nodes_.find(use.instruction);
                CHECK(it != nodes_.end());
                HloGraphNode* pred_node = it->second.get();
                it = nodes_.find(async_start);
                CHECK(it != nodes_.end());
                HloGraphNode* start_node = it->second.get();
                // Ignore token operands as they are not real aliasing.
                if (use.instruction->operand(use.operand_number)
                        ->shape()
                        .IsToken()) {
                  continue;
                }
                // If there is already a transitive link between the nodes the
                // other way then skip adding this one.
                if (IsPredecessorTransitively(pred_node, start_node)) {
                  continue;
                }
                HloGraphNode::AddDependency(pred_node, start_node, 1);
              }
            }
          }
        }
      }
    }
    // Add dependent edges from send-done operations to while loops which are
    // dependent on the recv-done control predecessor of the send-done.
    // This prevents send-done operations from being scheduled after dependent
    // while loops, which can caused send/recv overlap limits to be violated.
    //
    // Example HLO sequence:
    //
    //   %0 = recv-done --->
    //                     |
    //   %1 = send-done <--|
    //   %2 = while <------|
    //
    if (instr->opcode() == HloOpcode::kSendDone) {
      for (const auto* ctrl_pred : instr->control_predecessors()) {
        if (ctrl_pred->opcode() != HloOpcode::kRecvDone) {
          continue;
        }
        const HloInstruction* dependent_while_instr = nullptr;
        for (const auto* while_hlo : while_instrs) {
          if (!reachability->IsReachable(ctrl_pred, while_hlo)) {
            continue;
          }
          if (dependent_while_instr == nullptr) {
            dependent_while_instr = while_hlo;
            continue;
          }
          if (OriginalInstructionPosition(while_hlo) <
              OriginalInstructionPosition(dependent_while_instr)) {
            dependent_while_instr = while_hlo;
          }
        }
        // Add dependency edge from 'instr' to 'dependent_while_instr'.
        if (dependent_while_instr != nullptr) {
          auto send_done_it = nodes_.find(instr);
          CHECK(send_done_it != nodes_.end());
          HloGraphNode* send_done_node = send_done_it->second.get();
          auto while_it = nodes_.find(dependent_while_instr);
          CHECK(while_it != nodes_.end());
          HloGraphNode* while_node = while_it->second.get();
          HloGraphNode::AddDependency(send_done_node, while_node, 1);
        }
        break;
      }
    }
  }
}

std::string HloScheduleGraph::ToString(
    const AsyncTracker* async_tracker) const {
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
    absl::StrAppend(&result, (*it)->ToString(async_tracker));
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

std::vector<HloGraphNode*> HloScheduleGraph::FindTopRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetIndegree() == 0) {
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
    node.SetDepth(0.0);
    node.SetGraphDepth(0);
    if (node.GetIndegree() == 0) {
      stack.push_back(&node);
    }
  }
  while (!stack.empty()) {
    auto* node = stack.back();
    stack.pop_back();
    // If a node occupies a selective resource, it is the closest selective
    // resource occupier to itself and is 0 hops away. Otherwise, the num hops
    // to closest selective resource occupier is the minimum of that of all
    // predecessors plus 1.
    if (async_tracker->OccupiesSelectiveResource(node)) {
      node->num_hops_to_closest_selective_resource_occupier_ = 0;
    } else {
      int64_t closest_predecessor_distance =
          std::numeric_limits<int64_t>::max();
      for (auto& pred : node->GetPredecessors()) {
        closest_predecessor_distance = std::min(
            closest_predecessor_distance,
            pred.Target().num_hops_to_closest_selective_resource_occupier_);
      }
      if (closest_predecessor_distance != std::numeric_limits<int64_t>::max()) {
        node->num_hops_to_closest_selective_resource_occupier_ =
            closest_predecessor_distance + 1;
      }
    }
    if (async_tracker->IsSupportedAsyncDone(node->GetInstr())) {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth() + pred.Latency(),
                     node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    } else {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth(), node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    }
    for (auto& succ : node->GetSuccessors()) {
      if (--current_rank[&succ.Target()] == 0) {
        stack.push_back(&succ.Target());
      }
    }
  }
}
void HloScheduleGraph::AnnotateGraph(
    const AnnotationTracker* annotation_tracker) {
  const HloComputation* comp = original_order_[0]->parent();
  for (int64_t annotation : annotation_tracker->GetAnnotations(comp)) {
    for (const HloInstruction* instr :
         annotation_tracker->GetInstructions(comp, annotation)) {
      HloGraphNode& node = GetNode(instr);
      TF_CHECK_OK(node.SetAnnotation(annotation));
    }
  }
}

absl::Status DefaultSchedulerCore::InitializeScheduler(
    const HloModule* module) {
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
  module_pressure_state_ = std::make_unique<ModulePressureState>(
      module, alias_analysis_.get(), shape_size_bytes_);
  module_pressure_state_->InitializePressureStates();
  module_pressure_state_->SetMemoryPeak(0);
  annotation_tracker_ = std::make_unique<AnnotationTracker>(module);
  if (VLOG_IS_ON(2)) {
    annotation_tracker_->PrintAnnotationSets(2);
  }
  if (!scheduling_instruction_crosses_overlap_limit_) {
    scheduling_instruction_crosses_overlap_limit_ =
        [](const SchedulingState& sched_state, const HloGraphNode* node) {
          for (const auto& [resource, limit] :
               sched_state.max_concurrent_resource) {
            // No resources in flight of this kind. Continue.
            auto it = sched_state.resource_occupiers_in_flight.find(resource);
            if (it == sched_state.resource_occupiers_in_flight.end() ||
                it->second.empty()) {
              continue;
            }
            // Number of instances of 'resource' needed if this instruction was
            // to be scheduled.
            const int64_t num_resources_needed =
                sched_state.async_tracker->GetNumResourcesPerInstruction(
                    resource, node->GetInstr());
            if (limit < num_resources_needed) {
              return true;
            }
          }
          return false;
        };
  }
  return absl::OkStatus();
}

absl::Status DefaultSchedulerCore::SchedulingStep(
    SchedulingState* sched_state) {
  // Get the first available node for scheduling that is the node that
  // satisfies our ready heuristic the best.
  TF_ASSIGN_OR_RETURN(HloGraphNode * node,
                      FindAndExtractBestNodeAvailable(
                          *sched_state, /*should_skip_node=*/nullptr));
  CHECK(node != nullptr);
  TF_ASSIGN_OR_RETURN(sched_state->current_time,
                      ScheduleNode(node, sched_state));
  VLOG(2) << "Scheduled: " << node->GetInstr().name();
  XLA_VLOG_LINES(5, node->ToString());
  return absl::OkStatus();
}

absl::flat_hash_map<int64_t, int64_t>
DefaultSchedulerCore::GetNumResourcesNeededForAnnotation(
    const SchedulingState& sched_state, int64_t annotation) {
  absl::flat_hash_map<int64_t, int64_t> num_resources_needed;
  const HloComputation* comp =
      sched_state.sched_graph.GetOriginalInstrList()[0]->parent();
  for (const HloInstruction* instr :
       annotation_tracker_->GetInstructions(comp, annotation)) {
    absl::Span<const ResourcePair> rv =
        sched_state.async_tracker->GetResourcesFromInstruction(*instr);
    for (const auto& [resource, usage] : rv) {
      if (usage == ResourceUsageType::kResourceOccupy) {
        num_resources_needed[resource]++;
      }
    }
  }
  return num_resources_needed;
}

bool DefaultSchedulerCore::SchedulingAnnotationCrossesOverlapLimit(
    const SchedulingState& sched_state, int64_t annotation) {
  absl::flat_hash_map<int64_t, int64_t> num_resources_needed =
      GetNumResourcesNeededForAnnotation(sched_state, annotation);
  for (const auto& [resource, num_needed] : num_resources_needed) {
    int64_t limit = sched_state.max_concurrent_resource.at(resource);
    if (num_needed > limit) {
      return true;
    }
  }
  return false;
}
absl::StatusOr<std::vector<HloInstruction*>>
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
  if (annotation_tracker_->HasAnnotations(computation)) {
    sched_state.sched_graph.AnnotateGraph(annotation_tracker_.get());
    for (int64_t annotation :
         annotation_tracker_->GetAnnotations(computation)) {
      if (annotation_tracker_->GetSuccessors(computation, annotation).empty()) {
        VLOG(3) << "Annotation " << annotation
                << " does not have any successors, is ready to be scheduled";
        sched_state.ready_annotations.push_back(annotation);
      }
    }
  }
  sched_state.sched_graph.InitializeGraphAnalysis(async_tracker_);
  VLOG(5) << "Just built graph:";
  XLA_VLOG_LINES(5, sched_state.sched_graph.ToString(async_tracker_));
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
  while (!sched_state.ready_set.empty() || !sched_state.nop_set.empty()) {
    VLOG(10) << "Current ready time: " << sched_state.current_time;
    VLOG(2) << "Current ready queue:";
    XLA_VLOG_LINES(2, [&sched_state]() {
      struct LogFormatter {
        void operator()(std::string* out, const HloGraphNode* n) const {
          out->append(absl::StrCat("\t", n->GetInstr().name(),
                                   " Ready time: ", n->GetReadyTime(),
                                   " Depth: ", n->GetGraphDepth()));
        }
      };
      return absl::StrJoin(sched_state.ready_set, "\n", LogFormatter());
    }());
    if (!sched_state.ready_annotations.empty()) {
      // Pick the first ready annotation whose scheduling will not cross the
      // overlap limit. If there is no such annotation, continue with scheduling
      // non-annotated ops.
      int64_t annotation_index = -1;
      for (int64_t i = 0; i < sched_state.ready_annotations.size(); ++i) {
        if (SchedulingAnnotationCrossesOverlapLimit(
                sched_state, sched_state.ready_annotations[i])) {
          continue;
        }
        annotation_index = i;
        break;
      }
      if (annotation_index != -1) {
        std::swap(sched_state.ready_annotations[annotation_index],
                  sched_state.ready_annotations.back());
        int64_t annotation = sched_state.ready_annotations.back();
        sched_state.ready_annotations.pop_back();
        VLOG(2) << "------- BEGIN ANNOTATION: " << annotation << " -------";
        sched_state.ongoing_annotation = annotation;
        TF_RETURN_IF_ERROR(
            ScheduleAnnotation(computation, annotation, &sched_state));
        VLOG(2) << "-------  END ANNOTATION: " << annotation << " --------";
        sched_state.ongoing_annotation = -1;
        continue;
      }
    }
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
  absl::c_reverse(sched_state.new_sequence_reversed);
  if (post_processing_fn_) {
    post_processing_fn_(sched_state);
  }
  CHECK_EQ(sched_state.new_sequence_reversed.size(),
           sched_state.sched_graph.GetOriginalInstrList().size())
      << "Not all instructions have been scheduled "
      << sched_state.new_sequence_reversed.size() << " vs "
      << sched_state.sched_graph.GetOriginalInstrList().size();
  VLOG(2) << "Total time: "
          << sched_state.sched_graph
                 .GetNode(sched_state.new_sequence_reversed.front())
                 .GetReadyTime();

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
    kAllToAll,
    kRaggedAllToAll,
    kReduceScatter,
    kSend,
    kRecv,
    kCollectiveBroadcast,
  };
  auto opcode_to_async_kind = [](HloOpcode opcode) {
    switch (opcode) {
      case HloOpcode::kAllGather:
        return AsyncKind::kAllGather;
      case HloOpcode::kAllReduce:
        return AsyncKind::kAllReduce;
      case HloOpcode::kCollectiveBroadcast:
        return AsyncKind::kCollectiveBroadcast;
      case HloOpcode::kCollectivePermute:
        return AsyncKind::kCollectivePermute;
      case HloOpcode::kAllToAll:
        return AsyncKind::kAllToAll;
      case HloOpcode::kRaggedAllToAll:
        return AsyncKind::kRaggedAllToAll;
      case HloOpcode::kReduceScatter:
        return AsyncKind::kReduceScatter;
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
  auto find_outstanding_async = [&outstanding_collectives,
                                 async_tracker](const HloInstruction* instr) {
    const auto& collective_vec =
        outstanding_collectives[async_tracker->GetCanonicalAsyncOp(*instr)
                                    .inner];
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
      outstanding_collectives[async_tracker->GetCanonicalAsyncOp(*instr).inner]
          .push_back({instr, current_time, curr_pos});
    } else if (async_tracker->IsSupportedAsyncDone(*instr)) {
      const HloInstruction* start_instr = instr->operand(0);
      // TODO(b/329731042): Handle pipelined Send/Recv in while-body, which
      // is the only situation where an async done operand is not an async
      // start.
      if (async_tracker->IsSupportedAsyncStart(*start_instr)) {
        auto it = find_outstanding_async(start_instr);
        const HloGraphNode& start_node =
            schedule_graph.GetNode(std::get<0>(*it));
        auto edge_it = find_node_successor_edge(start_node, instr_node);
        const double async_wasted_cycles = std::max(
            0.0, edge_it->Latency() - (current_time - std::get<1>(*it)));
        AsyncKind kind = opcode_to_async_kind(
            async_tracker->GetCanonicalAsyncOp(*start_instr).inner);
        wasted_time_per_collective[kind] += async_wasted_cycles;
        current_time += async_wasted_cycles;
      }
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
      /*collective_broadcast_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kCollectiveBroadcast],
      /*collective_permute_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kCollectivePermute],
      /*all_to_all_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kAllToAll],
      /*ragged_all_to_all_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kRaggedAllToAll],
      /*reduce_scatter_wasted_cycles=*/
      wasted_time_per_collective[AsyncKind::kReduceScatter],
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
  if (const HloComputation* comp = sched_stats.computation) {
    absl::StrAppend(&result, "For computation: ", comp->name(), ", module ",
                    comp->parent()->name(), "(", comp->parent()->unique_id(),
                    ")\n");
  }
  absl::StrAppend(&result, "Total wasted cycles: ",
                  sched_stats.all_gather_wasted_cycles +
                      sched_stats.all_reduce_wasted_cycles +
                      sched_stats.collective_broadcast_wasted_cycles +
                      sched_stats.collective_permute_wasted_cycles +
                      sched_stats.all_to_all_wasted_cycles +
                      sched_stats.ragged_all_to_all_wasted_cycles +
                      sched_stats.reduce_scatter_wasted_cycles +
                      sched_stats.send_wasted_cycles +
                      sched_stats.recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Wasted cycles for all-reduce: ",
                  sched_stats.all_reduce_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-gather: ",
                  sched_stats.all_gather_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-broadcast: ",
                  sched_stats.collective_broadcast_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-permute: ",
                  sched_stats.collective_permute_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-to-all: ",
                  sched_stats.all_to_all_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for ragged-all-to-all: ",
                  sched_stats.ragged_all_to_all_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for reduce-scatter: ",
                  sched_stats.reduce_scatter_wasted_cycles, "\n");
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

absl::StatusOr<bool> LatencyHidingScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "Original module:";
  XLA_VLOG_LINES(5, module->ToString());
  // Currently we expect that a schedule that minimizes memory pressure is
  // provided as a base. It's not necessary for the algorithm itself but it
  // allows us to not having to think for now about memory pressure.
  CHECK(module->has_schedule()) << "LatencyHidingScheduler expects a base "
                                   "schedule that minimizes memory pressure.";
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
  absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>
      saved_schedules;
  TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
  for (HloComputation* computation : computations_to_schedule) {
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        scheduler_core_->ScheduleComputation(computation));
    saved_schedules[computation] = std::move(new_schedule);
  }
  uint64_t initial_memory_limit = scheduler_core_->GetMemoryLimit();
  for (int64_t iter = 0;
       iter < scheduler_core_->GetRerunTimes() &&
       scheduler_core_->GetMemoryPeak() > initial_memory_limit;
       iter++) {
    LOG(INFO) << "LatencyHidingScheduler current memory usage: "
              << scheduler_core_->GetMemoryPeak()
              << " bytes, does not fit in limit: "
              << scheduler_core_->GetMemoryLimit()
              << ". Setting the new limit to "
              << scheduler_core_->GetMemoryLimit() * 0.9;
    TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
    scheduler_core_->SetMemoryLimit(scheduler_core_->GetMemoryLimit() * 0.9);
    for (HloComputation* computation : computations_to_schedule) {
      TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                          scheduler_core_->ScheduleComputation(computation));
      saved_schedules[computation] = std::move(new_schedule);
    }
  }
  LOG(INFO) << "LatencyHidingScheduler current memory usage: "
            << scheduler_core_->GetMemoryPeak()
            << " bytes. Current limit: " << scheduler_core_->GetMemoryLimit();
  for (HloComputation* computation : computations_to_schedule) {
    VLOG(1) << "Statistics before scheduling:";
    LogScheduleStatistics(computation);
    module->schedule().set_sequence(
        computation, absl::MakeConstSpan(saved_schedules[computation]));
    VLOG(1) << "Statistics after scheduling:";
    LogScheduleStatistics(computation);
  }
  return true;
}

}  // namespace xla
