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
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

const int64_t kDefaultMemorySpace = 0;

bool IsNopInstruction(HloOpcode op, const HloInstruction& hlo) {
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kBroadcast || op == HloOpcode::kIota ||
         hlo.IsEffectiveBitcast(op) ||
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
  if (buffer_value_info.non_default_memory_space_layout) {
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

bool HasKeepOriginalSequenceOrderInGroupAttribute(const HloInstruction* instr) {
  auto attr =
      instr->get_frontend_attribute("keep_original_sequence_order_in_group");
  return attr.has_value() && attr.value() == "true";
}

bool IsCustomCallWithForceEarlyAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  return instr->opcode() == HloOpcode::kCustomCall && attr.has_value() &&
         attr.value() == "force_early";
}

bool IsCustomCallWithForceDelayAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  return instr->opcode() == HloOpcode::kCustomCall && attr.has_value() &&
         attr.value() == "force_delay";
}

int GetCustomCallForceDelayPriority(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_delay_priority");
  if (instr->opcode() == HloOpcode::kCustomCall && attr.has_value()) {
    int out;
    CHECK(absl::SimpleAtoi(attr.value(), &out))
        << "Failed to parse scheduler_delay_priority attribute: "
        << attr.value();
    return out;
  }
  return 0;
}

bool HasForceDelayAsyncAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  return attr.has_value() && attr.value() == "force_delay_async";
}

absl::flat_hash_map<int64_t, int64_t>
GetNumResourcesNeededForAnnotationWithKeepOriginalOrderAttrs(
    const DefaultSchedulerCore::SchedulingState& sched_state,
    std::vector<const HloInstruction*> instrs) {
  // For groups with forced sequence order, we need to obtain the accurate
  // resource usage info by traversing the instructions in order and keeping a
  // running record of the resource usage.
  std::sort(instrs.begin(), instrs.end(),
            [&sched_state](const HloInstruction* a, const HloInstruction* b) {
              auto& a_node = sched_state.sched_graph.GetNode(a);
              auto& b_node = sched_state.sched_graph.GetNode(b);
              return a_node.GetOriginalPosition() >
                     b_node.GetOriginalPosition();
            });
  absl::flat_hash_map<int64_t, int64_t> max_resources_needed;
  absl::flat_hash_map<int64_t, int64_t> current_resources_needed;
  for (const HloInstruction* instr : instrs) {
    // If "keep_original_sequence_order_in_group" attribute is set, we require
    // all instructions in the scheduling group to have this attribute set.
    CHECK(HasKeepOriginalSequenceOrderInGroupAttribute(instr));
    // Scheduling an async-start op will decrease the number of resources in
    // use.
    if (sched_state.async_tracker->IsSupportedAsyncStart(*instr)) {
      CHECK_EQ(instr->users().size(), 1);
      auto* async_done = *instr->users().begin();
      CHECK(sched_state.async_tracker->IsSupportedAsyncDone(*async_done));
      auto num_resources_needed_per_instr =
          sched_state.async_tracker->GetNumResourcesPerInstruction(*async_done);
      for (const auto& [resource, usage] : num_resources_needed_per_instr) {
        current_resources_needed[resource] -= usage;
      }
    } else {
      auto num_resources_needed_per_instr =
          sched_state.async_tracker->GetNumResourcesPerInstruction(*instr);
      for (const auto& [resource, usage] : num_resources_needed_per_instr) {
        current_resources_needed[resource] += usage;
      }
      for (const auto& [resource, usage] : current_resources_needed) {
        max_resources_needed[resource] =
            std::max(max_resources_needed[resource], usage);
      }
    }
  }
  return max_resources_needed;
}

int64_t EstimateFragmentationSize(HloModule* module,
                                  const HloAliasAnalysis& alias_analysis,
                                  const AliasInfo* alias_info) {
  // Run heap simulator on the whole module to estimate the fragmentation size.
  auto algorithm = std::make_unique<GlobalDecreasingSizeBestFitHeap<HloValue>>(
      /*alignment=*/1);
  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) -> int64_t {
    const Shape& shape = buffer.shape();
    if (!shape.IsArray()) {
      return 0;
    }
    if (!shape.has_layout()) {
      return 0;
    }
    if (shape.layout().memory_space() != Layout::kDefaultMemorySpace) {
      return 0;
    }
    return ShapeUtil::ByteSizeOf(shape);
  };
  auto result =
      HeapSimulator::Run(std::move(algorithm), *module, module->schedule(),
                         alias_analysis, alias_info, &size_fn);
  CHECK_OK(result.status());
  int64_t fragmentation_size = result.value().fragmentation_size;
  VLOG(3) << module->name() << ": Heap simulator estimated fragmentation size: "
          << fragmentation_size;
  return fragmentation_size > 0 ? fragmentation_size : 0;
}

}  // namespace

CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo) {
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

ResourceType AsyncTracker::GetResourceTypeForOp(HloOpcode op) {
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
}

ResourcesVector AsyncTracker::GetResourcesFromInstructionImpl(
    const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kAsyncStart || op.outer == HloOpcode::kAsyncDone) {
    ResourceType type = GetResourceTypeForOp(op.inner);
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
    case HloOpcode::kWhile: {
      ResourcesVector result;
      absl::flat_hash_set<int64_t> seen_occupied_resources;
      absl::flat_hash_set<int64_t> seen_released_resources;
      absl::flat_hash_set<int64_t> seen_no_resource;
      for (const HloInstruction* instr : hlo.while_body()->instructions()) {
        ResourcesVector rv = GetResourcesFromInstructionImpl(*instr);
        if (rv.empty()) {
          continue;
        }
        for (const auto& [resource, usage] : rv) {
          if (usage == ResourceUsageType::kResourceOccupy &&
              !seen_occupied_resources.contains(resource)) {
            seen_occupied_resources.insert(resource);
            result.push_back(std::make_pair(resource, usage));
          } else if (usage == ResourceUsageType::kResourceRelease &&
                     !seen_released_resources.contains(resource)) {
            seen_released_resources.insert(resource);
            result.push_back(std::make_pair(resource, usage));
          } else if (usage == ResourceUsageType::kNoResource &&
                     !seen_no_resource.contains(resource)) {
            seen_no_resource.insert(resource);
            result.push_back(std::make_pair(resource, usage));
          }
        }
      }
      return result;
    }
    default: {
      // At this point we are dealing with sync instructions that did not fall
      // into any of the cases above. We model their resources as a
      // kResourceOccupy and a kResourceRelease that follows immediately after.
      ResourcesVector res;
      if (config_.track_sync_op_resource_usage) {
        ResourceType type = GetResourceTypeForOp(hlo.opcode());
        if (type != ResourceType::kNoResource) {
          res.push_back(std::make_pair(ResourceTypeToIndex(type),
                                       ResourceUsageType::kResourceOccupy));
          res.push_back(std::make_pair(ResourceTypeToIndex(type),
                                       ResourceUsageType::kResourceRelease));
        }
      }
      return res;
    }
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

absl::flat_hash_map<int64_t, int64_t>
AsyncTracker::GetNumResourcesPerInstruction(const HloInstruction& instr) const {
  absl::flat_hash_map<int64_t, int64_t> num_resources_per_type;
  if (instr.called_computations().empty() ||
      instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    for (const auto& [resource_type, usage] :
         GetResourcesFromInstruction(instr)) {
      if (usage == ResourceUsageType::kResourceOccupy) {
        ++num_resources_per_type[resource_type];
      }
    }
  }

  // For instructions calling multiple computations, we assume that the called
  // computations do not execute in parallel (i.e., they are either mutually
  // exclusive, as in conditionals, or executed in sequence). Then for each
  // resource type, the usage across all called computations is the maximum
  // usage in any of the called computations.
  for (const HloComputation* computation : instr.called_computations()) {
    const auto& map = RecursivelyComputeResourceMap(computation);
    for (const auto& [resource_type, num_resources] : map) {
      num_resources_per_type[resource_type] =
          std::max(num_resources_per_type[resource_type], num_resources);
    }
  }
  return num_resources_per_type;
}

const absl::flat_hash_map<int64_t, int64_t>&
AsyncTracker::RecursivelyComputeResourceMap(
    const HloComputation* computation) const {
  // If the computation has a schedule, use the scheduled instruction order to
  // estimate the resource usage. We schedule the computation in post-order, so
  // it is guaranteed that all the callees are already scheduled before we
  // schedule the caller.
  auto& schedule = computation->parent()->schedule();
  if (schedule.is_computation_scheduled(computation)) {
    return RecursivelyComputeResourceMapForScheduledComputation(computation);
  }
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

const absl::flat_hash_map<int64_t, int64_t>&
AsyncTracker::RecursivelyComputeResourceMapForScheduledComputation(
    const HloComputation* computation) const {
  auto& schedule = computation->parent()->schedule();
  CHECK(schedule.is_computation_scheduled(computation));
  auto& m = async_in_computation_cache_[computation];
  if (m != nullptr) {
    return *m;
  }
  m = std::make_unique<absl::flat_hash_map<int64_t, int64_t>>();
  auto& res_map = *m;
  auto& inst_sequence = schedule.sequence(computation).instructions();
  // Traverse the sequence in reverse order and keep a running status of the
  // resources being used.
  absl::flat_hash_map<int64_t, int64_t> inflight_resource_usage;
  for (auto it = inst_sequence.rbegin(); it != inst_sequence.rend(); ++it) {
    const HloInstruction* inst = *it;
    if (IsSupportedAsyncDone(*inst)) {
      for (const auto& [type, _] : GetResourcesFromInstruction(*inst)) {
        int64_t current_usage = ++inflight_resource_usage[type];
        int64_t& max_usage = res_map[type];
        max_usage = std::max(max_usage, current_usage);
      }
    } else if (IsSupportedAsyncStart(*inst)) {
      for (const auto& resource : GetResourcesFromInstruction(*inst)) {
        --inflight_resource_usage[resource.first];
      }
    }
    for (const HloComputation* called_comp : inst->called_computations()) {
      for (auto& called_per_opcode_pair :
           RecursivelyComputeResourceMap(called_comp)) {
        int64_t type = called_per_opcode_pair.first;
        int64_t current_usage =
            inflight_resource_usage[type] + called_per_opcode_pair.second;
        int64_t& max_usage = res_map[type];
        max_usage = std::max(max_usage, current_usage);
      }
    }
  }
  return res_map;
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
        << "Target-defined resource " << GetResourceName(resource_type)
        << " with id " << resource_type
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
        // Skip computations that don't have schedules (e.g., host computations
        // created during compute offload that are executed separately).
        // Note: We don't need to track memory usage on CPU for now. If needed,
        // it can be added later.
        if (!module->schedule().is_computation_scheduled(computation)) {
          return;
        }
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
        // Skip computations that don't have schedules (e.g., host computations
        // created during compute offload that are executed separately).
        // Note: We don't need to track memory usage on CPU for now. If needed,
        // it can be added later.
        if (!module_->schedule().is_computation_scheduled(computation)) {
          return;
        }
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
          VLOG(15) << "Instruction: " << instruction->ToString();
          VLOG(15) << "Pressure change: "
                   << tracker.MemoryPressureDifference(instruction).first;
          VLOG(15) << "Current usage: " << tracker.memory_usage();
          tracker.UpdateBuffers(instruction);
          VLOG(15) << "Current usage after update: " << tracker.memory_usage();
          VLOG(15) << "Current peak after update: "
                   << tracker.pressure_state().memory_peak;
        }
        VLOG(15) << "Pressure peak for " << computation->name() << ": "
                 << tracker.pressure_state().memory_peak;
        UpdatePressureStateForComputation(computation,
                                          tracker.pressure_state());
      };
  process_computation(module_->entry_computation(), {});
}

void MemoryPressureTracker::Reset(const HloComputation* computation,
                                  const LiveBufferSet& initial_live_buffers) {
  live_memory_usage_ = 0;
  initial_memory_pressure_ = 0;
  pressure_state_ = MemoryPressureState{};
  live_buffers_set_.clear();

  if (!initial_live_buffers.empty()) {
    for (HloBuffer::Id id : initial_live_buffers) {
      auto& buffer = buffer_tracker_.GetBufferInfo(id);
      if (buffer.non_default_memory_space_layout) {
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

void MemoryPressureTracker::Initialize(
    const HloComputation* computation,
    const LiveBufferSet& initial_live_buffers) {
  int32_t next_id = 0;
  output_buffers_.clear();
  defined_buffers_.clear();

  for (auto* instruction : computation->instructions()) {
    instruction_ids_[instruction] = next_id++;
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

  alloc_release_spans_.resize(next_id);
  for (auto* instruction : computation->instructions()) {
    NodeAllocReleaseSpan s;
    s.start = alloc_release_ids_.size();
    s.num_alloc = ComputeBufferAllocations(instruction, &alloc_release_ids_);
    s.num_release = ComputeBufferReleases(instruction, &alloc_release_ids_);
    alloc_release_spans_[instruction_ids_[instruction]] = s;
  }

  // Call reset to set-up the mutable state.
  Reset(computation, initial_live_buffers);
}

int32_t MemoryPressureTracker::ComputeBufferAllocations(
    const HloInstruction* instruction, std::vector<HloBuffer::Id>* dst) {
  int32_t added = 0;
  for (auto* op : instruction->operands()) {
    auto it = output_buffers_.find(op);
    CHECK(it != output_buffers_.end());
    for (auto& b : it->second) {
      if (ShouldSkipBufferAllocations(instruction, b.second,
                                      b.first.first_definition) ||
          b.first.non_default_memory_space_layout) {
        continue;
      }
      dst->push_back(b.first.value->id());
      added++;
    }
  }
  return added;
}

int32_t MemoryPressureTracker::ComputeBufferReleases(
    const HloInstruction* instruction, std::vector<HloBuffer::Id>* dst) {
  int32_t added = 0;
  if (!ShouldSkipBufferReleases(instruction)) {
    auto it = defined_buffers_.find(instruction);
    CHECK(it != defined_buffers_.end());
    for (auto& b : it->second) {
      if (b.non_default_memory_space_layout) {
        continue;
      }
      if (InstructionFirstDefinesBuffer(instruction, b)) {
        dst->push_back(b.value->id());
        added++;
      }
    }
  }
  return added;
}

void MemoryPressureTracker::UpdateBuffers(const HloInstruction* instruction) {
  int64_t computations_peak = 0;
  for (auto* called_comp : instruction->called_computations()) {
    if (called_comp->IsFusionComputation()) {
      continue;
    }
    auto it = pressure_state_cache_.find(called_comp);
    // Skip computations that don't have pressure state tracked (e.g., host
    // computations created during compute offload that are executed
    // separately).
    if (it == pressure_state_cache_.end()) {
      continue;
    }
    computations_peak = std::max(computations_peak, it->second.memory_peak);
  }
  if (pressure_state_.memory_peak < live_memory_usage_ + computations_peak) {
    pressure_state_.memory_peak = live_memory_usage_ + computations_peak;
  }
  for (HloBuffer::Id id : allocated_buffer_ids(instruction)) {
    if (live_buffers_[id] == 0) {
      live_buffers_[id] = 1;
      live_buffers_set_.insert(id);
      live_memory_usage_ += buffer_tracker_.GetBufferInfo(id).buffer_size;
    }
  }
  pressure_state_.memory_peak =
      std::max(live_memory_usage_, pressure_state_.memory_peak);
  for (HloBuffer::Id id : released_buffer_ids(instruction)) {
    if (live_buffers_[id] != 0) {
      live_memory_usage_ -= buffer_tracker_.GetBufferInfo(id).buffer_size;
      live_buffers_set_.erase(id);
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
      // Skip computations that don't have pressure state tracked (e.g., host
      // computations created during compute offload that are executed
      // separately).
      if (it == pressure_state_cache_.end()) {
        continue;
      }
      // Take max increase of the called computations.
      peak = called_comp_peak =
          std::max(called_comp_peak, it->second.memory_peak);
    }
  }
  // Allocate memory increase from the operands and record increase in peak.
  for (HloBuffer::Id id : allocated_buffer_ids(instruction)) {
    if (!live_buffers_[id]) {
      increase += buffer_tracker_.GetBufferInfo(id).buffer_size;
    }
  }
  peak = std::max(increase, peak);
  // Decrease memory pressure if some buffers are released.
  for (HloBuffer::Id id : released_buffer_ids(instruction)) {
    if (live_buffers_[id]) {
      increase -= buffer_tracker_.GetBufferInfo(id).buffer_size;
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
 private:
  template <typename T>
  static inline int ThreeWay(T avalue, T bvalue) {
    if (ABSL_PREDICT_TRUE(avalue == bvalue)) {
      return 0;
    }
    return (avalue < bvalue) ? -1 : 1;
  }

  // Macros used for making the comparison logic more convenient and concise
  // Assumptions:
  // They are invoked in an environment where:
  //  "a" refers to a candidate DefaultSchedulerCore::ScheduleCandidate&
  //  "b" refers to a candidate DefaultSchedulerCore::ScheduleCandidate&
  //  "an" refers to a.node
  //  "bn" refers to b.node
  //  "reason" refers to const char** pointer used for returning a reason
#define RETURN_LOGIC(v, reason_str) \
  do {                              \
    if ((v) != 0) {                 \
      *reason = reason_str;         \
      return ((v) > 0);             \
    }                               \
  } while (0)

#define CMP_PROPERTY(property, reason_str)              \
  do {                                                  \
    if (int v = ThreeWay(an->property, bn->property)) { \
      RETURN_LOGIC(v, reason_str);                      \
    }                                                   \
  } while (0)
#define CMP_EXPLICIT(pa, pb, reason_str) \
  do {                                   \
    if (int v = ThreeWay((pa), (pb))) {  \
      RETURN_LOGIC(v, reason_str);       \
    }                                    \
  } while (0)

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
        early_target_scheduling_rule_(early_target_scheduling_rule),
        config_memory_limit_(sched_state_.config.memory_limit),
        config_has_memory_limit_(config_memory_limit_ != UINT64_MAX),
        has_target_scheduling_rule_(target_scheduling_rule_ != nullptr),
        has_early_target_scheduling_rule_(early_target_scheduling_rule_ !=
                                          nullptr) {}

  std::optional<bool> MemoryPressurePolicy(
      const HloGraphNode* an, std::pair<int64_t, int64_t>& a_increase,
      const HloGraphNode* bn, std::pair<int64_t, int64_t>& b_increase,
      const char** reason) const {
    // If out of memory reduce memory at all costs. Choose the instruction
    // that causes the most decrease (or least further increase) of memory
    // pressure.
    int64_t memory_usage = sched_state_.memory_pressure_tracker->memory_usage();
    if (memory_usage >= config_memory_limit_) {
      if (sched_state_.config.depth_based_memory_pressure_reduction) {
        CMP_EXPLICIT(
            a_increase.first < 0 && a_increase.first < b_increase.first,
            b_increase.first < 0 && b_increase.first < a_increase.first,
            "kOnlyDecreaseMemoryOverLimit");
        // If there's none than prefer a node that is the deepest. That
        // matches well with unlocking pressure-reducing nodes for typical
        // ML graphs.
        CMP_PROPERTY(GetGraphDepth(), "kDepthOverLimit");
      }
      // Otherwise pick a node that increases the pressure the least from
      // the list.
      if (a_increase.first != b_increase.first) {
        CMP_EXPLICIT(a_increase.first < b_increase.first,
                     b_increase.first < a_increase.first,
                     "kDecreaseMemoryOverLimit");
      }
    }
    // Avoid to bring peak beyond limit. Choose instruction that doesn't do
    // so.
    CMP_EXPLICIT(a_increase.second + memory_usage <= config_memory_limit_,
                 b_increase.second + memory_usage <= config_memory_limit_,
                 "kMemoryPeakOverLimit");
    return std::nullopt;
  }

  inline std::optional<bool> ReleaseStartPolicy(const HloGraphNode* an,
                                                const HloGraphNode* bn,
                                                const char** reason) const {
    // Prioritise scheduling ready "start" ops, to avoid useless extension
    // of start-done latencies. This benefits future latency ops, as ops
    // postponed here may be used to hide not-yet-scheduled latency ops.
    const ApproximateLatencyEstimator::TimeCost a_ready_interval =
        an->GetReadyTime() - sched_state_.current_time;
    const ApproximateLatencyEstimator::TimeCost b_ready_interval =
        bn->GetReadyTime() - sched_state_.current_time;
    bool a_ready_and_release =
        a_ready_interval <= 0 &&
        an->DoesReleaseResource(ResourceType::kCollectivePermute);
    bool b_ready_and_release =
        b_ready_interval <= 0 &&
        bn->DoesReleaseResource(ResourceType::kCollectivePermute);
    CMP_EXPLICIT(a_ready_and_release, b_ready_and_release, "kScheduleStart");
    if (a_ready_and_release && b_ready_and_release) {
      CMP_EXPLICIT(a_ready_interval, b_ready_interval, "kScheduleStart");
    }
    return std::nullopt;
  }

  inline bool AsyncDepth0CandidateCondition(
      DefaultSchedulerCore::ScheduleCandidate& a,
      const HloGraphNode* a_node) const {
    return !(a_node->DoesReleaseAnyResource() && a_node->GetAsyncDepth() == 0 &&
             !IsResourceConstrained(a, a_node));
  }

  inline bool ShouldScheduleAsyncDone(
      DefaultSchedulerCore::ScheduleCandidate& gn_cand,
      const HloGraphNode* gn_node) const {
    if (!gn_node->DoesOccupyAnyResource()) {
      return false;
    }
    return !ShouldDelaySendHostDone(gn_cand, gn_node);
  }

  inline std::optional<bool> IsValuableForSelectiveOverlap(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b, const char** reason) const {
    int64_t distance_to_selective_overlap_for_a =
        GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, a.node);
    int64_t distance_to_selective_overlap_for_b =
        GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, b.node);
    // If a is valuable for selective overlap and there is a selective
    // overlap in the near future a can be scheduled inside, hold off
    // scheduling a and schedule b instead. Same logic applies in reverse.
    int64_t max_distance =
        sched_state_.config.max_hops_to_closest_selective_overlap;
    CMP_EXPLICIT(
        // Reversal of b and a here is intentional due to comment above
        (b.node->GetValuableForSelectiveOverlap() &&
         distance_to_selective_overlap_for_b <= max_distance),
        (a.node->GetValuableForSelectiveOverlap() &&
         distance_to_selective_overlap_for_a <= max_distance),
        "kNotValuableForSelectiveOverlap");
    return std::nullopt;
  }

  // An adaptor that turns the TargetSchedulingRule function signature
  // into the more streamlined flow we expect where we just update
  // *reason and return true or false depending on whether we should copy
  // a into b.
  static inline std::optional<bool> InvokeTargetSchedulingFunction(
      DefaultSchedulerCore::TargetSchedulingRule func,
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b, const char** reason) {
    DCHECK(func != nullptr);
    if (std::optional<DefaultSchedulerCore::CandidateResult> r = func(a, b)) {
      *reason = r->reason;
      // Return true if we should move to "a"; false if we stay with "b"
      return (&r->result == &a);
    }
    return std::nullopt;
  }

  inline std::optional<bool> DelayAsyncStartCandidateCondition(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b, const HloGraphNode* a_node,
      const HloGraphNode* b_node, const char** reason) const {
    bool a_has_async_resource =
        a_node->DoesReleaseAnyResource() && !IsResourceConstrained(a, a_node);
    bool b_has_async_resource =
        b_node->DoesReleaseAnyResource() && !IsResourceConstrained(b, b_node);

    CMP_EXPLICIT(!a_has_async_resource, !b_has_async_resource,
                 "kDelayAsyncStartForCompute");
    if (a_has_async_resource && b_has_async_resource) {
      // If 2 nodes are both async nodes, we prioritize the one
      // with more depth to free up more computes to overlap
      // with the one with less depth which can be launched
      // early
      CMP_EXPLICIT(a_node->GetDepth() > b_node->GetDepth(),
                   b_node->GetDepth() > a_node->GetDepth(),
                   "kDelayAsyncStartForDepth");
    }
    return std::nullopt;
  }

  // The comparison here implements the priority for the nodes in the ready
  // set. The function compares a and b in a series of prioritized
  // comparisons. As soon as it finds one that is not equal, it stops.  If
  // "a" (the candidate) is "better" than b (the best node found so far), it
  // return true. Otherwise it returns false. *reason is always update to
  // point to a string describing which heuristic ultimately made the
  // decision.
  bool AIsBetterThanB(DefaultSchedulerCore::ScheduleCandidate& a,
                      DefaultSchedulerCore::ScheduleCandidate& b,
                      const char** reason) const {
    HloGraphNode* an = a.node;
    HloGraphNode* bn = b.node;
    // Schedule according to ForceEarly.
    CMP_PROPERTY(GetForceEarly(), "kForceEarly");
    // Schedule according to ForceDelay, if exactly one of the two instructions
    // has ForceDelay set.
    CMP_EXPLICIT(!an->GetForceDelay(), !bn->GetForceDelay(), "kForceDelay");
    // Schedule according to highest ForceDelay first, if both instructions
    // have ForceDelay set.
    CMP_EXPLICIT(-an->GetForceDelayPriority(), -bn->GetForceDelayPriority(),
                 "kForceDelayPriority");
    // Use the preference value (comes from a heuristic) to choose between
    // the two candidates. If two preferences are the same regular LHS logic
    // will run as usual, we take advantage of this fact when initializing
    // the heuristic algorithm.
    CMP_PROPERTY(GetPreference(), "kPreference");

    const SchedulerConfig& config = sched_state_.config;
    if (config.force_delay_over_memory_pressure) {
      if (ABSL_PREDICT_FALSE(has_early_target_scheduling_rule_)) {
        if (auto value = InvokeTargetSchedulingFunction(
                early_target_scheduling_rule_, a, b, reason)) {
          return *value;
        }
      }

      // Schedule according to ForceDelayAfterTarget when we executed the
      // early target scheduling rule.
      CMP_EXPLICIT(!an->GetForceDelayAfterTarget(),
                   !bn->GetForceDelayAfterTarget(), "kForceDelayAfterTarget");
    }

    std::pair<int64_t, int64_t> a_increase = {0, 0};
    std::pair<int64_t, int64_t> b_increase = {0, 0};
    bool computed_memory_increases = false;
    if (config_has_memory_limit_ &&
        sched_state_.memory_pressure_tracker->memory_usage() >
            (config_memory_limit_ / 2)) {
      a_increase = GetMemoryPressureChanges(a, an);
      b_increase = GetMemoryPressureChanges(b, bn);
      computed_memory_increases = true;

      if (auto value =
              MemoryPressurePolicy(an, a_increase, bn, b_increase, reason)) {
        return *value;
      }
    }

    if (!config.force_delay_over_memory_pressure) {
      if (ABSL_PREDICT_FALSE(has_early_target_scheduling_rule_)) {
        if (auto value = InvokeTargetSchedulingFunction(
                early_target_scheduling_rule_, a, b, reason)) {
          return *value;
        }
      }

      // Schedule according to ForceDelayAfterTarget when we executed the
      // early target scheduling rule.
      CMP_EXPLICIT(!an->GetForceDelayAfterTarget(),
                   !bn->GetForceDelayAfterTarget(), "kForceDelayAfterTarget");
    }

    // Some heuristic that try to prioritize unlocking "done" instructions
    // so that we can perform overlap. More fancy heuristics can be used by
    // discovering the closest "done" to every instruction and prioritize
    // those that are closer rather than ones that are further away.
    CMP_EXPLICIT(ShouldScheduleAsyncDone(a, an), ShouldScheduleAsyncDone(b, bn),
                 "kScheduleDone");
    // The following rule targets the async ops using resources that should
    // be released right after the op's estimated time cost has past. It
    // prevents increasing the overlaps of such async ops more than
    // necessary.
    HloGraphNode::TimeCost a_past_due =
        PastDueCyclesForNonextendableResource(an);
    HloGraphNode::TimeCost b_past_due =
        PastDueCyclesForNonextendableResource(bn);
    CMP_EXPLICIT(a_past_due, b_past_due, "kReleaseNonextendable");

    if (ABSL_PREDICT_FALSE(config.enable_release_start_policy)) {
      // Prioritise scheduling ready "start" ops, to avoid useless extension
      // of start-done latencies. This benefits future latency ops, as ops
      // postponed here may be used to hide not-yet-scheduled latency ops.
      if (auto value = ReleaseStartPolicy(an, bn, reason)) {
        return *value;
      }
    }
    const bool aggressive_scheduling_policies =
        config.aggressive_scheduling_policies;
    if (aggressive_scheduling_policies &&
        config.prioritize_async_depth_over_stall) {
      // If an instruction releasing a resource is not resource constrained and
      // has an async depth of 0, delay it as much as possible to avoid
      // potential cost model inefficiencies.  For example, if a pair of
      // async-start and async-done have no dependencies on other ops inside a
      // loop, the async-start will be pushed to the beginning of the loop.
      CMP_EXPLICIT(AsyncDepth0CandidateCondition(a, an),
                   AsyncDepth0CandidateCondition(b, bn), "kStartAtZeroDepth");
    }

    if (sched_state_.config.aggressive_scheduling_policies &&
        sched_state_.config.prioritize_compute_over_async_start) {
      // If an instruction releasing a resource is not resource constrained,
      // delay it as much as possible.
      if (auto value =
              DelayAsyncStartCandidateCondition(a, b, an, bn, reason)) {
        return *value;
      }
    }

    auto a_readytime = an->GetReadyTime();
    auto b_readytime = bn->GetReadyTime();
    if (a_readytime != b_readytime) {  // Quick test to avoid lots of work
      const ApproximateLatencyEstimator::TimeCost a_ready_interval =
          std::max(a_readytime - sched_state_.current_time, 0.0);
      const ApproximateLatencyEstimator::TimeCost b_ready_interval =
          std::max(b_readytime - sched_state_.current_time, 0.0);
      // Make sure that between two instructions that are not ready we first
      // emit the one that causes less stall. This allows to potentially
      // expose more opportunities for the other to overlap.
      CMP_EXPLICIT(a_ready_interval < b_ready_interval,
                   b_ready_interval < a_ready_interval, "kLessStall");
    }
    if (config.resource_serializing) {
      // Prioritize scheduling the instruction which has less serial-resource
      // conflicts with the resources in flight.  We negate since we want to
      // prefer those with higher conflicting serial resources.
      CMP_EXPLICIT(-GetNumConflictingSerialResources(a, an),
                   -GetNumConflictingSerialResources(b, bn),
                   "kLessSerialResourceConflict");
    }
    if (ABSL_PREDICT_FALSE(aggressive_scheduling_policies &&
                           !config.prioritize_async_depth_over_stall)) {
      CMP_EXPLICIT(AsyncDepth0CandidateCondition(a, an),
                   AsyncDepth0CandidateCondition(b, bn), "kStartAtZeroDepth");
    }
    CMP_EXPLICIT(an->DoesReleaseAnyResource() && IsResourceConstrained(a, an),
                 bn->DoesReleaseAnyResource() && IsResourceConstrained(b, bn),
                 "kFreeBackedupResource");

    if (aggressive_scheduling_policies) {
      // Try to favor paths that are dependent of chains of async operations
      // with long latency as we want to get to them as soon as possible to
      // overlap them with computation.
      CMP_PROPERTY(GetAsyncDepth(), "kAsyncDepth");

      // Favor nodes that are the closest in amount of latency they hide
      // with the highest amount of latency that needs to be hidden to avoid
      // wasting / big nodes over small async operations.
      if (!sched_state_.next_ready_stack.empty()) {
        HloGraphNode::TimeCost latest_ready =
            sched_state_.next_ready_stack.front()->GetReadyTime();
        HloGraphNode::TimeCost a_cost_diff =
            std::abs(latest_ready - sched_state_.current_time - an->GetCost());
        HloGraphNode::TimeCost b_cost_diff =
            std::abs(latest_ready - sched_state_.current_time - bn->GetCost());
        CMP_EXPLICIT(!an->DoesReleaseAnyResource() && a_cost_diff < b_cost_diff,
                     !bn->DoesReleaseAnyResource() && b_cost_diff < a_cost_diff,
                     "kAvoidWaste");
      }
    }

    //  Check if any operand is an async done operation of the two ops to be
    //  compared. Prioritize those to unlock async dones to be scheduled.
    //  TODO(maggioni): Develop a more complete analysis of the graph to
    //  prioritize candidates that would more likely unlock more async dones
    //  to be scheduled.
    CMP_PROPERTY(HasOperandThatIsSupportedAsyncDone(), "kUnlockDone");

    if (ABSL_PREDICT_FALSE(has_target_scheduling_rule_)) {
      if (auto value = InvokeTargetSchedulingFunction(target_scheduling_rule_,
                                                      a, b, reason)) {
        return *value;
      }
    }

    // If there are no selective overlaps open currently and there will be
    // overlaps opened in the near future, hold off scheduling instructions
    // that are valuable for selective overlaps.
    if (config.enable_selective_resources &&
        sched_state_.selective_resource_releasers.empty()) {
      if (auto value = IsValuableForSelectiveOverlap(a, b, reason)) {
        return *value;
      }
    }

    if (aggressive_scheduling_policies) {
      // Favor nodes that unlock other nodes to be scheduled if possible.
      // This makes us more flexible in what we can use in scheduling.
      CMP_PROPERTY(GetReadyNodesIfScheduled(), "kCreatesMoreReadyNodes");
    }
    // If we computed memory pressure increase of instructions when we don't
    // have a better choice let's just choose the one that decreases the
    // memory pressure.
    if (computed_memory_increases) {
      CMP_EXPLICIT(a_increase.first < 0, b_increase.first < 0,
                   "kDecreaseMemory");
    }

    // Finally, break ties with original position
    *reason = "kOriginalOrder";
    return (an->GetOriginalPosition() > bn->GetOriginalPosition());
  }

  // "a" is a candidate instruction, and "b" is the best instruction found so
  // far.  Compare "a" to "b" to determine if "a" should replace "b" as a better
  // scheduling candidate.  If "a" (the candidate) is "better" than b (the best
  // node found so far), it returns true and stores "a" in "b". Otherwise it
  // returns false. *reason is always update to point to a string describing
  // which heuristic ultimately made the decision.
  bool MaybeUpdate(DefaultSchedulerCore::ScheduleCandidate& a,
                   DefaultSchedulerCore::ScheduleCandidate& b,
                   const char** reason) const {
    bool result = AIsBetterThanB(a, b, reason);
    if (result) {
      // Based on profiling, memcpy is faster than "b = a"
      static_assert(
          std::is_trivially_copyable_v<DefaultSchedulerCore::ScheduleCandidate>,
          "ScheduleCandidate should be is_trivially_copyable");
      if (VLOG_IS_ON(2)) {
        DefaultSchedulerCore::ScheduleCandidate tmp = b;
        memcpy(&b, &a, sizeof(DefaultSchedulerCore::ScheduleCandidate));
        memcpy(&a, &tmp, sizeof(DefaultSchedulerCore::ScheduleCandidate));
      } else {
        memcpy(&b, &a, sizeof(DefaultSchedulerCore::ScheduleCandidate));
      }
    }
    return result;
  }

 private:
  const DefaultSchedulerCore::SchedulingState& sched_state_;
  DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule_;
  DefaultSchedulerCore::TargetSchedulingRule early_target_scheduling_rule_;
  DefaultSchedulerCore::OverlapLimitRule
      scheduling_instruction_crosses_overlap_limit_;
  uint64_t config_memory_limit_;
  bool config_has_memory_limit_;
  bool has_target_scheduling_rule_;
  bool has_early_target_scheduling_rule_;

  static bool IsNop(const HloGraphNode& gn) {
    return IsNopInstruction(gn.GetOpcode(), gn.GetInstr());
  }
  bool IsResourceConstrained(DefaultSchedulerCore::ScheduleCandidate& cand,
                             const HloGraphNode* cand_node) const {
    if (cand.has_resource_constrained) {
      return cand.resource_constrained;
    }
    if (cand_node->GetResources().empty()) {
      cand.set_resource_constrained(false);
      return cand.resource_constrained;
    }
    cand.set_resource_constrained(false);
    for (const auto& [resource_type, usage_type] : cand_node->GetResources()) {
      auto max_it = sched_state_.max_concurrent_resource.find(resource_type);
      auto res_it = sched_state_.resource_users_in_queue.find(resource_type);
      cand.set_resource_constrained(
          max_it != sched_state_.max_concurrent_resource.end() &&
          max_it->second == 0 &&
          res_it != sched_state_.resource_users_in_queue.end() &&
          res_it->second > 0);
      if (cand.resource_constrained) {
        return cand.resource_constrained;
      }
    }
    return cand.resource_constrained;
  }
  HloGraphNode::TimeCost PastDueCyclesForNonextendableResource(
      const HloGraphNode* cand_node) const {
    if (cand_node->GetReleasedNonExtendableResources().empty()) {
      return 0.0;
    }
    return std::max(sched_state_.current_time - cand_node->GetReadyTime(), 0.0);
  }
  bool ShouldDelaySendHostDone(DefaultSchedulerCore::ScheduleCandidate& gn_cand,
                               const HloGraphNode* gn_node) const {
    const HloGraphNode& gn = *gn_node;
    if ((gn.GetOpcode() != HloOpcode::kSendDone) ||
        !gn.UsesResourceType(ResourceType::kSendHost).has_value()) {
      return false;
    }
    // Try to delay the send-done for host based operations
    // like outside compilation to avoid allocating memory
    // unnecessarily.
    const HloGraphNode& start =
        sched_state_.sched_graph.GetNode(gn.GetInstr().operand(0));
    const LatencyEstimator::TimeCost latency =
        sched_state_.latency_estimator->GetLatencyBetween(start, gn);
    if (!gn_cand.has_estimated_connected_send_ready_time) {
      HloGraphNode::TimeCost start_ready_time = 0;
      for (const auto& succ : start.GetSuccessors()) {
        // If any successor is not ready skip this logic. We
        // detect this by checking that ready time is set to
        // max. This should never happen because sends always
        // have 1 or 2 successors that should be scheduled or
        // ready already, but in case somebody comes up with
        // different patterns lets keep this check here.
        if (succ.Target().GetReadyTime() >=
            std::numeric_limits<HloGraphNode::TimeCost>::max()) {
          return false;
        }
        start_ready_time = std::max(
            start_ready_time, succ.Latency() + succ.Target().GetReadyTime());
      }
      gn_cand.set_estimated_connected_send_ready_time(start_ready_time);
    }
    if (gn_cand.estimated_connected_send_ready_time -
            sched_state_.current_time <=
        latency) {
      return false;
    }
    return true;
  }
  // Compute and cache memory pressure change computation for
  // candidate.
  std::pair<int64_t, int64_t> GetMemoryPressureChanges(
      DefaultSchedulerCore::ScheduleCandidate& cand,
      const HloGraphNode* cand_node) const {
    if (cand.has_pressure_change) {
      return {cand.pressure_change_first, cand.pressure_change_second};
    }
    std::optional<std::pair<int64_t, int64_t>> start_result;
    // In case of async-done instruction they can increase the
    // memory pressure but its always a possible move to
    // schedule the start immediately after, so for memory
    // pressure purpose in the scheduling heuristic actually
    // use the memory pressure change of the start rather than
    // the -done.
    if (cand_node->IsSupportedAsyncDone()) {
      const HloGraphNode* start =
          !cand_node->GetPredecessors().empty()
              ? &cand_node->GetPredecessors()[0].Target()
              : nullptr;
      if (start != nullptr && start->IsSupportedAsyncStart()) {
        start_result =
            sched_state_.memory_pressure_tracker->MemoryPressureDifference(
                &start->GetInstr());
      }
    }
    auto p = sched_state_.memory_pressure_tracker->MemoryPressureDifference(
        &cand_node->GetInstr());
    if (start_result.has_value()) {
      p.first = std::min(start_result->first, p.first);
      p.second = std::max(start_result->second, p.second);
    }
    cand.set_pressure_change(p);
    return {p.first, p.second};
  }
  int64_t GetNumConflictingSerialResources(
      DefaultSchedulerCore::ScheduleCandidate& cand,
      const HloGraphNode* cand_node) const {
    auto resources =
        sched_state_.async_tracker->GetOccupiedSerialResourcesFromVector(
            cand_node->GetResources());
    int64_t num_conflicting_resources = 0;
    for (int64_t resource : resources) {
      if (!sched_state_.resource_occupiers_in_flight.count(resource)) {
        continue;
      }
      num_conflicting_resources +=
          sched_state_.resource_occupiers_in_flight.at(resource).size();
    }
    return num_conflicting_resources;
  }
#undef RETURN_LOGIC
#undef CMP_PROPERTY
#undef CMP_EXPLICIT
};  // namespace

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

// Helper function to find the best node from the queues of scheduling state
// for scheduling.
absl::StatusOr<HloGraphNode*>
DefaultSchedulerCore::FindAndExtractBestNodeAvailable(
    DefaultSchedulerCore::SchedulingState& sched_state,
    DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node) {
  while (true) {
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
    bool ready_chosen_valid = false;
    auto chosen_it = sched_state.ready_set.end();

    // Try to pick nodes from the ready set first that are the ones that cause
    // the most latency hiding.
    const bool vlog_2 = VLOG_IS_ON(2);
    const bool has_should_skip_node = (should_skip_node != nullptr);
    for (auto ready_node_it = sched_state.ready_set.begin(),
              e = sched_state.ready_set.end();
         ready_node_it != e; ++ready_node_it) {
      HloGraphNode* ready_node = *ready_node_it;
      if (has_should_skip_node && should_skip_node(ready_node)) {
        if (!ready_chosen_valid) {
          skipped_nodes_and_reasons.push_back(
              {ready_node, SkipNodeReason::kShouldSkipNodeFunction});
          if (ABSL_PREDICT_FALSE(vlog_2)) {
            VLOG(2) << SkipNodeReasonString(
                           skipped_nodes_and_reasons.back().second)
                    << " node: " << ready_node->GetInstr().name();
          }
        }
        continue;
      }
      // These ifs will be true when the iterator points to an annotated node,
      // but the chosen node is nullptr because the annotation group is not
      // ready to be scheduled yet (because of the annotation roots' successors
      // not being scheduled yet). So we skip this node and continue to the next
      // one.
      if (ABSL_PREDICT_FALSE(ready_node->GetAnnotation() != -1)) {
        if (!ready_chosen_valid) {
          skipped_nodes_and_reasons.push_back(
              {ready_node, SkipNodeReason::kAnnotationGroupNotReady});
          if (ABSL_PREDICT_FALSE(vlog_2)) {
            VLOG(2) << SkipNodeReasonString(
                           skipped_nodes_and_reasons.back().second)
                    << " node: " << ready_node->GetInstr().name();
          }
        }
        continue;
      }
      // If this node would cause the max_concurrent_resource count to go beyond
      // the limit do not schedule it and pass to the next node.
      if (is_default_scheduling_instruction_crosses_overlap_limit_ &&
          !ready_node->HasRecursiveResources()) {
        // Default scheduling_instruction_crosses_overlap_limit_ is a noop in
        // this case
      } else {
        // Either scheduling_instruction_crosses_overlap_limit_ is not the
        // default, or the node actually has recursive resources.
        if (scheduling_instruction_crosses_overlap_limit_(sched_state,
                                                          ready_node)) {
          if (ready_chosen.node == nullptr) {
            skipped_nodes_and_reasons.push_back(
                {ready_node, SkipNodeReason::kExceedsOverlapLimit});
            if (ABSL_PREDICT_FALSE(vlog_2)) {
              VLOG(2) << SkipNodeReasonString(
                             skipped_nodes_and_reasons.back().second)
                      << " node: " << ready_node->GetInstr().name();
            }
          }
          continue;
        }
      }
      ScheduleCandidate ready_candidate =
          InitializeCandidate(ready_node, sched_state);
      if (!ready_chosen_valid) {
        ready_chosen = ready_candidate;
        chosen_it = ready_node_it;
        ready_chosen_valid = true;
        if (ABSL_PREDICT_FALSE(vlog_2)) {
          VLOG(2) << "Choosing from ready ("
                  << ready_chosen.node->GetInstr().name()
                  << ") Reason: First Candidate";
        }
        continue;
      }

      const char* reason;
      bool new_candidate_selected =
          ready_lt.MaybeUpdate(ready_candidate, ready_chosen, &reason);
      if (ABSL_PREDICT_FALSE(vlog_2)) {
        auto print_pressure_change =
            [](const DefaultSchedulerCore::ScheduleCandidate& p) {
              if (p.has_pressure_change) {
                return std::to_string(p.pressure_change_first);
              }
              return std::string("N/A");
            };
        VLOG(2) << "Choosing from ready ("
                << ready_chosen.node->GetInstr().name() << ") vs ("
                << ready_candidate.node->GetInstr().name()
                << ") Reason: " << reason << " mem pressure chosen "
                << print_pressure_change(ready_chosen) << " mem pressure other "
                << print_pressure_change(ready_candidate);
      }

      if (new_candidate_selected) {
        chosen_it = ready_node_it;
        DCHECK_EQ(ready_chosen.node, *chosen_it);
      }
    }

    if (ready_chosen_valid) {
      CHECK(chosen_it != sched_state.ready_set.end());
      std::swap(*chosen_it, sched_state.ready_set.back());
      sched_state.ready_set.pop_back();
      return ready_chosen.node;
    }

    if (sched_state.config.deannotate_group_if_blocked) {
      // If no node was chosen, check if any were skipped due to
      // kAnnotationGroupNotReady. Among those groups, pick the one which has
      // the smallest number of nodes in it.
      HloGraphNode* node_to_deannotate = nullptr;
      int64_t min_annotation_size = std::numeric_limits<int64_t>::max();
      const HloComputation* comp =
          sched_state.sched_graph.GetOriginalInstrList()[0]->parent();

      for (const auto& pair : skipped_nodes_and_reasons) {
        if (pair.second == SkipNodeReason::kAnnotationGroupNotReady) {
          int64_t annotation = pair.first->GetAnnotation();
          int64_t current_annotation_size =
              annotation_tracker_->GetNumInstructions(comp, annotation);
          if (current_annotation_size < min_annotation_size) {
            min_annotation_size = current_annotation_size;
            node_to_deannotate = pair.first;
          }
        }
      }

      if (node_to_deannotate != nullptr) {
        int64_t annotation = node_to_deannotate->GetAnnotation();
        VLOG(2) << "FindAndExtractBestNodeAvailable failed, deannotating group "
                << annotation << " and retrying.";
        const HloComputation* comp =
            sched_state.sched_graph.GetOriginalInstrList()[0]->parent();
        auto instrs = annotation_tracker_->GetInstructions(comp, annotation);
        for (const HloInstruction* instr : instrs) {
          HloGraphNode& node = sched_state.sched_graph.GetNode(instr);
          node.ClearAnnotation();
        }
        // Clear the ongoing annotation state as well.
        if (sched_state.ongoing_annotation == annotation) {
          sched_state.ongoing_annotation = -1;
        }
        // Remove this annotation from ready_annotations if it's there.
        auto it = absl::c_find(sched_state.ready_annotations, annotation);
        if (it != sched_state.ready_annotations.end()) {
          sched_state.ready_annotations.erase(it);
        }
        continue;  // Retry the while loop.
      }
    }

    // If we reach here, no node was scheduled and no annotation group could be
    // deannotated.
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

// This function assumes the existing occupiers' latencies are already
// adjusted and sorted by their projected finish time. WARNING: Do not add an
// edge with a current time smaller than the current times when the existing
// edges were inserted.
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
// policies for the nodes in the annotated ready set. The policies are
// currently very minimal (recall that the scheduling is done in the reverse
// order):
//  1. Async done nodes are scheduled before any other nodes.
//  2. Among other nodes, async start nodes are scheduled after other nodes.
class AnnotationReadySetLt {
 public:
  explicit AnnotationReadySetLt() = default;
  // Implements the priority for the nodes in the annotated ready set.
  DefaultSchedulerCore::CandidateResult operator()(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b) const {
    // Schedule based on the original sequence order if requested.
    auto a_attr = a.node->GetInstr().get_frontend_attribute(
        "keep_original_sequence_order_in_group");
    auto b_attr = b.node->GetInstr().get_frontend_attribute(
        "keep_original_sequence_order_in_group");
    if (a_attr.has_value() && b_attr.has_value()) {
      if (a_attr.value() == "true" && b_attr.value() == "true") {
        return {a.node->GetOriginalPosition() > b.node->GetOriginalPosition()
                    ? a
                    : b,
                "kOriginalSequenceOrder"};
      }
    }

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
    DefaultSchedulerCore::SchedulingState& sched_state,
    const std::function<bool(const DefaultSchedulerCore::SchedulingState&,
                             HloGraphNode*)>&
        scheduling_instruction_crosses_overlap_limit,
    bool is_default_scheduling_instruction_crosses_overlap_limit) {
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
    HloGraphNode* ready_node = *ready_node_it;
    if (is_default_scheduling_instruction_crosses_overlap_limit &&
        !ready_node->HasRecursiveResources()) {
      // Default scheduling_instruction_crosses_overlap_limit is a noop in
      // this case
    } else {
      // Either scheduling_instruction_crosses_overlap_limit_ is not the
      // default, or the ready_node actually has recursive resoures, so we
      // invoke the scheduling_instruction_crosses_overlap_limit function to
      // check if this node would cause the max_concurrent_resource count to
      // go beyond the limit.  If so, we do not schedule it and pass on to the
      // next node.
      if (scheduling_instruction_crosses_overlap_limit(sched_state,
                                                       ready_node)) {
        VLOG(2) << "Skipping node (" << ready_node->GetInstr().name()
                << ") Reason: scheduling instruction crosses overlap limit";
        continue;
      }
    }
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
        [](const DefaultSchedulerCore::ScheduleCandidate& p) {
          if (p.has_pressure_change) {
            return std::to_string(p.pressure_change_first);
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
            << print_pressure_change(new_candidate_selected ? ready_candidate
                                                            : ready_chosen)
            << " mem pressure other "
            << print_pressure_change(new_candidate_selected ? ready_chosen
                                                            : ready_candidate);
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
  int64_t non_ready_instr = 0;
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
    TF_ASSIGN_OR_RETURN(
        HloGraphNode * node,
        FindAndExtractBestAnnotatedNode(
            *sched_state, scheduling_instruction_crosses_overlap_limit_,
            is_default_scheduling_instruction_crosses_overlap_limit_));

    TF_RET_CHECK(node != nullptr)
        << "Couldn't find an annotated node to schedule.";
    // Delay last instruction of annotation maybe.
    if (sched_state->config.flexible_scheduling_annotation_scheduling &&
        num_scheduled == annotation_size - 1 &&
        scheduling_context_->GetAsyncTracker()->IsSupportedAsyncStart(
            node->GetInstr())) {
      // Give instruction back to the scheduler to schedule.
      VLOG(2) << "Non ready instr: " << node->GetInstr().name();
      ++non_ready_instr;
      node->ClearAnnotation();
      if (sched_state->config.aggressive_flexible_annotation_scheduling) {
        node->SetForceDelayAfterTarget(true);
      }
      sched_state->nodes_holding_annotations.insert(node);
      continue;
    }
    // Delete the node from the ready set.
    auto node_it = absl::c_find(sched_state->ready_set, node);
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
  // If for some reason we could not schedule all the instructions in the
  // annotation in one go, we clear the annotation for the remaining
  // instruction. Currently this should only happen for async-start
  // instructions.
  if (num_scheduled < annotation_size - non_ready_instr) {
    for (auto* inst :
         annotation_tracker_->GetInstructions(computation, annotation)) {
      HloGraphNode& node = sched_state->sched_graph.GetNode(inst);
      if (!node.IsScheduled()) {
        TF_RET_CHECK(
            scheduling_context_->GetAsyncTracker()->IsSupportedAsyncStart(
                node.GetInstr()));
        VLOG(2) << "Could not schedule all annotated nodes with annotation ID "
                << annotation << " in one go; clearing annotation for "
                << node.GetInstr().name();
        node.ClearAnnotation();
        sched_state->nodes_holding_annotations.insert(&node);
      }
    }
  }
  return absl::OkStatus();
}

// Returns the vector of annotations that the given node is a successor of,
// but is not included in that annotation group itself.
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
  sched_state->nodes_holding_annotations.erase(n);

  // If this node was a successor to one or more scheduling groups, update the
  // number of scheduled successors for each of those groups and add the group
  // the ready_annotations set if all of its successors have been scheduled.
  std::vector<int64_t> annotations = GetPredecessorAnnotations(n);
  if (!annotations.empty()) {
    VLOG(2) << "Scheduled node is a frontier: " << n->GetInstr().name();
    for (int64_t annotation : annotations) {
      DefaultSchedulerCore::SchedulingState::NumSuccessorsForAnnotation&
          num_successors_for_annotation =
              sched_state->num_successors_for_annotation[annotation];
      num_successors_for_annotation.scheduled++;
      VLOG(2) << "Annotation: " << annotation << " scheduled num successors: "
              << num_successors_for_annotation.scheduled
              << " total num successors: " << num_successors_for_annotation.all;
      // LegalizeSchedulingAnnotations pass should have made sure that we will
      // eventually reach a state where all successors of the annotation are
      // scheduled.
      if (num_successors_for_annotation.scheduled ==
          num_successors_for_annotation.all) {
        sched_state->ready_annotations.push_back(annotation);
      }
    }
  }
  // Remove scheduled node from selective_resources_releasers if it
  // was there.
  if (sched_state->config.enable_selective_resources &&
      n->ReleasesSelectiveResource()) {
    auto it = absl::c_find(sched_state->selective_resource_releasers, n);
    // Perform sanity check node was in selective_resources_releasers.
    if (it == sched_state->selective_resource_releasers.end()) {
      LOG(WARNING) << "Selective resource releasers list does not contain node "
                      "that releases a selective resource: "
                   << n->ToString();
    } else {
      sched_state->selective_resource_releasers.erase(it);
    }
  }

  // If scheduled node cannot overlap with nodes that hold selective
  // resources, we increment the ready time of all nodes that release a
  // selective resource with the cost of the scheduled node.
  if (sched_state->config.enable_selective_resources &&
      !n->GetValuableForSelectiveOverlap()) {
    for (HloGraphNode* node : sched_state->selective_resource_releasers) {
      node->SetReadyTime(node->GetReadyTime() + n->GetCost());
    }
  }

  // If this node is an instruction that occupies/releases resource(s), then
  // handle the increase/decrease.
  for (auto& resource : n->GetNetResources()) {
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
  for (HloEdge& pred : n->GetSuccessors()) {
    const HloGraphNode::TimeCost time_from_edge =
        pred.Target().GetReadyTime() + pred.Latency();
    schedule_time = std::max(schedule_time, time_from_edge);
    if (sched_state->config.resource_sharing) {
      // Adjust the ready time if this edge uses shareable resources
      auto occupied_resources =
          n->GetShareableResourcesOnEdge(&sched_state->sched_graph, pred);
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
      auto released_resources =
          n->GetShareableResourcesOnEdge(&sched_state->sched_graph, edge);
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
          auto occupied_resources = edge.Target().GetShareableResourcesOnEdge(
              &sched_state->sched_graph, inverse_edge);
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

  // Update the target defined states for the node before we release its
  // successors.
  scheduling_context_->GetAsyncTracker()->UpdateTargetDefinedStates(
      n->GetInstr(), &sched_state->sched_graph,
      scheduling_context_->GetLatencyEstimator().get(), current_time);

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
  // ready_set. If a released node ready time is higher than the current time
  // we put it also in the next_ready_stack, which is used in the ReadySetLt
  // class for nodes cost comparison.
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
    for (HloEdge& pred : edge.Target().GetSuccessors()) {
      const LatencyEstimator::TimeCost edge_time =
          pred.Target().GetReadyTime() + pred.Latency();
      ready_time = std::max(ready_time, edge_time);
      if (sched_state->config.resource_sharing) {
        // Adjust the ready time if this edge uses shareable resources
        auto occupied_resources = edge.Target().GetShareableResourcesOnEdge(
            &sched_state->sched_graph, pred);
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
    if (IsNopInstruction(edge.Target().GetOpcode(), edge.Target().GetInstr()) &&
        annotation == -1) {
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
      // For supported async collective done ops, save their corresponding
      // start ops in the map
      if (n->IsSupportedAsyncDone() &&
          scheduling_context_->GetAsyncTracker()->IsSupportedAsyncStart(
              *n->GetInstr().operand(0))) {
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
  int64_t memory_after = sched_state->memory_pressure_tracker->memory_usage();
  int64_t memory_peak =
      sched_state->memory_pressure_tracker->pressure_state().memory_peak;

  if (schedule_proto_.has_value()) {
    sched_state->memory_trace[&n->GetInstr()] = {memory_after, memory_peak};
  }

  VLOG(10) << "Memory pressure after schedule: " << memory_after;
  VLOG(10) << "Memory peak after schedule: " << memory_peak;
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
      to_visit_queue.push_back(GetNodePtr(&edge.Target().GetInstr()));
    }
  }
  return false;
}

HloScheduleGraph::HloScheduleGraph(
    const std::vector<HloInstruction*>* post_order_instructions,
    std::shared_ptr<const SchedulingContext> scheduling_context)
    : original_order_(post_order_instructions->begin(),
                      post_order_instructions->end()),
      scheduling_context_(scheduling_context) {
  HloComputation* comp = (*post_order_instructions)[0]->parent();
  auto reachability = HloReachabilityMap::Build(comp);
  std::vector<const HloInstruction*> while_instrs;
  auto latency_estimator = scheduling_context->GetLatencyEstimator();
  auto async_tracker = scheduling_context->GetAsyncTracker();
  auto alias_analysis = scheduling_context->GetAliasAnalysis();

  const int n_inst = post_order_instructions->size();
  // Allocating the graph nodes. One for each of the instructions in the
  // original instructions order.
  node_storage_.resize(n_inst);

  // Single entry initially, which is a shared entry that is shared by
  // all instructions for which the rare data structures are all empty
  rare_storage_.push_back(std::make_unique<HloGraphNode::Rare>());

  // The first entry of sharable_resources_storage_ is an empty vector,
  // which can be shared by all edges that have an empty sharable resources
  // set.
  sharable_resources_storage_.push_back(std::vector<int64_t>());

  int64_t current_pos = 0;
  for (HloInstruction* instr : *post_order_instructions) {
    auto [new_node_it, inserted] = nodes_.try_emplace(instr, current_pos);
    CHECK(inserted) << "Expected the value to not be already in the map";
    HloGraphNode* n = &node_storage_[current_pos];
    DCHECK_EQ(n, GetNodePtr(instr));
    n->instr_ = instr;
    n->opcode_ = instr->opcode();
    n->original_position_ = current_pos;
    current_pos++;

    n->cost_ = latency_estimator->NodeCost(instr);
    auto resources_span = async_tracker->GetResourcesFromInstruction(*instr);
    ResourcesVector resources(resources_span.begin(), resources_span.end());
    HloGraphNode::Rare r;
    r.released_non_extendable_resources =
        async_tracker->GetReleasedNonextendableResourcesFromVector(resources);
    r.released_shareable_resources =
        async_tracker->GetReleasedShareableResourcesFromVector(resources);
    r.occupied_shareable_resources =
        async_tracker->GetOccupiedShareableResourcesFromVector(resources);
    r.recursive_resources =
        async_tracker->GetNumResourcesPerInstruction(*instr);
    r.resources = resources;

    n->has_recursive_resources_ = !(r.recursive_resources.empty());
    if (!r.recursive_resources.empty() ||                //
        !r.released_non_extendable_resources.empty() ||  //
        !r.released_shareable_resources.empty() ||       //
        !r.occupied_shareable_resources.empty() ||       //
        !r.resources.empty()) {
      // At least one entry is non-empty, so we need an actual
      rare_storage_.push_back(std::make_unique<HloGraphNode::Rare>(r));
      n->rare_ = rare_storage_.back().get();
      n->has_rare_ = true;
    } else {
      // The default entry where all are empty can be shared by this
      // instruction
      n->rare_ = rare_storage_[0].get();
      n->has_rare_ = false;
    }

    n->is_supported_async_done_ = async_tracker->IsSupportedAsyncDone(*instr);
    n->is_supported_async_start_ = async_tracker->IsSupportedAsyncStart(*instr);
    n->has_operand_that_is_supported_async_done_ = absl::c_any_of(
        instr->operands(), [async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    n->releases_selective_resource_ =
        async_tracker->ReleasesSelectiveResource(n);
    n->occupies_selective_resource_ =
        async_tracker->OccupiesSelectiveResource(n);
    n->does_occupy_any_resource_ =
        absl::c_any_of(resources, [](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceOccupy;
        });
    n->does_release_any_resource_ =
        absl::c_any_of(resources, [](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceRelease;
        });
    // Gather while instructions for subsequent send-done dependency checks.
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instrs.push_back(instr);
    }

    if (IsCustomCallWithForceEarlyAttribute(instr)) {
      n->SetForceEarly(true);
    }
    if (IsCustomCallWithForceDelayAttribute(instr)) {
      n->SetForceDelay(true);
      n->SetForceDelayPriority(GetCustomCallForceDelayPriority(instr));
    }
    if (n->IsSupportedAsyncStart() && HasForceDelayAsyncAttribute(instr)) {
      n->SetForceDelay(true);
    }
  }

  // num_predecessors[i]: number of predecessors for instruction number "i"
  std::vector<int> num_predecessors(n_inst, 0);
  // num_successors[i]: number of successors for instruction number "i"
  std::vector<int> num_successors(n_inst, 0);

  // We do two passes over the graph.  In phase 0, we just determine how
  // many successor and predecessor edges each node is likely to have (there are
  // some rare exceptions where the first pass can't fully see extra edges
  // added, but these are rare).  This first pass then allows us to lay out
  // storage for all the successor and predecessor edges for all the nodes in
  // the HloSchedulerGraph::predecessor_storage_ and
  // HloSchedulerGraph::successor_storage_ vectors.  This makes the cache
  // locality of traversing the graph via these edges better than if we
  // allocated them individually hanging off each of the HloGraphNode objects.
  //
  // In phase 1, we actually fill in the edge data into the storage arrays
  // we've allocated.
  for (int phase = 0; phase < 2; phase++) {
    // Add dependencies edges between each of the graph nodes.
    for (const HloInstruction* instr : *post_order_instructions) {
      HloGraphNode* instr_node = GetNodePtr(instr);
      // Lambda invoked for each edge or potential edge during phase 0 and phase
      // 1
      auto add_edge = [&](const char* type, HloGraphNode* from,
                          HloGraphNode* to, const LatencyEstimator* estimator) {
        const int from_index = from->original_position_;
        const int to_index = to->original_position_;
        if (phase == 0) {
          // In the first phase, we just count the number of predecessors and
          // successors
          num_successors[from_index]++;
          num_predecessors[to_index]++;
        } else {
          if (estimator != nullptr) {
            // Use overload of AddDependency that uses estimator to look up
            // latency
            HloGraphNode::AddDependency(from, to, estimator);
          } else {
            // Use overload of AddDependency that explicitly specifies latency
            // value
            HloGraphNode::AddDependency(from, to, 1);
          }
        }
      };

      VLOG(10) << "Adding users for " << instr_node->GetInstr().ToString();
      // Add edges that derive from def->use relationships of the HLO graph.
      for (const HloInstruction* user : instr->users()) {
        VLOG(10) << "\tUser: " << user->ToString();
        HloGraphNode* user_node = GetNodePtr(user);
        add_edge("user", instr_node, user_node, latency_estimator.get());
      }
      for (const HloInstruction* ctrl_succ : instr->control_successors()) {
        VLOG(10) << "\tCtrl Successor: " << ctrl_succ->ToString();
        HloGraphNode* ctrl_succ_node = GetNodePtr(ctrl_succ);
        add_edge("ctrl_succ", instr_node, ctrl_succ_node,
                 latency_estimator.get());
      }

      // To make sure an instruction that aliases with the buffer produced
      // by the async-done operation is not scheduled in between the start and
      // the done instruction as that buffer is in flux when the start
      // happens. Add an edge between this instruction and the start in this
      // case.
      if (instr_node->IsSupportedAsyncDone()) {
        const HloInstruction* async_start = instr->operand(0);
        if (alias_analysis != nullptr) {
          for (const HloBuffer* buffer :
               alias_analysis->ComputeBuffersAt(instr, {})) {
            for (const HloValue* value : buffer->values()) {
              if (value->defining_instruction() == instr) {
                continue;
              }
              for (const HloUse& use : value->GetUses()) {
                if (ContainsKey(nodes_, use.instruction)) {
                  // The instruction itself and later ones might be
                  // identified as use.instruction. Add checks here to avoid
                  // adding dependencies for these instructions.
                  if (use.instruction == async_start ||
                      reachability->IsReachable(instr, use.instruction)) {
                    continue;
                  }
                  HloGraphNode* pred_node = GetNodePtr(use.instruction);
                  HloGraphNode* start_node = GetNodePtr(async_start);
                  // Ignore token operands as they are not real aliasing.
                  if (use.instruction->operand(use.operand_number)
                          ->shape()
                          .IsToken()) {
                    continue;
                  }
                  if (phase == 0) {
                    // Be conservative and allocate space for a potential
                    // edge. The conditions below mean we might not actually
                    // add this edge, but some of the conditions depend on
                    // having some of the edges in the graph already set up,
                    // which won't really happen until phase 1.  We might
                    // allocate a bit of extra unused space in the edge
                    // storage vectors by being conservative here, but it's
                    // not a big deal.
                    add_edge("async0", pred_node, start_node,
                             nullptr /* unused for phase 0*/);
                    continue;
                  }
                  // If there is already a transitive link between the nodes
                  // the other way then skip adding this one.
                  if (IsPredecessorTransitively(pred_node, start_node)) {
                    continue;
                  }
                  add_edge("async1", pred_node, start_node,
                           nullptr /*==use latency of 1*/);
                }
              }
            }
          }
        }
      }
      // Add dependent edges from send-done operations to while loops which
      // are dependent on the recv-done control predecessor of the send-done.
      // This prevents send-done operations from being scheduled after
      // dependent while loops, which can caused send/recv overlap limits to
      // be violated.
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
            HloGraphNode* send_done_node = GetNodePtr(instr);
            HloGraphNode* while_node = GetNodePtr(dependent_while_instr);
            add_edge("while", send_done_node, while_node,
                     nullptr /*==use latency of 1*/);
          }
          break;
        }
      }
    }

    if (phase == 0) {
      // After phase 0 runs above, we now know an upper bound on the number of
      // successors and predecessors we will have, and so can allocate space
      // in the predecessors and successor storage vectors.

      int total_predecessors = 0;
      for (const auto& n : num_predecessors) {
        total_predecessors += n;
      }
      int total_successors = 0;
      for (const auto& n : num_successors) {
        total_successors += n;
      }
      predecessors_storage_.resize(total_predecessors);
      successors_storage_.resize(total_successors);

      // Set pointers to reserved ranges of predecessors_storage_ and
      // successors_storage_ within each node
      int next_index = 0;
      int pstart = 0;
      int sstart = 0;
      for (const HloInstruction* instr : *post_order_instructions) {
        HloGraphNode* n = GetNodePtr(instr);
        n->predecessors_.SetEmptyPointingToSharedSpace(
            (pstart < total_predecessors) ? &predecessors_storage_[pstart]
                                          : nullptr,
            num_predecessors[next_index]);
        n->successors_.SetEmptyPointingToSharedSpace(
            (sstart < total_successors) ? &successors_storage_[sstart]
                                        : nullptr,
            num_successors[next_index]);
        pstart += num_predecessors[next_index];
        sstart += num_successors[next_index];
        next_index++;
      }
      CHECK_EQ(next_index, post_order_instructions->size());
    }
  }

  // Initialize ready_nodes_if_scheduled_ for all nodes.
  for (auto& node : node_storage_) {
    node.UpdateReadyNodesIfScheduled();
  }

  // Post process the schedule graph based on the supplied async_tracker.
  async_tracker->PostProcessScheduleGraph(this, latency_estimator.get());
}

std::string HloScheduleGraph::ToString() const {
  std::string result;
  std::vector<std::pair<const HloGraphNode*, int>> stack;
  for (auto& node : node_storage_) {
    if (node.GetPredecessors().empty()) {
      stack.push_back(std::make_pair(&node, 0));
    }
  }
  std::vector<const HloGraphNode*> order;
  absl::flat_hash_set<const HloGraphNode*> visited;
  while (!stack.empty()) {
    auto& val = stack.back();
    if (val.second == val.first->GetSuccessors().size()) {
      order.push_back(val.first);
      stack.pop_back();
      continue;
    }
    const int64_t next_child = val.second++;
    if (visited.insert(&val.first->GetSuccessors()[next_child].Target())
            .second) {
      stack.push_back(
          std::make_pair(&val.first->GetSuccessors()[next_child].Target(), 0));
    }
  }
  for (auto it = order.rbegin(), e = order.rend(); it != e; ++it) {
    absl::StrAppend(
        &result, (*it)->ToString(scheduling_context_->GetAsyncTracker().get()));
  }
  return result;
}

HloGraphNode& HloScheduleGraph::GetNode(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return node_storage_[it->second];
}

HloGraphNode* HloScheduleGraph::GetNodePtr(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return &node_storage_[it->second];
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

void HloScheduleGraph::InitializeGraphAnalysis() {
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
    if (scheduling_context_->GetAsyncTracker()->OccupiesSelectiveResource(
            node)) {
      node->num_hops_to_closest_selective_resource_occupier_ = 0;
    } else {
      int64_t closest_predecessor_distance =
          std::numeric_limits<int64_t>::max();
      for (auto& pred : node->GetPredecessors()) {
        closest_predecessor_distance = std::min<int64_t>(
            closest_predecessor_distance,
            pred.Target().num_hops_to_closest_selective_resource_occupier_);
      }
      if (closest_predecessor_distance != std::numeric_limits<int64_t>::max()) {
        node->num_hops_to_closest_selective_resource_occupier_ =
            closest_predecessor_distance + 1;
      }
    }
    if (node->IsSupportedAsyncDone()) {
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
      CHECK_OK(node.SetAnnotation(annotation));
    }
  }
}

absl::Status DefaultSchedulerCore::InitializeScheduler(
    const HloModule* module) {
  module_ = module;
  module_pressure_state_ = std::make_unique<ModulePressureState>(
      module, scheduling_context_->GetAliasAnalysis().get(),
      scheduling_context_->GetShapeSizeBytes());

  module_pressure_state_->InitializePressureStates();
  module_pressure_state_->SetMemoryPeak(0);
  annotation_tracker_ = std::make_unique<AnnotationTracker>(module);
  if (VLOG_IS_ON(2)) {
    annotation_tracker_->PrintAnnotationSets(2);
  }

  if (!scheduling_instruction_crosses_overlap_limit_) {
    scheduling_instruction_crosses_overlap_limit_ =
        [](const SchedulingState& sched_state, const HloGraphNode* node) {
          if (!node->HasRecursiveResources()) {
            return false;
          }
          auto& num_resources_needed = node->GetRecursiveResources();
          for (const auto& [resource, count] : num_resources_needed) {
            auto it = sched_state.max_concurrent_resource.find(resource);
            if (it == sched_state.max_concurrent_resource.end()) {
              continue;
            }
            if (count > it->second) {
              VLOG(5) << "Cross overlap limit for resource: " << resource
                      << " count: " << count << " limit: " << it->second;
              return true;
            }
          }
          return false;
        };
    is_default_scheduling_instruction_crosses_overlap_limit_ = true;
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
    const SchedulingState& sched_state, int64_t annotation,
    bool get_max_resources) {
  absl::flat_hash_map<int64_t, int64_t> num_resources_needed;
  const HloComputation* comp =
      sched_state.sched_graph.GetOriginalInstrList()[0]->parent();
  auto instrs = annotation_tracker_->GetInstructions(comp, annotation);
  CHECK(!instrs.empty());
  if (HasKeepOriginalSequenceOrderInGroupAttribute(instrs[0])) {
    return GetNumResourcesNeededForAnnotationWithKeepOriginalOrderAttrs(
        sched_state, instrs);
  }
  for (const HloInstruction* instr : instrs) {
    auto num_resources_needed_per_instr =
        sched_state.async_tracker->GetNumResourcesPerInstruction(*instr);
    for (const auto& [resource, usage] : num_resources_needed_per_instr) {
      if (instr->opcode() == HloOpcode::kAsyncDone) {
        // There are two cases where the resources used by the async-done op
        // need to be accumulated:
        // 1. if a async-done op's matching start op is not in the
        // same annotation group, then the live range of the resources used
        // by this async-done op extends beyond this annotation group.
        // 2. if get_max_resources is true, then we compute the resource usage
        // assuming maximum overlapping, where the resources used by the
        // async-done ops need to be accumulated.
        const HloInstruction* start = instr->operand(0);
        if (absl::c_find(instrs, start) == instrs.end() || get_max_resources) {
          num_resources_needed[resource] += usage;
          continue;
        }
      }
      // The minimum number of resources needed for the annotation group is
      // the maximum number of resources needed for any instruction in the
      // group.
      num_resources_needed[resource] =
          std::max(num_resources_needed[resource], usage);
    }
  }
  return num_resources_needed;
}

int64_t DefaultSchedulerCore::GetNumSuccessorsForAnnotation(
    const SchedulingState& sched_state, int64_t annotation) const {
  const HloComputation* comp =
      sched_state.sched_graph.GetOriginalInstrList()[0]->parent();
  int64_t num_successors = 0;
  std::vector<const HloInstruction*> instrs =
      annotation_tracker_->GetInstructions(comp, annotation);
  absl::flat_hash_set<const HloInstruction*> seen_instrs(instrs.begin(),
                                                         instrs.end());
  for (const HloInstruction* instr : instrs) {
    for (const HloEdge& edge :
         sched_state.sched_graph.GetNode(instr).GetSuccessors()) {
      const HloGraphNode& user = edge.Target();
      if (seen_instrs.insert(&user.GetInstr()).second &&
          (user.GetAnnotation() != annotation)) {
        ++num_successors;
      }
    }
  }
  return num_successors;
}

bool DefaultSchedulerCore::SchedulingAnnotationCrossesOverlapLimit(
    const SchedulingState& sched_state, int64_t annotation,
    bool use_max_resources) {
  absl::flat_hash_map<int64_t, int64_t> num_resources_needed =
      GetNumResourcesNeededForAnnotation(sched_state, annotation,
                                         use_max_resources);
  for (const auto& [resource, num_needed] : num_resources_needed) {
    int64_t limit = sched_state.max_concurrent_resource.at(resource);
    if (num_needed > limit) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<bool> DefaultSchedulerCore::TryScheduleOneAnnotationGroup(
    DefaultSchedulerCore::SchedulingState* sched_state,
    const HloComputation* computation, bool use_max_resources) {
  if (sched_state->ready_annotations.empty() ||
      !sched_state->nodes_holding_annotations.empty()) {
    return false;
  }
  // Pick the first ready annotation whose scheduling will not cross the
  // overlap limit. If there is no such annotation, continue with
  // scheduling non-annotated ops.
  int64_t annotation_index = -1;
  for (int64_t i = 0; i < sched_state->ready_annotations.size(); ++i) {
    if (SchedulingAnnotationCrossesOverlapLimit(
            *sched_state, sched_state->ready_annotations[i],
            /*use_max_resources=*/use_max_resources)) {
      continue;
    }
    annotation_index = i;
    break;
  }
  if (annotation_index != -1) {
    std::swap(sched_state->ready_annotations[annotation_index],
              sched_state->ready_annotations.back());
    int64_t annotation = sched_state->ready_annotations.back();
    sched_state->ready_annotations.pop_back();
    VLOG(2) << "------- BEGIN ANNOTATION: " << annotation << " -------";
    sched_state->ongoing_annotation = annotation;
    TF_RETURN_IF_ERROR(
        ScheduleAnnotation(computation, annotation, sched_state));
    VLOG(2) << "-------  END ANNOTATION: " << annotation << " --------";
    sched_state->ongoing_annotation = -1;
    return true;
  }
  return false;
}

absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
DefaultSchedulerCore::MakeSchedulingState(const HloComputation* computation) {
  const HloSchedule& module_schedule = computation->parent()->schedule();
  std::unique_ptr<MemoryPressureTracker> memory_pressure_tracker =
      std::make_unique<MemoryPressureTracker>(
          scheduling_context_->GetAliasAnalysis().get(),
          module_pressure_state_->buffer_tracker(),
          module_pressure_state_->pressure_state_cache());
  memory_pressure_tracker->Initialize(
      computation,
      module_pressure_state_->GetPressureStateForComputation(computation)
          .live_ids_at_bottom);
  std::shared_ptr<SchedulingState> sched_state =
      std::make_shared<SchedulingState>(
          &module_schedule.sequence(computation), scheduling_context_,
          std::move(memory_pressure_tracker), config_);
  sched_state->sched_graph.InitializeGraphAnalysis();
  return sched_state;
}

absl::StatusOr<std::vector<HloInstruction*>>
DefaultSchedulerCore::ScheduleComputation(const HloComputation* computation) {
  TF_ASSIGN_OR_RETURN(auto sched_state, MakeSchedulingState(computation));
  return ScheduleComputation(computation, sched_state);
}

absl::StatusOr<std::vector<HloInstruction*>>
DefaultSchedulerCore::ScheduleComputation(
    const HloComputation* computation,
    std::shared_ptr<SchedulerCore::SchedulingState> _sched_state) {
  // Up-cast the scheduling state DefaultSchedulerCore::SchedulingState.
  std::shared_ptr<DefaultSchedulerCore::SchedulingState> sched_state =
      std::dynamic_pointer_cast<DefaultSchedulerCore::SchedulingState>(
          _sched_state);

  CHECK_NE(sched_state, nullptr)
      << "ScheduleComputation must be called with a "
      << "DefaultSchedulerCore::SchedulingState object.";

  // Reset the scheduling graph.
  sched_state->sched_graph.InitializeGraphAnalysis();
  sched_state->sched_graph.ResetScheduling();

  sched_state->memory_pressure_tracker->Reset(
      computation,
      module_pressure_state_->GetPressureStateForComputation(computation)
          .live_ids_at_bottom);

  if (graph_processing_hook_) {
    TF_RETURN_IF_ERROR(graph_processing_hook_(&sched_state->sched_graph));
  }

  VLOG(5) << "Just built graph:";

  auto& memory_pressure_tracker = *sched_state->memory_pressure_tracker;

  if (annotation_tracker_->HasAnnotations(computation)) {
    sched_state->sched_graph.AnnotateGraph(annotation_tracker_.get());
    for (int64_t annotation :
         annotation_tracker_->GetAnnotations(computation)) {
      int64_t num_successors =
          GetNumSuccessorsForAnnotation(*sched_state, annotation);
      sched_state->num_successors_for_annotation[annotation].all =
          num_successors;
      if (num_successors == 0) {
        VLOG(3) << "Annotation " << annotation
                << " does not have any successors, is ready to be scheduled";
        sched_state->ready_annotations.push_back(annotation);
      }
    }
  }

  XLA_VLOG_LINES(5, sched_state->sched_graph.ToString());
  scheduling_context_->GetAsyncTracker()->SetConcurrentResourceLimits(
      sched_state->max_concurrent_resource);
  // Collect the bottom roots of the graph (nodes that don't have any
  // successor)
  // We are going to use them as starting point for scheduling.
  auto roots = sched_state->sched_graph.FindBottomRoots();
  for (HloGraphNode* root : roots) {
    // Set ready time for the roots 0.
    root->SetReadyTime(0.0);
  }
  VLOG(5) << "Initial memory pressure for " << computation->name() << ": "
          << memory_pressure_tracker.memory_usage();
  sched_state->ready_set.insert(sched_state->ready_set.end(), roots.begin(),
                                roots.end());
  // Schedule in order bottom up.
  while (!sched_state->ready_set.empty() || !sched_state->nop_set.empty()) {
    VLOG(10) << "Current ready time: " << sched_state->current_time;
    VLOG(2) << "Current ready queue:";
    XLA_VLOG_LINES(2, [&sched_state]() {
      struct LogFormatter {
        void operator()(std::string* out, const HloGraphNode* n) const {
          absl::StrAppend(out, "\t", n->GetInstr().name(),
                          " Ready time: ", n->GetReadyTime(),
                          " Depth: ", n->GetGraphDepth());
        }
      };
      return absl::StrJoin(sched_state->ready_set, "\n", LogFormatter());
    }());
    auto scheduled_with_max_resources = TryScheduleOneAnnotationGroup(
        sched_state.get(), computation, /*use_max_resource*/ true);
    if (!scheduled_with_max_resources.ok()) {
      return scheduled_with_max_resources.status();
    }
    if (*scheduled_with_max_resources) {
      continue;
    }
    auto scheduling_step_status = SchedulingStep(sched_state.get());
    // If we cannot schedule any non-annotated ops, try scheduling any of the
    // ready annotation groups again using minimum resources.
    if (!scheduling_step_status.ok()) {
      VLOG(3) << "Failed to schedule any non-annotated ops, trying again with "
                 "minimum resources for annotation groups";
      auto scheduled_with_min_resources = TryScheduleOneAnnotationGroup(
          sched_state.get(), computation, /*use_max_resource*/ false);
      if (!scheduled_with_min_resources.ok()) {
        return scheduled_with_min_resources.status();
      }
      if (*scheduled_with_min_resources) {
        continue;
      }
      VLOG(3)
          << "Failed to schedule any annotation groups with minimum resources";
      return scheduling_step_status;
    }
  }

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "New order";
    for (auto r_it = sched_state->new_sequence_reversed.rbegin(),
              e_it = sched_state->new_sequence_reversed.rend();
         r_it != e_it; ++r_it) {
      LogInstruction(*r_it);
    }
  }

  module_pressure_state_->UpdatePressureStateForComputation(
      computation, memory_pressure_tracker.pressure_state());
  absl::c_reverse(sched_state->new_sequence_reversed);
  if (post_processing_fn_) {
    post_processing_fn_(*sched_state);
  }
  CHECK_EQ(sched_state->new_sequence_reversed.size(),
           sched_state->sched_graph.GetOriginalInstrList().size())
      << "Not all instructions have been scheduled "
      << sched_state->new_sequence_reversed.size() << " vs "
      << sched_state->sched_graph.GetOriginalInstrList().size();
  VLOG(2) << "Total time: "
          << sched_state->sched_graph
                 .GetNode(sched_state->new_sequence_reversed.front())
                 .GetReadyTime();

  if (schedule_proto_.has_value()) {
    *schedule_proto_->add_computation_schedules() = ComputationScheduleToProto(
        computation, *sched_state, *scheduling_context_->GetLatencyEstimator(),
        sched_state->new_sequence_reversed);
  }
  return std::move(sched_state->new_sequence_reversed);
}

ScheduleProto::ComputationScheduleProto
DefaultSchedulerCore::ComputationScheduleToProto(
    const HloComputation* computation, const SchedulingState& sched_state,
    const LatencyEstimator& estimator,
    const std::vector<HloInstruction*>& instructions) {
  const HloScheduleGraph& schedule_graph = sched_state.sched_graph;
  ScheduleProto::ComputationScheduleProto proto;
  proto.set_computation_id(computation->unique_id());
  proto.set_cycles_per_microsecond(estimator.CyclesPerMicrosecond());
  *proto.mutable_scheduler_statistics() =
      LatencyHidingScheduler::LatencyHidingStatistics(computation,
                                                      scheduling_context_)
          .ToProto();

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

    auto it = sched_state.memory_trace.find(instr);
    if (it != sched_state.memory_trace.end()) {
      instr_msg->set_memory_usage_after(it->second.first);
      instr_msg->set_peak_memory_after(it->second.second);
    }
  }
  return proto;
}

LatencyHidingScheduler::SchedulerStatistics
LatencyHidingScheduler::LatencyHidingStatistics(
    const HloComputation* computation,
    std::shared_ptr<const SchedulingContext> scheduling_context,
    const ModulePressureState* module_pressure_state,
    MemoryPressureTracker* memory_pressure_tracker) {
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
    kCall,
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
      case HloOpcode::kCall:
        return AsyncKind::kCall;
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
  auto find_outstanding_async = [&outstanding_collectives, scheduling_context](
                                    const HloInstruction* instr) {
    const auto& collective_vec =
        outstanding_collectives[scheduling_context->GetAsyncTracker()
                                    ->GetCanonicalAsyncOp(*instr)
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
  auto instructions_post_order = computation->MakeInstructionPostOrder();
  HloScheduleGraph schedule_graph(&instructions_post_order, scheduling_context);
  int64_t curr_pos = 0;
  for (const HloInstruction* instr :
       module->schedule().sequence(computation).instructions()) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    current_time += instr_node.GetCost();
    if (instr_node.IsSupportedAsyncStart()) {
      outstanding_collectives[scheduling_context->GetAsyncTracker()
                                  ->GetCanonicalAsyncOp(*instr)
                                  .inner]
          .push_back({instr, current_time, curr_pos});
    } else if (instr_node.IsSupportedAsyncDone()) {
      const HloInstruction* start_instr = instr->operand(0);
      // TODO(b/329731042): Handle pipelined Send/Recv in while-body, which
      // is the only situation where an async done operand is not an async
      // start.
      if (scheduling_context->GetAsyncTracker()->IsSupportedAsyncStart(
              *start_instr)) {
        auto it = find_outstanding_async(start_instr);
        const HloGraphNode& start_node =
            schedule_graph.GetNode(std::get<0>(*it));
        auto edge_it = find_node_successor_edge(start_node, instr_node);
        const double async_wasted_cycles = std::max(
            0.0, edge_it->Latency() - (current_time - std::get<1>(*it)));
        AsyncKind kind =
            opcode_to_async_kind(scheduling_context->GetAsyncTracker()
                                     ->GetCanonicalAsyncOp(*start_instr)
                                     .inner);
        wasted_time_per_collective[kind] += async_wasted_cycles;
        current_time += async_wasted_cycles;
      }
    }
    curr_pos++;
  }
  // Check if the optional arguments alias_analysis and module_pressure_state
  // are null. If so, create a new instance of them.
  std::unique_ptr<ModulePressureState> module_pressure_state_ptr;
  if (module_pressure_state == nullptr) {
    module_pressure_state_ptr = std::make_unique<ModulePressureState>(
        module, scheduling_context->GetAliasAnalysis().get(),
        scheduling_context->GetShapeSizeBytes());
    module_pressure_state_ptr->InitializePressureStates();
    module_pressure_state = module_pressure_state_ptr.get();
  }
  bool memory_tracked =
      module_pressure_state->ComputationIsMemoryTracked(computation);
  const MemoryPressureTracker::MemoryPressureState& computation_pressure_state =
      module_pressure_state->GetPressureStateForComputation(computation);
  const MemoryPressureTracker::MemoryPressureState* memory_pressure_state =
      memory_tracked ? &computation_pressure_state : nullptr;
  std::unique_ptr<MemoryPressureTracker> memory_pressure_tracker_ptr;
  if (memory_pressure_tracker == nullptr) {
    memory_pressure_tracker_ptr = std::make_unique<MemoryPressureTracker>(
        scheduling_context->GetAliasAnalysis().get(),
        module_pressure_state->buffer_tracker(),
        module_pressure_state->pressure_state_cache());
    if (memory_pressure_state != nullptr) {
      memory_pressure_tracker_ptr->Initialize(
          computation, memory_pressure_state->live_ids_at_bottom);
    }
    memory_pressure_tracker = memory_pressure_tracker_ptr.get();
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
      /*call_wasted_cycles=*/wasted_time_per_collective[AsyncKind::kCall],
      /*total_cycles=*/current_time,
      /*memory_pressure_peak=*/
      memory_pressure_state
          ? memory_pressure_tracker->initial_memory_pressure() +
                memory_pressure_state->memory_peak
          : 0};
}

// Prints a SchedulerStatistics object.
std::string LatencyHidingScheduler::SchedulerStatistics::ToString() const {
  std::string result;
  if (const HloComputation* comp = this->computation) {
    absl::StrAppend(&result, "For computation: ", comp->name(), ", module ",
                    comp->parent()->name(), "(", comp->parent()->unique_id(),
                    ")\n");
  }
  absl::StrAppend(&result,
                  "Total wasted cycles: ", this->GetTotalWastedCycles(), "\n");
  absl::StrAppend(&result, "Wasted cycles for all-reduce: ",
                  this->all_reduce_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-gather: ",
                  this->all_gather_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-broadcast: ",
                  this->collective_broadcast_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-permute: ",
                  this->collective_permute_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-to-all: ",
                  this->all_to_all_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for ragged-all-to-all: ",
                  this->ragged_all_to_all_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for reduce-scatter: ",
                  this->reduce_scatter_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for send: ", this->send_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Wasted cycles for recv: ", this->recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Wasted cycles for asynchronous call: ",
                  this->call_wasted_cycles, "\n");
  absl::StrAppend(&result, "Total cycles: ", this->total_cycles, "\n");
  absl::StrAppend(&result,
                  "Memory pressure peak (bytes): ", this->memory_pressure_peak,
                  "\n");
  return result;
}
ScheduleProto::SchedulerStatisticsProto
LatencyHidingScheduler::SchedulerStatistics::ToProto() const {
  ScheduleProto::SchedulerStatisticsProto proto;
  proto.set_all_gather_wasted_cycles(all_gather_wasted_cycles);
  proto.set_all_reduce_wasted_cycles(all_reduce_wasted_cycles);
  proto.set_collective_broadcast_wasted_cycles(
      collective_broadcast_wasted_cycles);
  proto.set_collective_permute_wasted_cycles(collective_permute_wasted_cycles);
  proto.set_all_to_all_wasted_cycles(all_to_all_wasted_cycles);
  proto.set_ragged_all_to_all_wasted_cycles(ragged_all_to_all_wasted_cycles);
  proto.set_reduce_scatter_wasted_cycles(reduce_scatter_wasted_cycles);
  proto.set_send_wasted_cycles(send_wasted_cycles);
  proto.set_recv_wasted_cycles(recv_wasted_cycles);
  proto.set_call_wasted_cycles(call_wasted_cycles);
  proto.set_total_wasted_cycles(this->GetTotalWastedCycles());
  proto.set_total_cycles(total_cycles);
  proto.set_memory_pressure_peak(memory_pressure_peak);
  return proto;
}

void LatencyHidingScheduler::LogScheduleStatistics(
    const HloComputation* computation) {
  XLA_VLOG_LINES(
      1, LatencyHidingStatistics(computation, scheduling_context_).ToString());
}

absl::StatusOr<std::pair<std::vector<HloInstruction*>, ComputationScheduleInfo>>
LatencyHidingScheduler::ScheduleWithPreferences(
    HloModule* module, const std::vector<double>& preferences,
    const HloComputation* computation) {
  auto set_preferences = [&](HloScheduleGraph* graph) -> absl::Status {
    VLOG(3) << "Setting scheduling preferences.";
    graph->SetPreferences(preferences);
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(scheduler_core_->SetGraphProcessingHook(set_preferences));
  TF_ASSIGN_OR_RETURN(auto new_schedule,
                      scheduler_core_->ScheduleComputation(computation));

  // Save the old schedule.
  auto old_schedule = std::vector<HloInstruction*>(
      module->schedule().sequence(computation).instructions());
  // Temporarily use the new schedule to capture stats.
  module->schedule().set_sequence(computation,
                                  absl::MakeConstSpan(new_schedule));
  LatencyHidingScheduler::SchedulerStatistics stats =
      LatencyHidingStatistics(computation, scheduling_context_);
  // Restore the old schedule.
  module->schedule().set_sequence(computation,
                                  absl::MakeConstSpan(old_schedule));

  ComputationScheduleInfo schedule_info;
  schedule_info.total_wasted_cycles = stats.GetTotalWastedCycles();

  // Return the peak memory of this computation instead of the whole module
  // to allow heuristic to optimize this functions memory usage.
  schedule_info.peak_memory = stats.memory_pressure_peak;

  return std::make_pair(new_schedule, schedule_info);
}

absl::StatusOr<bool> LatencyHidingScheduler::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "Original module:";
  XLA_VLOG_LINES(5, module->ToString());
  // Currently we expect that a schedule that minimizes memory pressure is
  // provided as a base. It's not necessary for the algorithm itself but it
  // allows us to not having to think for now about memory pressure.
  CHECK(module->has_schedule()) << "LatencyHidingScheduler expects a base "
                                   "schedule that minimizes memory pressure.";
  computations_to_schedule_.reserve(module->computation_count());
  // Collect which computations have latency hiding opportunities.
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (auto* instr : computation->instructions()) {
      if (scheduling_context_->GetAsyncTracker()->IsSupportedAsyncStart(
              *instr) ||
          scheduling_context_->GetAsyncTracker()->IsSupportedAsyncDone(
              *instr) ||
          IsCustomCallWithForceDelayAttribute(instr)) {
        computations_to_schedule_.push_back(computation);
        break;
      }
    }
  }

  VLOG(2) << "Computations to schedule " << computations_to_schedule_.size()
          << " size HloGraphNode: " << sizeof(HloGraphNode) << " size HloEdge "
          << sizeof(HloEdge);
  if (computations_to_schedule_.empty()) {
    return false;
  }
  TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
  const auto& debug_options = module->config().debug_options();
  if (debug_options.xla_dump_latency_hiding_schedule()) {
    TF_RETURN_IF_ERROR(scheduler_core_->CaptureScheduleProto());
  }
  if (VLOG_IS_ON(1)) {
    // Log the statistics before scheduling. We batch the per-computation
    // statistics to speed up the calculation.
    ModulePressureState pressure_state = ModulePressureState(
        module, scheduling_context_->GetAliasAnalysis().get(),
        scheduling_context_->GetShapeSizeBytes());
    pressure_state.InitializePressureStates();
    for (HloComputation* computation : computations_to_schedule_) {
      VLOG(1) << "[" << name() << "] Statistics before scheduling:";
      XLA_VLOG_LINES(1, LatencyHidingStatistics(
                            computation, scheduling_context_, &pressure_state)
                            .ToString());
    }
  }
  for (HloComputation* computation : computations_to_schedule_) {
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        scheduler_core_->ScheduleComputation(computation));
    // Update target specific states that may include altering the
    // computation.
    scheduling_context_->GetAsyncTracker()->UpdateTargetDefinedStates(
        computation);
    module->schedule().set_sequence(computation,
                                    absl::MakeConstSpan(new_schedule));
    scheduling_context_->GetAsyncTracker()->ResetTargetDefinedStates();
    scheduling_context_->GetAsyncTracker()->InvalidateCache(computation);
  }
  int64_t fragmentation_size =
      scheduling_context_->GetAsyncTracker()
              ->GetConfig()
              .estimate_fragmentation_size
          ? EstimateFragmentationSize(module,
                                      *scheduling_context_->GetAliasAnalysis(),
                                      scheduling_context_->GetAliasInfo())
          : 0;
  uint64_t initial_memory_limit = scheduler_core_->GetMemoryLimit();
  for (int64_t iter = 0; iter < scheduler_core_->GetRerunTimes() &&
                         scheduler_core_->GetMemoryPeak() + fragmentation_size >
                             initial_memory_limit;
       iter++) {
    LOG(INFO) << "LatencyHidingScheduler current memory usage: "
              << scheduler_core_->GetMemoryPeak() + fragmentation_size
              << " bytes, does not fit in initial limit: "
              << initial_memory_limit << ". Setting the new limit to "
              << static_cast<uint64_t>(scheduler_core_->GetMemoryLimit() * 0.9);
    TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
    scheduler_core_->SetMemoryLimit(scheduler_core_->GetMemoryLimit() * 0.9);
    for (HloComputation* computation : computations_to_schedule_) {
      TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                          scheduler_core_->ScheduleComputation(computation));
      scheduling_context_->GetAsyncTracker()->UpdateTargetDefinedStates(
          computation);
      module->schedule().set_sequence(computation,
                                      absl::MakeConstSpan(new_schedule));
      scheduling_context_->GetAsyncTracker()->ResetTargetDefinedStates();
      scheduling_context_->GetAsyncTracker()->InvalidateCache(computation);
    }
    fragmentation_size =
        scheduling_context_->GetAsyncTracker()
                ->GetConfig()
                .estimate_fragmentation_size
            ? EstimateFragmentationSize(
                  module, *scheduling_context_->GetAliasAnalysis(),
                  scheduling_context_->GetAliasInfo())
            : 0;
  }
  LOG(INFO) << "[" << name() << "]"
            << " LatencyHidingScheduler current memory usage: "
            << scheduler_core_->GetMemoryPeak()
            << " bytes. Current limit: " << scheduler_core_->GetMemoryLimit();
  if (VLOG_IS_ON(1)) {
    // Log the statistics after scheduling.
    ModulePressureState post_scheduling_pressure_state = ModulePressureState(
        module, scheduling_context_->GetAliasAnalysis().get(),
        scheduling_context_->GetShapeSizeBytes());
    post_scheduling_pressure_state.InitializePressureStates();
    for (HloComputation* computation : computations_to_schedule_) {
      VLOG(1) << "[" << name() << "] Statistics after scheduling:";
      XLA_VLOG_LINES(1,
                     LatencyHidingStatistics(computation, scheduling_context_,
                                             &post_scheduling_pressure_state)
                         .ToString());
    }
  }
  if (debug_options.xla_dump_latency_hiding_schedule()) {
    TF_ASSIGN_OR_RETURN(ScheduleProto proto,
                        scheduler_core_->GetCapturedScheduleProto());
    const std::string filename = absl::StrFormat("%s.schedule", module->name());
    DumpProtobufToFile(proto, debug_options, filename);
  }
  return true;
}

}  // namespace xla
