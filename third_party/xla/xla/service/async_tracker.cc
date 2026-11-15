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

#include "xla/service/async_tracker.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/latency_estimator.h"
#include "xla/util.h"

namespace xla {

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
                                  ? (config_.top_down_scheduling
                                         ? ResourceUsageType::kResourceOccupy
                                         : ResourceUsageType::kResourceRelease)
                                  : (config_.top_down_scheduling
                                         ? ResourceUsageType::kResourceRelease
                                         : ResourceUsageType::kResourceOccupy);
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
                    config_.top_down_scheduling
                        ? ResourceUsageType::kResourceOccupy
                        : ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceOccupy
                                   : ResourceUsageType::kResourceRelease)};
    case HloOpcode::kSend:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(&hlo)->is_host_transfer()
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceOccupy
                                   : ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceOccupy
                                   : ResourceUsageType::kResourceRelease)};
    case HloOpcode::kRecvDone: {
      const HloSendRecvInstruction* recv =
          DynCast<HloSendRecvInstruction>(hlo.operand(0));
      return ResourcesVector{
          (recv != nullptr && recv->is_host_transfer())
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    config_.top_down_scheduling
                        ? ResourceUsageType::kResourceRelease
                        : ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceRelease
                                   : ResourceUsageType::kResourceOccupy)};
    }
    case HloOpcode::kSendDone: {
      const HloSendRecvInstruction* send =
          DynCast<HloSendRecvInstruction>(hlo.operand(0));
      return ResourcesVector{
          (send != nullptr && send->is_host_transfer())
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceRelease
                                   : ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               config_.top_down_scheduling
                                   ? ResourceUsageType::kResourceRelease
                                   : ResourceUsageType::kResourceOccupy)};
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
  {
    absl::ReaderMutexLock lock(&resources_cache_mu_);
    auto it = resources_cache_.find(&hlo);
    if (it != resources_cache_.end()) {
      return *(it->second);
    }
  }
  absl::WriterMutexLock lock(&resources_cache_mu_);
  auto& val = resources_cache_[&hlo];
  if (val == nullptr) {
    val =
        std::make_unique<ResourcesVector>(GetResourcesFromInstructionImpl(hlo));
  }
  return *val;
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

  // Only the start of an async op reports its async computation as a called
  // computation; account for it at the update and done of the async op as
  // well, so that the resources used inside the async op are checked at both
  // ends of it.
  // TODO(deqiangc): Converge this behaviour with GetNumResourcesPerInstruction.
  std::vector<const HloComputation*> nested_computations(
      instr.called_computations().begin(), instr.called_computations().end());
  if (instr.IsAsynchronous() && nested_computations.empty()) {
    nested_computations.push_back(instr.async_wrapped_computation());
  }

  // For instructions calling multiple computations, we assume that the called
  // computations do not execute in parallel (i.e., they are either mutually
  // exclusive, as in conditionals, or executed in sequence). Then for each
  // resource type, the usage across all called computations is the maximum
  // usage in any of the called computations.
  for (const HloComputation* computation : nested_computations) {
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
  {
    absl::ReaderMutexLock lock(&async_in_computation_cache_mu_);
    auto it = async_in_computation_cache_.find(computation);
    if (it != async_in_computation_cache_.end()) {
      return *(it->second);
    }
  }
  auto m = std::make_unique<absl::flat_hash_map<int64_t, int64_t>>();
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
      for (const auto& [type, usage] :
           RecursivelyComputeResourceMap(called_comp)) {
        if (seen_resources_per_comp.contains(type)) {
          continue;
        }
        (*m)[type] += usage;
        seen_resources_per_comp.insert(type);
      }
    }
  }
  absl::WriterMutexLock lock(&async_in_computation_cache_mu_);
  auto& per_opcode_map = async_in_computation_cache_[computation];
  if (per_opcode_map == nullptr) {
    per_opcode_map = std::move(m);
  }
  return *per_opcode_map;
}

const absl::flat_hash_map<int64_t, int64_t>&
AsyncTracker::RecursivelyComputeResourceMapForScheduledComputation(
    const HloComputation* computation) const {
  auto& schedule = computation->parent()->schedule();
  CHECK(schedule.is_computation_scheduled(computation));
  {
    absl::ReaderMutexLock lock(&async_in_computation_cache_mu_);
    auto it = async_in_computation_cache_.find(computation);
    if (it != async_in_computation_cache_.end()) {
      return *(it->second);
    }
  }
  auto m = std::make_unique<absl::flat_hash_map<int64_t, int64_t>>();
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
    if (inst->called_computations().empty()) {
      continue;
    }
    for (const auto& [type, usage] : GetNumResourcesPerInstruction(*inst)) {
      int64_t current_usage = inflight_resource_usage[type] + usage;
      int64_t& max_usage = res_map[type];
      max_usage = std::max(max_usage, current_usage);
    }
  }
  absl::WriterMutexLock lock(&async_in_computation_cache_mu_);
  auto& per_opcode_map = async_in_computation_cache_[computation];
  if (per_opcode_map == nullptr) {
    per_opcode_map = std::move(m);
  }
  return *per_opcode_map;
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

}  // namespace xla
