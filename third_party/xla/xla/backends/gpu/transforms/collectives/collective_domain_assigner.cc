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

#include "xla/backends/gpu/transforms/collectives/collective_domain_assigner.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/side_effect_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<absl::flat_hash_set<CollectiveCommunicationDomain>>
ParseDomainsToAssign(absl::string_view value) {
  absl::flat_hash_set<CollectiveCommunicationDomain> domains;
  for (absl::string_view domain_name :
       absl::StrSplit(value, ',', absl::SkipWhitespace())) {
    domain_name = absl::StripAsciiWhitespace(domain_name);
    ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain domain,
                     ParseCollectiveCommunicationDomain(domain_name));
    if (domain != kUnspecifiedCollectiveDomain) {
      domains.insert(domain);
    }
  }
  return domains;
}

absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetCollectiveParticipantGroups(const HloInstruction& collective,
                               const DeviceAssignment& device_assignment) {
  ABSL_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                   GetCollectiveOpGroupMode(&collective));

  if (HloPredicateIsOp<HloOpcode::kCollectivePermute,
                       HloOpcode::kCollectivePermuteStart>(&collective)) {
    std::vector<ReplicaGroup> source_target_groups;
    source_target_groups.reserve(collective.source_target_pairs().size());
    for (const auto& [source, target] : collective.source_target_pairs()) {
      ReplicaGroup& group = source_target_groups.emplace_back();
      group.add_replica_ids(source);
      group.add_replica_ids(target);
    }
    return GetParticipatingDevicesGroups(device_assignment,
                                         source_target_groups, group_mode);
  }

  return GetParticipatingDevicesGroups(device_assignment,
                                       collective.replica_groups(), group_mode);
}

bool IsWithinScaleUpFabricDomain(absl::Span<const GlobalDeviceId> group,
                                 int32_t scale_up_fabric_size) {
  if (group.empty()) {
    return false;
  }

  auto [min_device, max_device] =
      std::minmax_element(group.begin(), group.end());
  int64_t min_device_id = min_device->value();
  int64_t max_device_id = max_device->value();
  if (min_device_id < 0) {
    return false;
  }

  int64_t min_domain_id = min_device_id / scale_up_fabric_size;
  int64_t max_domain_id = max_device_id / scale_up_fabric_size;
  return min_domain_id == max_domain_id;
}

bool IsWithinScaleUpFabricDomain(
    absl::Span<const std::vector<GlobalDeviceId>> groups,
    int32_t scale_up_fabric_size) {
  if (groups.empty()) {
    return false;
  }

  return absl::c_all_of(groups, [&](absl::Span<const GlobalDeviceId> group) {
    return IsWithinScaleUpFabricDomain(group, scale_up_fabric_size);
  });
}

absl::StatusOr<bool> IsScaleUpFabricCollective(
    const HloInstruction& collective, const DeviceAssignment& device_assignment,
    int32_t scale_up_fabric_size) {
  ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain current_domain,
                   GetCollectiveCommunicationDomain(collective));
  if (current_domain != kUnspecifiedCollectiveDomain) {
    return current_domain == kScaleUpFabricCollectiveDomain;
  }
  ABSL_ASSIGN_OR_RETURN(auto participant_groups, GetCollectiveParticipantGroups(
                                                collective, device_assignment));
  return IsWithinScaleUpFabricDomain(participant_groups, scale_up_fabric_size);
}

absl::StatusOr<bool> IsScaleUpFabricEligible(
    const HloInstruction& instruction,
    const DeviceAssignment& device_assignment, int32_t scale_up_fabric_size) {
  if (instruction.opcode() != HloOpcode::kCall) {
    return IsScaleUpFabricCollective(instruction, device_assignment,
                                     scale_up_fabric_size);
  }

  bool has_collective = false;
  for (const HloInstruction* member : instruction.to_apply()->instructions()) {
    if (!hlo_query::IsCollectiveCommunicationOp(member->opcode())) {
      continue;
    }
    has_collective = true;
    ABSL_ASSIGN_OR_RETURN(bool is_scale_up,
                     IsScaleUpFabricCollective(*member, device_assignment,
                                               scale_up_fabric_size));
    if (!is_scale_up) {
      return false;
    }
  }
  return has_collective;
}

absl::StatusOr<bool> AssignDomain(HloInstruction& instruction,
                                  const DeviceAssignment& device_assignment,
                                  int32_t scale_up_fabric_size) {
  ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain current_domain,
                   GetCollectiveCommunicationDomain(instruction));
  if (current_domain != kUnspecifiedCollectiveDomain) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(bool is_eligible,
                   IsScaleUpFabricEligible(instruction, device_assignment,
                                           scale_up_fabric_size));
  if (!is_eligible) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                   instruction.backend_config<GpuBackendConfig>());
  config.mutable_collective_backend_config()->set_communication_domain(
      kScaleUpFabricCollectiveDomain);
  ABSL_RETURN_IF_ERROR(instruction.set_backend_config(config));
  return true;
}

bool IsDomainAssignmentCandidate(const HloInstruction& instruction) {
  return hlo_query::IsCollectiveCommunicationOp(instruction.opcode()) ||
         (instruction.opcode() == HloOpcode::kCall &&
          instruction.frontend_attributes().map().contains(
              kCollectiveGroupMarkerAttr));
}

}  // namespace

CollectiveDomainAssigner::CollectiveDomainAssigner(
    const GpuTopology& gpu_topology)
    : gpu_topology_(gpu_topology) {}

absl::StatusOr<bool> CollectiveDomainAssigner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();

  absl::string_view domain_assignment =
      config.debug_options().xla_gpu_collective_domain_assignment();
  ABSL_ASSIGN_OR_RETURN(auto domains_to_assign,
                   ParseDomainsToAssign(domain_assignment));
  if (!domains_to_assign.contains(kScaleUpFabricCollectiveDomain)) {
    return false;
  }

  int32_t scale_up_fabric_size = gpu_topology_.slice_size();
  if (scale_up_fabric_size <= 1) {
    return false;
  }

  // XLA:GPU supports IOTA device assignments. When a useful static assignment
  // is unavailable, use the corresponding zero-based IOTA assignment. The
  // all-zero case is a sentinel used by functional_hlo_runner.
  DeviceAssignment default_device_assignment(config.replica_count(),
                                             config.num_partitions());
  default_device_assignment.FillIota(0);

  const DeviceAssignment* device_assignment = &default_device_assignment;
  if (config.has_static_device_assignment() &&
      !config.static_device_assignment().IsAll(0)) {
    device_assignment = &config.static_device_assignment();
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!IsDomainAssignmentCandidate(*instruction)) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(
          bool instruction_changed,
          AssignDomain(*instruction, *device_assignment, scale_up_fabric_size));
      changed |= instruction_changed;
    }
  }

  return changed;
}

}  // namespace xla::gpu
