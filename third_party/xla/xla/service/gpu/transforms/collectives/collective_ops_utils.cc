/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct CommunicationMetadata {
  absl::flat_hash_map<int64_t, size_t> node_to_participant_count;
  int num_devices_per_host;
  int64_t replica_count;
};

bool SameParticipantCounts(const absl::flat_hash_map<int64_t, size_t>& lhs,
                           const absl::flat_hash_map<int64_t, size_t>& rhs) {
  std::vector<size_t> lhs_counts, rhs_counts;
  lhs_counts.reserve(lhs.size());
  for (const auto& [_, v] : lhs) {
    lhs_counts.push_back(v);
  }

  rhs_counts.reserve(rhs.size());
  for (const auto& [_, v] : rhs) {
    rhs_counts.push_back(v);
  }
  std::sort(lhs_counts.begin(), lhs_counts.end());
  std::sort(rhs_counts.begin(), rhs_counts.end());
  return lhs_counts == rhs_counts;
}

absl::StatusOr<CommunicationMetadata> CommunicationContext(
    const HloChannelInstruction& instr, int num_devices_per_host) {
  absl::flat_hash_map<int64_t, size_t> node_to_participant_count;

  if (auto* collective =
          dynamic_cast<const HloCollectiveInstruction*>(&instr)) {
    for (const ReplicaGroup& replica_group :
         collective->device_list().replica_groups()) {
      absl::flat_hash_map<int64_t, size_t> buffer;
      for (int64_t rank : replica_group.replica_ids()) {
        int64_t node_id = rank / num_devices_per_host;
        buffer[node_id]++;
      }
      if (!node_to_participant_count.empty() &&
          !SameParticipantCounts(buffer, node_to_participant_count)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Non homogenous replica group: ",
                         collective->device_list().ToString()));
      }
      if (node_to_participant_count.empty()) {
        node_to_participant_count = buffer;
      }
    }
  } else if (auto* permute =
                 dynamic_cast<const HloCollectivePermuteInstruction*>(&instr)) {
    for (const auto& [source, target] : permute->source_target_pairs()) {
      int64_t source_node = source / num_devices_per_host;
      int64_t target_node = target / num_devices_per_host;
      node_to_participant_count[source_node]++;
      node_to_participant_count[target_node]++;
    }
  } else {
    return absl::FailedPreconditionError(
        "Cannot determine communication context for non-collective channel "
        "instruction");
  }

  return CommunicationMetadata{node_to_participant_count, num_devices_per_host,
                               instr.GetModule()->config().replica_count()};
}

bool IsSingleHost(const CommunicationMetadata& pattern) {
  if (pattern.node_to_participant_count.size() == 1) {
    return true;
  }
  return pattern.replica_count > 0 &&
         pattern.node_to_participant_count.empty() &&
         pattern.replica_count <= pattern.num_devices_per_host;
}

bool IsRailAligned(const CommunicationMetadata& pattern) {
  if (!IsSingleHost(pattern) && pattern.node_to_participant_count.empty()) {
    return true;
  }
  return absl::c_all_of(
      pattern.node_to_participant_count, [&pattern](const auto& elem) {
        const auto& [node_id, participant_count] = elem;
        return participant_count == pattern.num_devices_per_host;
      });
}

bool IsNonRailAligned(const CommunicationMetadata& pattern) {
  return !IsSingleHost(pattern) && !IsRailAligned(pattern);
}

}  // namespace

bool IsGPUSyncCollective(const HloInstruction& instr) {
  auto backend_config = instr.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }
  return backend_config->collective_backend_config().is_sync();
}

absl::StatusOr<GPUCommunicationType> CommunicationType(
    int num_devices_per_host, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version) {
  if (!std::holds_alternative<se::CudaComputeCapability>(gpu_version)) {
    return absl::FailedPreconditionError("Only CUDA is supported.");
  }

  TF_ASSIGN_OR_RETURN(CommunicationMetadata comm,
                      CommunicationContext(instr, num_devices_per_host));
  if (IsSingleHost(comm)) {
    return GPUCommunicationType::SINGLE_HOST;
  }
  if (IsRailAligned(comm)) {
    return GPUCommunicationType::RAIL_ALIGNED;
  }
  if (IsNonRailAligned(comm)) {
    return GPUCommunicationType::NON_RAIL_ALIGNED;
  }

  return GPUCommunicationType::UNDEFINED;
}

std::optional<bool> IsMultiHostTopology(
    const HloModuleConfig& config,
    const se::DeviceDescription& device_description) {
  se::CudaComputeCapability cc = device_description.cuda_compute_capability();
  // TODO: b/390095346 - Use topology information once available at compile
  // time.
  if (cc.IsHopper()) {
    return config.num_partitions() * config.replica_count() > 8;
  }
  if (cc.IsAmpere()) {
    return config.num_partitions() * config.replica_count() > 16;
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
