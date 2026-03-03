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

#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Computes a map from source partition ID to a set of target partition IDs for
// a collective-permute instruction. A partition ID is computed by dividing the
// device (replica) ID by the number of devices per host.
absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>
GetSourceToTargetsNodeMap(const HloCollectivePermuteInstruction& instr,
                          int num_devices_per_partition) {
  absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>
      source_to_targets_partition_map;
  for (const auto& [source, target] : instr.source_target_pairs()) {
    int64_t source_partition = source / num_devices_per_partition;
    int64_t target_partition = target / num_devices_per_partition;
    source_to_targets_partition_map[source_partition].insert(target_partition);
  }
  return source_to_targets_partition_map;
}

struct CollectiveMetadata {
  // map for ops with `replica_groups`, e.g. all-gather.
  absl::flat_hash_map<int64_t, size_t> partition_to_participant_count;
  int num_devices_per_partition;
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
  absl::c_sort(lhs_counts);
  absl::c_sort(rhs_counts);
  return lhs_counts == rhs_counts;
}

absl::StatusOr<CollectiveMetadata> CommunicationContext(
    const HloCollectiveInstruction& instr, int num_devices_per_partition) {
  absl::flat_hash_map<int64_t, size_t> partition_to_participant_count;

  for (const ReplicaGroup& replica_group :
       instr.device_list().replica_groups()) {
    absl::flat_hash_map<int64_t, size_t> buffer;
    for (int64_t rank : replica_group.replica_ids()) {
      int64_t partition_id = rank / num_devices_per_partition;
      buffer[partition_id]++;
    }
    if (!partition_to_participant_count.empty() &&
        !SameParticipantCounts(buffer, partition_to_participant_count)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Non homogenous replica group: ", instr.device_list().ToString()));
    }
    if (partition_to_participant_count.empty()) {
      partition_to_participant_count = buffer;
    }
  }
  return CollectiveMetadata{partition_to_participant_count,
                            num_devices_per_partition,
                            instr.GetModule()->config().replica_count()};
}

bool IsSingleHost(const CollectiveMetadata& pattern) {
  if (pattern.partition_to_participant_count.size() == 1) {
    return true;
  }
  return pattern.replica_count > 0 &&
         pattern.partition_to_participant_count.empty() &&
         pattern.replica_count <= pattern.num_devices_per_partition;
}

bool IsWorldLevelCommunication(const CollectiveMetadata& pattern) {
  if (!IsSingleHost(pattern) &&
      pattern.partition_to_participant_count.empty()) {
    return true;
  }
  return absl::c_all_of(
      pattern.partition_to_participant_count, [&pattern](const auto& elem) {
        const auto& [partition_id, participant_count] = elem;
        return participant_count == pattern.num_devices_per_partition;
      });
}

bool IsNonWorldLevelCommunication(const CollectiveMetadata& pattern) {
  return !IsSingleHost(pattern) && !IsWorldLevelCommunication(pattern);
}

// Properties of a collective-permute instruction, categorizing its
// communication pattern.
struct CollectivePermuteProperty {
  std::vector<std::pair<int64_t, int64_t>> intra_partition_source_target_pairs;
  std::vector<std::pair<int64_t, int64_t>> inter_partition_source_target_pairs;
  // If true, at least one device both sends and receives data. If false, every
  // device involved in the collective-permute either only sends or only
  // receives data.
  bool has_devices_with_two_edges = false;
  // True if for every pair (s,t) in source_target_pairs, the pair (t,s) is
  // also present in source_target_pairs.
  bool is_all_mutual = false;
};

// TODO(b/460155942): remove the optional wrapper once the HLO verifier stop
// supporting empty source-target pairs.
std::optional<CollectivePermuteProperty> GetCollectivePermuteProperty(
    const HloCollectivePermuteInstruction& instr,
    int64_t num_devices_per_partition) {
  CHECK_GT(num_devices_per_partition, 0);
  if (instr.source_target_pairs().empty()) {
    return std::nullopt;
  }

  CollectivePermuteProperty property;
  absl::flat_hash_set<int64_t> sources, targets;
  absl::flat_hash_set<std::pair<int64_t, int64_t>> pairs_set;
  absl::c_for_each(instr.source_target_pairs(),
                   [&](const auto& pair) { pairs_set.insert(pair); });

  property.is_all_mutual = true;

  for (const auto& [source, target] : instr.source_target_pairs()) {
    sources.insert(source);
    targets.insert(target);

    bool is_intra_partition = (source / num_devices_per_partition ==
                               target / num_devices_per_partition);

    if (is_intra_partition) {
      property.intra_partition_source_target_pairs.push_back({source, target});
    } else {
      property.inter_partition_source_target_pairs.push_back({source, target});
    }
    // If anyone of the pair (t,s) is not present in source_target_pairs, the
    // communication pattern is not all-mutual.
    if (property.is_all_mutual && !pairs_set.contains({target, source})) {
      property.is_all_mutual = false;
    }
  }

  // If any source device is a target device, then it has two edges.
  for (int64_t source : sources) {
    if (targets.contains(source)) {
      property.has_devices_with_two_edges = true;
      break;
    }
  }

  return property;
}

}  // namespace

CollectivePermuteCostModelType GetCollectivePermuteCostModelType(
    const HloCollectivePermuteInstruction& instr,
    int64_t num_devices_per_partition) {
  std::optional<CollectivePermuteProperty> property =
      GetCollectivePermuteProperty(instr, num_devices_per_partition);
  if (!property) {
    return CollectivePermuteCostModelType::kUnknown;
  }

  if (!property->inter_partition_source_target_pairs.empty()) {
    if (property->has_devices_with_two_edges) {
      return property->is_all_mutual ? CollectivePermuteCostModelType::
                                           kInterPartitionTwoWayAllMutual
                                     : CollectivePermuteCostModelType::
                                           kInterPartitionTwoWayHasNonMutual;
    }
    return CollectivePermuteCostModelType::kInterPartitionOneWay;
  }

  if (property->has_devices_with_two_edges) {
    return property->is_all_mutual
               ? CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual
               : CollectivePermuteCostModelType::
                     kIntraPartitionTwoWayHasNonMutual;
  }
  return CollectivePermuteCostModelType::kIntraPartitionOneWay;
}

bool IsGPUSyncCollective(const HloInstruction& instr) {
  auto backend_config = instr.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }
  return backend_config->collective_backend_config().is_sync();
}

absl::StatusOr<GPUCommunicationType> CommunicationType(
    int num_devices_per_partition, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version) {
  if (!gpu_version.IsCuda()) {
    return absl::FailedPreconditionError("Only CUDA is supported.");
  }

  if (const auto* collective = DynCast<HloCollectiveInstruction>(&instr)) {
    TF_ASSIGN_OR_RETURN(
        CollectiveMetadata comm,
        CommunicationContext(*collective, num_devices_per_partition));
    if (IsSingleHost(comm)) {
      return GPUCommunicationType::SINGLE_PARTITION;
    }
    if (IsWorldLevelCommunication(comm)) {
      return GPUCommunicationType::MULTI_HOST_WORLD_LEVEL;
    }
    if (IsNonWorldLevelCommunication(comm)) {
      return GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL;
    }
  } else if (const auto* collective_permute =
                 DynCast<HloCollectivePermuteInstruction>(&instr)) {
    const auto source_to_targets_partition_map = GetSourceToTargetsNodeMap(
        *collective_permute, num_devices_per_partition);
    for (const auto& [source_partition, target_partition_set] :
         source_to_targets_partition_map) {
      if (target_partition_set.size() > 1) {
        return GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL;
      }
      CHECK_EQ(target_partition_set.size(), 1);
      if (source_partition != *target_partition_set.begin()) {
        return GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL;
      }
    }
    return GPUCommunicationType::SINGLE_PARTITION;
  } else {
    return absl::FailedPreconditionError(
        "Cannot determine communication type for non-collective channel "
        "instruction");
  }

  return GPUCommunicationType::UNDEFINED;
}

bool IsIntraNVLinkDomain(const HloModuleConfig& config, int64_t slice_size) {
  int device_count = config.num_partitions() * config.replica_count();
  bool is_intra = device_count <= slice_size;
  VLOG(1) << "IsIntraNVLinkDomain: device_count=" << device_count
          << " slice_size=" << slice_size << " is_intra=" << is_intra;
  return is_intra;
}

}  // namespace gpu
}  // namespace xla
