/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/collective_ops_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Match the instruction to a reduction kind. We can represent and/or of pred as
// min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
std::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo) {
  PrimitiveType type = hlo->shape().element_type();
  switch (hlo->opcode()) {
    case HloOpcode::kAdd:
      return ReductionKind::SUM;
    case HloOpcode::kMultiply:
      return ReductionKind::PRODUCT;
    case HloOpcode::kMinimum:
      return ReductionKind::MIN;
    case HloOpcode::kMaximum:
      return ReductionKind::MAX;
    case HloOpcode::kAnd:
      return type == PRED ? std::optional<ReductionKind>(ReductionKind::MIN)
                          : std::nullopt;
    case HloOpcode::kOr:
      return type == PRED ? std::optional<ReductionKind>(ReductionKind::MAX)
                          : std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation) {
  namespace m = match;
  const HloInstruction* root = computation->root_instruction();
  auto kind = MatchReductionInstruction(root);
  if (kind && !Match(root, m::Op()
                               .WithBinaryOperandsAnyOrder(m::Parameter(0),
                                                           m::Parameter(1))
                               .WithShape(m::Shape().IsEffectiveScalar()))) {
    kind = std::nullopt;
  }
  return kind;
}

std::optional<Literal> GetReductionIdentity(ReductionKind kind,
                                            PrimitiveType type) {
  switch (kind) {
    case ReductionKind::SUM:
      return LiteralUtil::Zero(type);
    case ReductionKind::PRODUCT:
      return LiteralUtil::One(type);
    case ReductionKind::MIN:
      return LiteralUtil::MaxValue(type);
    case ReductionKind::MAX:
      return LiteralUtil::MinValue(type);
    default:
      return std::nullopt;
  }
}

absl::StatusOr<std::vector<int>> GetParticipatingIDs(
    CollectiveOpGroupMode group_mode, int current_id,
    std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups) {
  // Empty replica_groups() means that all replicas participate.
  if (groups.empty()) {
    TF_RET_CHECK(total_participant_count.has_value());
    std::vector<int> all_participants(*total_participant_count);
    absl::c_iota(all_participants, 0);
    return all_participants;
  }

  // Formatter for printing replica groups in StrJoin.
  auto group_formatter = [](std::string* out, const ReplicaGroup& group) {
    out->append("[");
    out->append(absl::StrJoin(group.replica_ids(), ", "));
    out->append("]");
  };

  // Figure out the other replicas that go together with this one.
  std::optional<ReplicaGroup> group;
  for (const ReplicaGroup& g : groups) {
    if (absl::c_linear_search(g.replica_ids(), current_id)) {
      TF_RET_CHECK(!group.has_value())
          << "Replica ID " << current_id << " appears twice in replica groups"
          << "; group_mode=" << CollectiveOpGroupModeToString(group_mode)
          << "; groups_size=" << groups.size()
          << "; groups= " << absl::StrJoin(groups, ", ", group_formatter);
      group = g;
    }
  }
  TF_RET_CHECK(group.has_value())
      << "Replica ID " << current_id << " doesn't appear in replica groups"
      << "; group_mode=" << CollectiveOpGroupModeToString(group_mode)
      << "; groups_size=" << groups.size()
      << "; groups= " << absl::StrJoin(groups, ", ", group_formatter);
  return std::vector<int>(group->replica_ids().begin(),
                          group->replica_ids().end());
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (!use_global_device_ids.has_value() || !*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplica;
    } else {
      return InvalidArgument(
          "Invalid combination of has_channel_id and use_global_device_ids");
    }
  } else {
    if (!use_global_device_ids.has_value()) {
      return CollectiveOpGroupMode::kCrossPartition;
    } else if (!*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplicaAndPartition;
    } else {
      return CollectiveOpGroupMode::kFlattenedID;
    }
  }
}

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode) {
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return "kCrossReplica";
    case CollectiveOpGroupMode::kCrossPartition:
      return "kCrossPartition";
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return "kCrossReplicaAndPartition";
    case CollectiveOpGroupMode::kFlattenedID:
      return "kFlattenedID";
  }
}

absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const DeviceAssignment& device_assignment,
                              absl::Span<const ReplicaGroup> replica_groups,
                              CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  std::vector<ReplicaGroup> participating_replica_groups =
      SpanToVector(replica_groups);

  // If replica groups are empty, assume a group with all replicas.
  if (replica_groups.empty()) {
    if (group_mode == CollectiveOpGroupMode::kFlattenedID) {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";
    }

    int total_participant_count;
    if (group_mode == CollectiveOpGroupMode::kCrossPartition) {
      // replica group are partition ids.
      total_participant_count = partition_count;
    } else {
      // replica group are replica ids.
      total_participant_count = replica_count;
    }

    ReplicaGroup replica_group = ReplicaGroup();
    for (int id = 0; id < total_participant_count; id++) {
      replica_group.add_replica_ids(id);
    }
    participating_replica_groups.push_back(replica_group);
  }

  std::vector<std::vector<GlobalDeviceId>> groups;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      for (const auto& replica_group : participating_replica_groups) {
        // replica_group contains replica id, participants contains all
        // replica_group's replica_ids for the current partition.
        for (int partition_id = 0; partition_id < partition_count;
             partition_id++) {
          std::vector<GlobalDeviceId> participants;
          participants.reserve(replica_group.replica_ids().size());

          for (int replica_id : replica_group.replica_ids()) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
          groups.push_back(participants);
        }
      }
      return groups;
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        // replica_group contains partition id, participants contains all
        // replica_group's partition_ids for the current replica_id.
        for (int replica_id = 0; replica_id < replica_count; replica_id++) {
          std::vector<GlobalDeviceId> participants;
          participants.reserve(replica_group.replica_ids().size());

          for (int partition_id : replica_group.replica_ids()) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
          groups.push_back(participants);
        }
      }
      return groups;
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        std::vector<GlobalDeviceId> participants;
        participants.reserve(replica_group.replica_ids().size() *
                             partition_count);

        // replica_group contains replica id, participants contains all
        // replica_group's replica_ids for all partitions.
        for (int replica_id : replica_group.replica_ids()) {
          for (int partition_id = 0; partition_id < partition_count;
               partition_id++) {
            participants.emplace_back(
                device_assignment(replica_id, partition_id));
          }
        }
        groups.push_back(participants);
      }
      return groups;
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      for (const auto& replica_group : participating_replica_groups) {
        std::vector<GlobalDeviceId> participants;
        participants.reserve(replica_group.replica_ids().size());

        for (int flattened_id : replica_group.replica_ids()) {
          // Map from flattened id back to replica_id, partition_id.
          int replica_id = flattened_id / partition_count;
          int partition_id = flattened_id % partition_count;
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
        groups.push_back(participants);
      }
      return groups;
    }
  }
}

absl::StatusOr<std::vector<ReplicaGroup>> GetParticipatingFlattenedIdGroups(
    const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  // Compute the device_id to flattened_id mapping once to avoid brute force
  // searching through device assignment repeatedly.
  absl::flat_hash_map<GlobalDeviceId, int64_t> device_id_to_flattened_id;
  for (int r = 0; r < device_assignment.replica_count(); ++r) {
    for (int c = 0; c < device_assignment.computation_count(); ++c) {
      GlobalDeviceId device_id = GlobalDeviceId(device_assignment(r, c));
      int64_t flattened_id = r * device_assignment.computation_count() + c;
      device_id_to_flattened_id[device_id] = flattened_id;
    }
  }

  std::vector<ReplicaGroup> flattened_id_groups;
  TF_ASSIGN_OR_RETURN(std::vector<std::vector<GlobalDeviceId>> device_groups,
                      GetParticipatingDevicesGroups(
                          device_assignment, replica_groups, group_mode));
  for (const auto& device_group : device_groups) {
    ReplicaGroup flattened_id_group;
    flattened_id_group.mutable_replica_ids()->Reserve(device_group.size());
    for (const GlobalDeviceId& device_id : device_group) {
      flattened_id_group.add_replica_ids(device_id_to_flattened_id[device_id]);
    }
    flattened_id_groups.push_back(flattened_id_group);
  }
  return flattened_id_groups;
}

absl::StatusOr<std::vector<ReplicaGroup>> GetParticipatingFlattenedIdGroups(
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode replica_group_mode, int replica_count,
    int partition_count) {
  std::vector<ReplicaGroup> filled_empty_replica_group;
  absl::Span<const ReplicaGroup> original_replica_groups = replica_groups;
  std::vector<ReplicaGroup> flattened_replica_groups;
  if (replica_groups.empty()) {
    filled_empty_replica_group.emplace_back();
    const int64_t id_count =
        replica_group_mode == CollectiveOpGroupMode::kCrossPartition
            ? partition_count
            : replica_count;
    for (int i = 0; i < id_count; ++i) {
      filled_empty_replica_group.back().add_replica_ids(i);
    }
    original_replica_groups = filled_empty_replica_group;
  }
  if (replica_group_mode == CollectiveOpGroupMode::kFlattenedID) {
    flattened_replica_groups.insert(flattened_replica_groups.end(),
                                    original_replica_groups.begin(),
                                    original_replica_groups.end());
  } else if (replica_group_mode == CollectiveOpGroupMode::kCrossReplica) {
    flattened_replica_groups.resize(original_replica_groups.size() *
                                    partition_count);
    for (int64_t i = 0, current_group_offset = 0;
         i < original_replica_groups.size();
         ++i, current_group_offset += partition_count) {
      for (int64_t replica_id : original_replica_groups.at(i).replica_ids()) {
        for (int64_t partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          const int64_t flattened_id =
              replica_id * partition_count + partition_id;
          flattened_replica_groups[current_group_offset + partition_id]
              .add_replica_ids(flattened_id);
        }
      }
    }
  } else if (replica_group_mode == CollectiveOpGroupMode::kCrossPartition) {
    flattened_replica_groups.resize(original_replica_groups.size() *
                                    replica_count);
    for (int64_t i = 0, current_group_offset = 0;
         i < original_replica_groups.size();
         ++i, current_group_offset += replica_count) {
      for (int64_t partition_id : original_replica_groups.at(i).replica_ids()) {
        for (int64_t replica_id = 0; replica_id < replica_count; ++replica_id) {
          const int64_t flattened_id =
              replica_id * partition_count + partition_id;
          flattened_replica_groups[current_group_offset + replica_id]
              .add_replica_ids(flattened_id);
        }
      }
    }
  } else {
    CHECK(replica_group_mode ==
          CollectiveOpGroupMode::kCrossReplicaAndPartition);
    flattened_replica_groups.resize(original_replica_groups.size());
    for (int64_t i = 0; i < original_replica_groups.size(); ++i) {
      for (int64_t replica_id : original_replica_groups.at(i).replica_ids()) {
        for (int64_t partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          const int64_t flattened_id =
              replica_id * partition_count + partition_id;
          flattened_replica_groups[i].add_replica_ids(flattened_id);
        }
      }
    }
  }
  return flattened_replica_groups;
}

absl::StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      device_assignment.LogicalIdForDevice(device_id));
  int current_replica_id = logical_id.replica_id;
  int current_partition_id = logical_id.computation_id;
  TF_RET_CHECK(0 <= current_replica_id && current_replica_id < replica_count)
      << current_replica_id << " " << replica_count;
  TF_RET_CHECK(0 <= current_partition_id &&
               current_partition_id < partition_count)
      << current_partition_id << " " << partition_count;

  std::vector<GlobalDeviceId> participants;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      // This is a cross replica operation. replica group contains replica id.
      // use current replica id to find the set of participating replicas. If
      // replica groups are empty, assume a group with all replicas.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(group_mode, current_replica_id,
                                              replica_count, replica_groups));

      // The set of participating devices is the replicas from the current
      // partition.
      participants.reserve(participating_replicas.size());
      for (int replica_id : participating_replicas) {
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        participants.emplace_back(
            device_assignment(replica_id, current_partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossPartition: {
      // replica_groups contain partition_id, group contains all partitions for
      // the current replica.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_partitions,
                          GetParticipatingIDs(group_mode, current_partition_id,
                                              partition_count, replica_groups));
      participants.reserve(participating_partitions.size());
      for (int partition_id : participating_partitions) {
        TF_RET_CHECK(0 <= partition_id && partition_id < partition_count)
            << partition_id << " " << partition_count;
        participants.emplace_back(
            device_assignment(current_replica_id, partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      // replica_groups contain replica_ids. Group contains replicas for all
      // partitions.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(group_mode, current_replica_id,
                                              replica_count, replica_groups));
      participants.reserve(participating_replicas.size() * partition_count);
      for (int replica_id : participating_replicas) {
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        for (int partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
      }
      return participants;
    }

    case CollectiveOpGroupMode::kFlattenedID: {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";

      int current_flattened_id =
          current_replica_id * partition_count + current_partition_id;

      // Find participants based on flattened id. replica_groups cannot be empty
      // so no need to pass in total_participant_count.
      TF_ASSIGN_OR_RETURN(
          std::vector<int> participating_flattened_ids,
          GetParticipatingIDs(group_mode, current_flattened_id,
                              /*total_participant_count=*/std::nullopt,
                              replica_groups));

      participants.reserve(participating_flattened_ids.size());
      for (int flattened_id : participating_flattened_ids) {
        // Map from flattened id back to replica_id, partition_id.
        int replica_id = flattened_id / partition_count;
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        int partition_id = flattened_id % partition_count;
        participants.emplace_back(device_assignment(replica_id, partition_id));
      }
      return participants;
    }
  }
}

absl::StatusOr<std::vector<int64_t>> GetPariticipantCountsForReplicaGroups(
    int64_t num_replicas, int64_t num_partitions,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  std::vector<int64_t> participant_counts;
  std::vector<ReplicaGroup> participating_replica_groups =
      SpanToVector(replica_groups);

  // If replica groups are empty, assume a group with all replicas.
  if (replica_groups.empty()) {
    if (group_mode == CollectiveOpGroupMode::kFlattenedID) {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";
    }

    int total_participant_count;
    if (group_mode == CollectiveOpGroupMode::kCrossPartition) {
      // replica group are partition ids.
      total_participant_count = num_partitions;
    } else {
      // replica group are replica ids.
      total_participant_count = num_replicas;
    }

    ReplicaGroup replica_group = ReplicaGroup();
    for (int id = 0; id < total_participant_count; id++) {
      replica_group.add_replica_ids(id);
    }
    participating_replica_groups.push_back(replica_group);
  }

  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      participant_counts.resize(participating_replica_groups.size(),
                                num_partitions);
      return participant_counts;
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size());
      }
      return participant_counts;
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size() *
                                     num_partitions);
      }
      return participant_counts;
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size());
      }
      return participant_counts;
    }
  }
}

bool ReplicaGroupsOrthogonal(absl::Span<const ReplicaGroup> first,
                             absl::Span<const ReplicaGroup> second) {
  if (first.size() != second[0].replica_ids_size()) {
    return false;
  }
  if (first[0].replica_ids_size() != second.size()) {
    return false;
  }
  for (int64_t i = 0; i < first.size(); ++i) {
    for (int64_t j = 0; j < first[i].replica_ids_size(); ++j) {
      if (first[i].replica_ids(j) != second[j].replica_ids(i)) {
        return false;
      }
    }
  }
  return true;
}

bool ReplicaGroupsEqual(absl::Span<const ReplicaGroup> first,
                        absl::Span<const ReplicaGroup> second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (int64_t i = 0; i < first.size(); ++i) {
    if (first[i].replica_ids_size() != second[i].replica_ids_size()) {
      return false;
    }
    for (int j = 0; j < first[i].replica_ids_size(); ++j) {
      if (first[i].replica_ids(j) != second[i].replica_ids(j)) {
        return false;
      }
    }
  }
  return true;
}

bool IsCollective(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kReduceScatter:
      return true;
    case HloOpcode::kFusion:
      if (instruction->IsCustomFusion()) {
        for (const auto* inner_inst : instruction->fused_instructions()) {
          if (IsCollective(inner_inst)) {
            return true;
          }
        }
      }
      return false;
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return IsCollective(instruction->async_wrapped_instruction());
    default:
      return false;
  }
}

HloInstruction* IsOrHasCollectiveWithChannelId(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    for (auto* inner_inst : instruction->fused_instructions()) {
      if (IsOrHasCollectiveWithChannelId(inner_inst) != nullptr) {
        return inner_inst;
      }
    }
    return nullptr;
  }
  if (DynCast<HloChannelInstruction>(instruction) == nullptr) {
    return nullptr;
  }
  if (IsCollective(instruction) && instruction->channel_id().has_value()) {
    return instruction;
  }
  return nullptr;
}

bool IsSyncCollective(const HloInstruction* instr) {
  auto backend_config = instr->backend_config<xla::gpu::GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }
  return backend_config->collective_backend_config().is_sync();
}

}  // end namespace xla
