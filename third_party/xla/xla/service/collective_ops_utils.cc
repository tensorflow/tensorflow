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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/collective_permute_cycle.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
using CycleType = collective_permute_cycle::CycleType;

absl::StatusOr<ReductionKind> StringToReductionKind(
    absl::string_view reduction_kind) {
  if (reduction_kind == "sum") {
    return ReductionKind::SUM;
  } else if (reduction_kind == "prod") {
    return ReductionKind::PRODUCT;
  } else if (reduction_kind == "min") {
    return ReductionKind::MIN;
  } else if (reduction_kind == "max") {
    return ReductionKind::MAX;
  }
  return InvalidArgument("Invalid reduction kind: %s", reduction_kind);
}

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

absl::StatusOr<std::vector<std::vector<int64_t>>> GetAsyncReplicaGroups(
    const HloInstruction* instruction) {
  std::vector<std::vector<int64_t>> replica_groups;
  if (instruction->opcode() == HloOpcode::kCollectivePermuteStart) {
    absl::c_transform(instruction->source_target_pairs(),
                      std::back_inserter(replica_groups),
                      [](const std::pair<int64_t, int64_t>& pair) {
                        std::vector<int64_t> ids({pair.first, pair.second});
                        return ids;
                      });
  } else if (instruction->IsAsynchronous() ||
             instruction->opcode() == HloOpcode::kAllGatherStart ||
             instruction->opcode() == HloOpcode::kAllReduceStart) {
    absl::c_transform(
        instruction->replica_groups(), std::back_inserter(replica_groups),
        [](const ReplicaGroup& group) {
          std::vector<int64_t> ids;
          absl::c_transform(group.replica_ids(), std::back_inserter(ids),
                            [](auto id) { return id; });
          return ids;
        });
  } else {
    return InvalidArgument(
        "Unexpected instruction type: %s is not an async collective "
        "instruction",
        instruction->ToString());
  }
  return replica_groups;
}

absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    const HloInstruction* instr) {
  if (auto collective = DynCast<HloAllGatherInstruction>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    collective->use_global_device_ids());
  } else if (auto collective = DynCast<HloAllReduceInstructionBase>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    collective->use_global_device_ids());
  } else if (auto collective = DynCast<HloAllToAllInstruction>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    std::nullopt);
  } else if (auto collective =
                 DynCast<HloCollectiveBroadcastInstruction>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    std::nullopt);
  } else if (auto collective =
                 DynCast<HloCollectivePermuteInstruction>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    std::nullopt);
  } else if (auto collective = DynCast<HloRaggedAllToAllInstruction>(instr)) {
    return GetCollectiveOpGroupMode(collective->channel_id().has_value(),
                                    std::nullopt);
  }
  return Internal("Unexpected instruction type.");
}

const CollectiveDeviceList& GetCollectiveDeviceList(const HloInstruction* hlo) {
  return Cast<HloCollectiveInstruction>(hlo)->device_list();
}

const std::vector<ReplicaGroup>& GetCollectiveReplicaGroups(
    const HloInstruction* hlo) {
  return Cast<HloCollectiveInstruction>(hlo)->replica_groups();
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

absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const HloInstruction* collective) {
  CHECK(collective->GetModule()->config().has_static_device_assignment());
  const DeviceAssignment& device_assignment =
      collective->GetModule()->config().static_device_assignment();
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode mode,
                      GetCollectiveOpGroupMode(collective));
  return GetParticipatingDevicesGroups(
      device_assignment, GetCollectiveReplicaGroups(collective), mode);
}

absl::StatusOr<CollectiveDeviceList> GetParticipatingFlattenedIdGroups(
    const DeviceAssignment& device_assignment,
    const CollectiveDeviceList& collective_device_list,
    CollectiveOpGroupMode group_mode) {
  return GetParticipatingFlattenedIdGroups(
      collective_device_list, group_mode, device_assignment.replica_count(),
      device_assignment.computation_count());
}

absl::StatusOr<CollectiveDeviceList> GetParticipatingFlattenedIdGroups(
    const CollectiveDeviceList& collective_device_list,
    CollectiveOpGroupMode group_mode, int replica_count, int partition_count) {
  if (group_mode == CollectiveOpGroupMode::kFlattenedID) {
    return collective_device_list;
  }
  std::vector<ReplicaGroup> filled_empty_replica_group;
  absl::Span<const ReplicaGroup> original_replica_groups =
      collective_device_list.replica_groups();
  std::vector<ReplicaGroup> flattened_replica_groups;
  if (collective_device_list.replica_groups().empty()) {
    filled_empty_replica_group.emplace_back();
    const int64_t id_count =
        group_mode == CollectiveOpGroupMode::kCrossPartition ? partition_count
                                                             : replica_count;
    for (int i = 0; i < id_count; ++i) {
      filled_empty_replica_group.back().add_replica_ids(i);
    }
    original_replica_groups = filled_empty_replica_group;
  }
  if (group_mode == CollectiveOpGroupMode::kCrossReplica) {
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
  } else if (group_mode == CollectiveOpGroupMode::kCrossPartition) {
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
    CHECK(group_mode == CollectiveOpGroupMode::kCrossReplicaAndPartition);
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
  return CollectiveDeviceList(flattened_replica_groups);
}

absl::StatusOr<CollectiveDeviceList> GetParticipatingFlattenedIdGroups(
    const HloInstruction* hlo, const DeviceAssignment& device_assignment) {
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode mode,
                      GetCollectiveOpGroupMode(hlo));
  TF_ASSIGN_OR_RETURN(
      CollectiveDeviceList collective_device_list,
      GetParticipatingFlattenedIdGroups(device_assignment,
                                        GetCollectiveDeviceList(hlo), mode));
  return collective_device_list;
}

// Same as above, used for cases where static_device_assignment is not present.
absl::StatusOr<CollectiveDeviceList> GetParticipatingFlattenedIdGroups(
    const HloInstruction* hlo, int replica_count, int partition_count) {
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode mode,
                      GetCollectiveOpGroupMode(hlo));
  TF_ASSIGN_OR_RETURN(
      CollectiveDeviceList collective_device_list,
      GetParticipatingFlattenedIdGroups(GetCollectiveDeviceList(hlo), mode,
                                        replica_count, partition_count));
  return collective_device_list;
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
      for (const auto& replica_group : participating_replica_groups) {
        for (int partition_id = 0; partition_id < num_partitions;
             ++partition_id) {
          participant_counts.push_back(replica_group.replica_ids().size());
        }
      }
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

absl::StatusOr<std::optional<std::pair<int64_t, int64_t>>>
GetReplicaGroupCountAndSize(const HloInstruction* hlo) {
  const CollectiveDeviceList& device_list = GetCollectiveDeviceList(hlo);
  auto config = hlo->GetModule()->config();

  if (device_list.iota_replica_group_list().has_value()) {
    return std::make_pair(
        device_list.iota_replica_group_list()->num_replica_groups(),
        device_list.iota_replica_group_list()->num_devices_per_group());
  }
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(hlo));
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> participant_counts,
                      GetPariticipantCountsForReplicaGroups(
                          config.replica_count(), config.num_partitions(),
                          device_list.replica_groups(), group_mode));
  int64_t replica_group_size = participant_counts[0];
  for (int64_t participant_count : participant_counts) {
    if (participant_count != replica_group_size) {
      return std::nullopt;
    }
  }
  return std::make_pair(participant_counts.size(), replica_group_size);
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

bool IsNonFusionCollective(const HloInstruction* instruction) {
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
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return true;
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return IsNonFusionCollective(instruction->async_wrapped_instruction());
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return !Cast<HloSendRecvInstruction>(instruction)->is_host_transfer();
    default:
      return false;
  }
}

bool IsCollective(const HloInstruction* instruction) {
  if (IsNonFusionCollective(instruction)) {
    return true;
  }
  if (instruction->opcode() == HloOpcode::kFusion &&
      instruction->IsCustomFusion()) {
    for (const auto* inner_inst : instruction->fused_instructions()) {
      if (IsCollective(inner_inst)) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<bool> IsAsyncCollective(const HloInstruction* instruction) {
  if (!IsNonFusionCollective(instruction)) {
    return false;
  }
  if (instruction->IsAsynchronous()) {
    switch (instruction->async_wrapped_opcode()) {
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllToAll:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kRaggedAllToAll:
      case HloOpcode::kReduceScatter:
        return true;
      default:
        return absl::InvalidArgumentError("Async instruction " +
                                          instruction->ToString() +
                                          " is not a collective.");
    }
  }
  switch (instruction->opcode()) {
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
      return true;
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return !Cast<HloSendRecvInstruction>(instruction)->is_host_transfer();
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return false;
    default:
      return absl::InvalidArgumentError("Instruction " +
                                        instruction->ToString() +
                                        " is not an async collective.");
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

using SourceTargetPairType = std::pair<int64_t, int64_t>;
using SourceTargetPairsType = std::vector<SourceTargetPairType>;

std::pair<CycleType, std::set<int>> GetCycleTypeAndIndices(
    const SourceTargetPairsType& pairs) {
  std::set<int> seen_replica_ids;
  std::set<std::pair<int64_t, int64_t>> tentative_results;
  // first figure out if we're dealing with a potential forward or backward
  // cycle.
  int forward_edge_counter = 0;
  int backward_edge_counter = 0;
  for (auto pair : pairs) {
    pair.first < pair.second ? forward_edge_counter++ : backward_edge_counter++;
  }
  bool is_forward_cycle = forward_edge_counter > backward_edge_counter;
  for (int64_t i = 0; i < pairs.size(); ++i) {
    const SourceTargetPairType& pair = pairs[i];
    if (is_forward_cycle) {
      // check if the source of the current pair is smaller than the target
      if (pair.first < pair.second) {
        seen_replica_ids.insert(pair.first);
      } else {
        // the source of the current pair is larger than the target, so the
        // current pair may be part of a cycle. We keep track of the target ID
        // and the index of the pair in the original pairs array.
        tentative_results.insert(std::make_pair(pair.second, i));
      }
    } else {
      // The backward cycle check uses similar logic but in reverse.
      if (pair.first > pair.second) {
        seen_replica_ids.insert(pair.second);
      } else {
        tentative_results.insert(std::make_pair(pair.first, i));
      }
    }
  }
  std::set<int> final_results;
  // Iterate over the tentative results and only keep the indices that form an
  // actual cycle. This is done by checking if the target replica ID of the
  // pair is in the set of seen replica IDs. Note that the tentative results
  // array will be fairly small in practice, so this is not adding too much to
  // the runtime.
  for (auto& [replica_id, index] : tentative_results) {
    if (seen_replica_ids.find(replica_id) != seen_replica_ids.end()) {
      final_results.insert(index);
    }
  }
  CycleType cycle_type = final_results.empty() ? CycleType::kNone
                         : is_forward_cycle    ? CycleType::kForward
                                               : CycleType::kBackward;
  return std::make_pair(cycle_type, final_results);
}

bool IsExclusivelyCrossModule(absl::Span<const ReplicaGroup> replica_groups,
                              bool use_global_ids, bool has_channel_id,
                              const DeviceAssignment& device_assignment) {
  if (!has_channel_id) {
    return false;
  }
  if (!use_global_ids) {
    // Each id in a replica group is a replica id. If any group
    // has more than one id then this is not exclusively cross module.
    for (const ReplicaGroup& replica_group : replica_groups) {
      if (replica_group.replica_ids_size() != 1) {
        return false;
      }
    }
    return true;
  }
  // Each id in a replica group is a global id. Check if all replica groups are
  // exclusively cross module (all participants in a group have the same replica
  // id).
  const int64_t partition_count = device_assignment.computation_count();
  for (const ReplicaGroup& replica_group : replica_groups) {
    std::optional<int64_t> first_replica_id;
    for (int64_t global_id : replica_group.replica_ids()) {
      int64_t replica_id = global_id / partition_count;
      if (!first_replica_id.has_value()) {
        first_replica_id = replica_id;
      } else if (replica_id != first_replica_id) {
        return false;
      }
    }
  }
  return true;
}

bool IsExclusivelyCrossReplica(absl::Span<const ReplicaGroup> replica_groups,
                               bool use_global_ids, bool has_channel_id,
                               const DeviceAssignment& device_assignment) {
  if (!has_channel_id) {
    return true;
  }
  const int64_t partition_count = device_assignment.computation_count();
  if (!use_global_ids) {
    // Each id in a replica group is a replica id and we will perform the
    // collective between all devices with that replica id. If partition count
    // is > 1, then this is not exclusively cross replica.
    return partition_count == 1;
  }
  // Each id in a replica group is a global id. Check if all replica groups are
  // exclusively cross replica (all participants in a group have the same
  // partition id).
  for (const ReplicaGroup& replica_group : replica_groups) {
    std::optional<int64_t> first_partition_id;
    for (int64_t global_id : replica_group.replica_ids()) {
      int64_t partition_id = global_id % partition_count;
      if (!first_partition_id.has_value()) {
        first_partition_id = partition_id;
      } else if (partition_id != first_partition_id) {
        return false;
      }
    }
  }
  return true;
}
}  // end namespace xla
