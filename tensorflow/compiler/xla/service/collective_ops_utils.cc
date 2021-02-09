/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#include "tensorflow/compiler/xla/service/global_device_id.h"

namespace xla {

absl::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation) {
  namespace m = match;
  const HloInstruction* root = computation->root_instruction();

  auto match_opcode = [&](HloOpcode opcode) {
    return Match(
        root, m::Op()
                  .WithOpcode(opcode)
                  .WithBinaryOperandsAnyOrder(m::Parameter(0), m::Parameter(1))
                  .WithShape(m::Shape().IsEffectiveScalar()));
  };

  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type = computation->root_instruction()->shape().element_type();
  if (match_opcode(HloOpcode::kAdd)) {
    return ReductionKind::SUM;
  } else if (match_opcode(HloOpcode::kMultiply)) {
    return ReductionKind::PRODUCT;
  } else if (match_opcode(HloOpcode::kMinimum) ||
             (type == PRED && match_opcode(HloOpcode::kAnd))) {
    return ReductionKind::MIN;
  } else if (match_opcode(HloOpcode::kMaximum) ||
             (type == PRED && match_opcode(HloOpcode::kOr))) {
    return ReductionKind::MAX;
  } else {
    return absl::nullopt;
  }
}

StatusOr<std::vector<int>> GetParticipatingReplicas(
    int replica_id, int total_replica_count,
    absl::Span<const ReplicaGroup> replica_groups) {
  // Empty replica_groups() means that all replicas participate.
  if (replica_groups.empty()) {
    std::vector<int> all_replicas(total_replica_count);
    absl::c_iota(all_replicas, 0);
    return all_replicas;
  }

  // Figure out the other replicas that go together with this one.
  absl::optional<ReplicaGroup> replica_group;
  for (const ReplicaGroup& g : replica_groups) {
    if (absl::c_linear_search(g.replica_ids(), replica_id)) {
      TF_RET_CHECK(!replica_group.has_value())
          << "Replica " << replica_id << " appears twice in replica groups";
      replica_group = g;
    }
  }
  TF_RET_CHECK(replica_group.has_value())
      << "Replica " << replica_id << " doesn't appear in replica groups";
  return std::vector<int>(replica_group->replica_ids().begin(),
                          replica_group->replica_ids().end());
}

StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    int total_replica_count, absl::Span<const ReplicaGroup> replica_groups) {
  std::vector<GlobalDeviceId> participants;
  // Fast path for common case, avoiding logical IDs lookup.
  if (replica_groups.empty() && device_assignment.computation_count() == 1) {
    participants.reserve(total_replica_count);
    for (int replica_id = 0; replica_id < total_replica_count; ++replica_id) {
      participants.emplace_back(
          device_assignment(replica_id, /*computation_id=*/0));
    }
    return participants;
  }

  std::pair<int, int> logical_ids;
  TF_ASSIGN_OR_RETURN(logical_ids,
                      device_assignment.LogicalIdsForDevice(device_id));
  int replica_id = logical_ids.first;
  int computation_id = logical_ids.second;
  TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                      GetParticipatingReplicas(replica_id, total_replica_count,
                                               replica_groups));

  participants.reserve(participating_replicas.size());
  for (int replica_id : participating_replicas) {
    participants.emplace_back(device_assignment(replica_id, computation_id));
  }
  return participants;
}

}  // end namespace xla
