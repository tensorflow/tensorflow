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

  if (match_opcode(HloOpcode::kAdd)) {
    return ReductionKind::SUM;
  } else if (match_opcode(HloOpcode::kMultiply)) {
    return ReductionKind::PRODUCT;
  } else if (match_opcode(HloOpcode::kMinimum)) {
    return ReductionKind::MIN;
  } else if (match_opcode(HloOpcode::kMaximum)) {
    return ReductionKind::MAX;
  } else {
    return absl::nullopt;
  }
}

StatusOr<std::vector<int64>> GetParticipatingReplicas(
    int64 device_ordinal, absl::Span<const ReplicaGroup> replica_groups,
    int64 total_replica_count, const DeviceAssignment& device_assn) {
  std::vector<int64> participating_replicas;

  // Empty replica_groups() means that all replicas participate in one big
  // group.
  if (replica_groups.empty()) {
    participating_replicas.resize(total_replica_count);
    absl::c_iota(participating_replicas, 0);
    return participating_replicas;
  }

  // Use the DeviceAssignment to figure out our replica-id.
  TF_ASSIGN_OR_RETURN(int replica_id,
                      device_assn.ReplicaIdForDeviceOrdinal(device_ordinal));

  // Figure out the other replicas that go together with this one.
  absl::optional<ReplicaGroup> replica_group;
  for (const ReplicaGroup& g : replica_groups) {
    if (absl::c_linear_search(g.replica_ids(), replica_id)) {
      CHECK(!replica_group.has_value())
          << "Replica " << replica_id << " appears twice in replica groups";
      replica_group = g;
    }
  }
  CHECK(replica_group.has_value())
      << "Replica " << replica_id << " doesn't appear in replica groups? ";

  participating_replicas.insert(participating_replicas.begin(),
                                replica_group->replica_ids().begin(),
                                replica_group->replica_ids().end());
  return participating_replicas;
}

}  // end namespace xla
