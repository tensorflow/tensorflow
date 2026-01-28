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

#ifndef XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/literal.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_permute_cycle.h"
#include "xla/service/computation_placer.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/source_target_pairs.h"

namespace xla {

absl::StatusOr<ReductionKind> StringToReductionKind(
    absl::string_view reduction_kind);

// Attempts to match instruction to one of the possible cases for ReductionKind.
std::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo);

// Attempts to match computation to one of the possible cases in ReductionKind.
std::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation);

// Creates a computation that performs a reduction operation.
std::unique_ptr<HloComputation> MakeReductionComputation(
    ReductionKind reduction_kind, PrimitiveType element_type);

// Returns the HloOpcode corresponding to the given ReductionKind.
std::optional<HloOpcode> ReductionKindToOpcode(ReductionKind reduction_kind);

// Returns the reduction identity value for a certain ReductionKind and
// PrimitiveType.
std::optional<Literal> GetReductionIdentity(ReductionKind kind,
                                            PrimitiveType type);

// Figures out which IDs are participating in the collective subgroup.
// An empty `groups` indicates that all [0, total_participant_count) IDs
// are participating. Note that for CollectiveOpGroupMode::kFlattenedID,
// groups cannot be empty, so `total_participant_count` is an optional.
absl::StatusOr<std::vector<int>> GetParticipatingIDs(
    CollectiveOpGroupMode group_mode, int current_id,
    std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups);

// Returns the replica groups for the given async collective instruction.
absl::StatusOr<std::vector<std::vector<int64_t>>> GetAsyncReplicaGroups(
    const HloInstruction* instruction);

const CollectiveDeviceListBase& GetCollectiveDeviceList(
    const HloInstruction* hlo);

const std::vector<ReplicaGroup>& GetCollectiveReplicaGroups(
    const HloInstruction* hlo);

// Returns the group formation mode of instr, assuming that instr is, or is
// derived from on the following instructions:
//   * HloAllGatherInstruction
//   * HloAllReduceInstructionBase
//   * HloAllToAllInstruction
//   * HloCollectiveBroadcastInstruction
//   * HloCollectivePermuteInstruction
//   * HloRaggedAllToAllInstruction
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    const HloInstruction* instr);

// Figures out subgroups of participating devices from given replica_groups and
// group_mode.
//
// Returns list of participants, where each participant is a list of
// GlobalDeviceIds.
//
// For example:
//   device_assignment={{33, 34}, {44, 45}, {55, 56}}  3 replicas 2 partitions
//   replica_groups={{0}, {1, 2}}
//   group_mode=CollectiveOpGroupMode::kCrossReplica
//
//   This functions returns {{33}, {34}, {44, 45}, {55, 56}}.
//   Partition 0 has 2 subgroups of participating devices {33}, {44, 55} and
//   partition 1 has 2 subgroups of participating devices {34}, {45, 56}.
//
// Another example:
//   device_assignment={{33, 34}, {44, 45}, {55, 56}}  3 replicas 2 partitions
//   replica_groups={{0}, {1, 2}, {3, 4, 5}}
//   group_mode=CollectiveOpGroupMode::kFlattenedID
//
//   This functions returns {{33}, {34, 44}, {45, 55, 56}}. The replica_ids map
//   into a flattened version of device_assignment.
absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const DeviceAssignment& device_assignment,
                              absl::Span<const ReplicaGroup> replica_groups,
                              CollectiveOpGroupMode group_mode);

// Same as above, except taking an HloInstruction instead.
absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const HloInstruction* collective);

// Same as above, except that it returns the flattened id in the replica groups
// instead of device id.
absl::StatusOr<std::unique_ptr<CollectiveDeviceListBase>>
GetParticipatingFlattenedIdGroups(
    const DeviceAssignment& device_assignment,
    const CollectiveDeviceListBase& collective_device_list,
    CollectiveOpGroupMode group_mode);

// Same as above, but take replica/partition count instead of device assignment.
absl::StatusOr<std::unique_ptr<CollectiveDeviceListBase>>
GetParticipatingFlattenedIdGroups(
    const CollectiveDeviceListBase& collective_device_list,
    CollectiveOpGroupMode group_mode, int replica_count, int partition_count);

// Same as above, with collective group mode determined by the collective
// instruction.
absl::StatusOr<std::unique_ptr<CollectiveDeviceListBase>>
GetParticipatingFlattenedIdGroups(const HloInstruction* hlo,
                                  const DeviceAssignment& device_assignment);

// Figures out which devices are participating in the collective subgroup.
absl::StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Figures out how many ranks are participating in each collective subgroup.
absl::StatusOr<std::vector<int64_t>> GetPariticipantCountsForReplicaGroups(
    int64_t num_replicas, int64_t num_partitions,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

absl::StatusOr<std::optional<std::pair<int64_t, int64_t>>>
GetReplicaGroupCountAndSize(const HloInstruction* hlo);

// Returns true if the two replica group are orthogonal.
bool ReplicaGroupsOrthogonal(absl::Span<const ReplicaGroup> first,
                             absl::Span<const ReplicaGroup> second);

// Returns true if the two replica group are Equal.
bool ReplicaGroupsEqual(absl::Span<const ReplicaGroup> first,
                        absl::Span<const ReplicaGroup> second);

// Returns true if all subgroups in replica_groups are exclusively cross-module.
bool IsExclusivelyCrossModule(absl::Span<const ReplicaGroup> replica_groups,
                              bool use_global_ids, bool has_channel_id,
                              const DeviceAssignment& device_assignment);

// Returns true if all subgroups in replica_groups are exclusively
// cross-replica.
bool IsExclusivelyCrossReplica(absl::Span<const ReplicaGroup> replica_groups,
                               bool use_global_ids, bool has_channel_id,
                               const DeviceAssignment& device_assignment);

bool HasDuplicateSourcesOrTargets(const SourceTargetPairs& pairs);

// A custom call target that can be used to create a nop that can legally
// replace a collective op.
inline constexpr absl::string_view kNopCustomCallTarget = "AllocateBuffer";
// A custom call target that can be used to create a nop that can legally
// replace a collective op and it returns a token.
inline constexpr absl::string_view kNopReturnTokenCustomCallTarget =
    "NopReturnToken";

// Returns true if instruction is a collective op that is not a collective
// fusion.
bool IsNonFusionCollective(const HloInstruction* instruction);

// Returns true if instruction is a collective op or a collective fusion.
bool IsCollective(const HloInstruction* instruction);

// Returns true if instruction is an async collective op.
absl::StatusOr<bool> IsAsyncCollective(const HloInstruction* instruction);

// Returns the collective instruction if argument is a collective op (or a
// collective fusion) with channel_id.
HloInstruction* IsOrHasCollectiveWithChannelId(HloInstruction* instruction);

// Key that identifies a particular Rendezvous object in our global hashtable.
// This determines which calls to ExecuteOnStream communicate with each other.
// The rules are as follows.
//
// * Only ops with the same RunId can communicate with each other. (This is the
//   whole purpose of RunId).
//
// * Only ops with the same set of participating replicas can communicate with
//   each other.  This is how we separate out different replica groups (e.g. a
//   single AllReduce HLO might do two reductions, between say GPUs {0,2} and
//   {1,3}).
//
// * Only ops with the same opcode can communicate with each other.  At the
//   moment we only support kAllReduce, so we don't check for this explicitly.
//
// * For cross-module all-reduces (i.e. instr->channel_id().has_value()),
//   only ops with the same value for channel_id() can communicate with each
//   other.
//
// * For cross-replica (i.e. same-module) all-reduces (i.e.
//   !channel_id().has_value()), only ops from the same module (as
//   identified by its unique_id()) can communicate with each other.
//
struct RendezvousKey {
  enum CollectiveOpKind {
    kCrossModule,
    kCrossReplica,
  };

  explicit RendezvousKey(const RunId& run_id,
                         std::vector<GlobalDeviceId> global_devices,
                         int num_local_participants,
                         CollectiveOpKind collective_op_kind, int64_t op_id)
      : run_id(run_id),
        global_devices(std::move(global_devices)),
        num_local_participants(num_local_participants),
        collective_op_kind(collective_op_kind),
        op_id(op_id) {}

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.global_devices,
                      k.num_local_participants, k.collective_op_kind, k.op_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id && a.global_devices == b.global_devices &&
           a.num_local_participants == b.num_local_participants &&
           a.collective_op_kind == b.collective_op_kind &&  //
           a.op_id == b.op_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }

  absl::string_view CollectiveOpKindString() const {
    switch (collective_op_kind) {
      case kCrossModule:
        return "cross_module";
      case kCrossReplica:
        return "cross_replica";
    }
  }

  std::string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, global_devices=[%s], "
        "num_local_participants=%d, collective_op_kind=%s, op_id=%d}",
        run_id.ToString(), absl::StrJoin(global_devices, ", "),
        num_local_participants, CollectiveOpKindString(), op_id);
  }

  RunId run_id;
  std::vector<GlobalDeviceId> global_devices;
  int num_local_participants;
  CollectiveOpKind collective_op_kind;
  int64_t op_id;
};

// We only pipeline Send-Recv chains with channel_id > 0, where each chain
// has a unique channel_id, and allows multiple Send-Recv chains using
// channel_id 0.
inline bool MayPipelineSendRecvChannel(int64_t channel_id) {
  return channel_id > 0;
}

// When a Send or Recv is annotated with frontend attribute
// _xla_send_recv_pipeline="1", asynchronous stream kP2P1 is used to execute the
// Send or Recv. For all other cases, asynchronous stream kP2P0 is used.
constexpr char kSendRecvPipelineAttr[] = "_xla_send_recv_pipeline";

// Attribute to indicate that collective operations should be issued on a
// dedicated p2p stream. This is a hint and there is no guarantee that this will
// be honored.
inline constexpr absl::string_view kCollectiveStreamAttrName =
    "_xla_gpu_collective_stream";
inline constexpr absl::string_view kCollectiveStreamP2P = "p2p";

// Returns latency metadata in microseconds(us) if the instruction is a custom
// call with latency metadata. Returns `std::nullopt` if the instruction is not
// a custom call with latency metadata or invalid latency metadata is provided.
std::optional<double> GetCustomCallLatencyMetadata(const HloInstruction* instr);

int64_t GetSubgroupSize(const HloCollectiveInstruction* hlo,
                        CollectiveOpGroupMode group_mode);

}  // end namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
