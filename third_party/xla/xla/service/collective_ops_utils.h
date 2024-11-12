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

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/blocking_counter.h"

namespace xla {

enum class ReductionKind { SUM, PRODUCT, MIN, MAX };

constexpr std::string_view ReductionKindToString(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return "sum";
    case ReductionKind::PRODUCT:
      return "prod";
    case ReductionKind::MIN:
      return "min";
    case ReductionKind::MAX:
      return "max";
  }
}

// Attempts to match instruction to one of the possible cases for ReductionKind.
std::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo);

// Attempts to match computation to one of the possible cases in ReductionKind.
std::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation);

// Returns the reduction identity value for a certain ReductionKind and
// PrimitiveType.
std::optional<Literal> GetReductionIdentity(ReductionKind kind,
                                            PrimitiveType type);

// There are broadly 4 modes that collective communication ops use to describe
// which sets of devices are participating with a given device in the operation.
// These modes are determined by the values of channel_id (optional) and
// use_global_device_ids (optional). The modes are as follows:
//
// kCrossReplica:
//    implied by: no channel id, use_global_device_ids = false, or
//                no channel_id, no use_global_device_ids:
//    replica_groups contain replica_id, group contains all replicas for the
//    current partition
//
// kCrossPartition:
//    implied by: channel_id is set, no use_global_device_ids:
//    replica_groups contain partition_id, group contains all partitions for the
//    current replica.
//
// kCrossReplicaAndPartition:
//    implied by: channel_id is set, use_global_device_ids = false:
//    replica_groups contain replica_id, group contains all replicas for all
//    partitions (as opposed to just current partition).
//
// kFlattenedID:
//    implied by: channel_id is set, use_global_device_ids = true:
//    replica_groups contain flattened-ids, group contains devices that are
//    listed in the flattened-id list.
//
// Rest of the combinations are invalid.
//
// Since the actual value of channel_id does not matter, we use a bool argument
// `has_channel_id`, and optional<bool> for use_global_device_ids.
// Note that use_global_device_ids true requires channel_id to be set as well.
// Additionally, if use_global_device_ids = true, replica groups cannot be
// empty (verified in the HLO verifier).
enum class CollectiveOpGroupMode {
  kCrossReplica,
  kCrossPartition,
  kCrossReplicaAndPartition,
  kFlattenedID,
};

// Figures out which IDs are participating in the collective subgroup.
// An empty `groups` indicates that all [0, total_participant_count) IDs
// are participating. Note that for CollectiveOpGroupMode::kFlattenedID,
// groups cannot be empty, so `total_participant_count` is an optional.
absl::StatusOr<std::vector<int>> GetParticipatingIDs(
    CollectiveOpGroupMode group_mode, int current_id,
    std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups);

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode);

// Returns the group formation mode of instr, assuming that instr is, or is
// dervied from, an HloAllGatherInstruction, HloAllReduceInstructionBase,
// HloAllToAllInstruction, HloCollectiveBroadcastInstruction or
// HloCollectivePermuteInstruction.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    const HloInstruction* instr);

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids);

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

// Same as above, except that it returns the flattened id in the replica groups
// instead of device id.
absl::StatusOr<std::vector<ReplicaGroup>> GetParticipatingFlattenedIdGroups(
    const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Same as above, but take replica/partition count instead of device assignment.
absl::StatusOr<std::vector<ReplicaGroup>> GetParticipatingFlattenedIdGroups(
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode replica_group_mode, int replica_count,
    int partition_count);

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

// Returns the collective instruction if argument is a collective op (or a
// collective fusion) with channel_id.
HloInstruction* IsOrHasCollectiveWithChannelId(HloInstruction* instruction);

// Returns true if instruction is a synchronous collective op.
bool IsSyncCollective(const HloInstruction* instr);

// Returns true if the (a, b) pairs form a forward cycle with all participants
// in the cycle, such as {{0,1},{1,2},{2,3},{3,0}}. We assume that the (a, b)
// pairs are ordered as they are generated by SPMD partitioning.
bool IsForwardCycle(const std::vector<std::pair<int64_t, int64_t>>& pairs);

// Returns true if the (a, b) pairs form a backward cycle with all participants
// in the cycle, such as {{0,3},{1,0},{2,1},{3,2}}. We assume that the (a, b)
// pairs are ordered as they are generated by SPMD partitioning.
bool IsBackwardCycle(const std::vector<std::pair<int64_t, int64_t>>& pairs);

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
        run_id.ToString(), GlobalDeviceIdsToString(global_devices),
        num_local_participants, CollectiveOpKindString(), op_id);
  }

  RunId run_id;
  std::vector<GlobalDeviceId> global_devices;
  int num_local_participants;
  CollectiveOpKind collective_op_kind;
  int64_t op_id;
};

template <typename DescFn>
void WaitAndLogIfStuck(tsl::BlockingCounter* counter, const DescFn& desc_fn) {
  VLOG(3) << "Begin: " << desc_fn();
  const std::chrono::milliseconds timeout(5000);
  bool ok = counter->WaitFor(timeout);
  if (ok) {
    VLOG(3) << "Finished: " << desc_fn();
    return;
  }
  LOG(ERROR) << "This thread has been waiting for " << timeout.count()
             << "ms for and may be stuck: " << desc_fn();
  counter->Wait();
  LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                "Perhaps the timeout is too short: "
             << desc_fn();
}

// Participant data for each rendezvous.
struct ParticipantData {
  ParticipantData(const RendezvousKey& rendezvous_key, int local_rank)
      : rendezvous_key(rendezvous_key), local_rank(local_rank) {}

  virtual ~ParticipantData() {}

  RendezvousKey rendezvous_key;
  int local_rank;  // Which of the local participants is this?

  virtual std::string ToString() const = 0;
};

// The set of threads that want to do a collective op together all pick the same
// Rendezvous object out of the global cache and call SubmitParticipant.
//
// The Rendezvous instance handles waiting for all threads to join, ensuring
// that a clique exists for the desired set of GPUs, etc.
//
// Rendezvous objects can only be used once.
//
// I: Participant data.
// O: Participant output.
template <typename I, typename O,
          typename =
              std::enable_if_t<std::is_base_of<ParticipantData, I>::value>>
class Rendezvous {
 public:
  virtual ~Rendezvous() {}
  explicit Rendezvous(const RendezvousKey& k)
      : participants_(k.num_local_participants), key_(k) {}

  // Submit a participant to the rendezvous. We get the rendezvous from
  // `rendezvous_getter`, which we can then use to drop the existing reference.
  static absl::StatusOr<O> SubmitParticipant(
      absl::FunctionRef<std::shared_ptr<Rendezvous<I, O>>()> rendezvous_getter,
      I participant) {
    std::shared_ptr<Rendezvous<I, O>> rendezvous = rendezvous_getter();
    TF_ASSIGN_OR_RETURN(auto p, rendezvous->SubmitParticipant(participant));

    // Drop our reference to the Rendezvous and wait for all other threads to do
    // the same.  If we didn't do this, one of the threads could run past this
    // point, reenter ExecuteOnStream for another all-reduce, and attempt to
    // reuse the Rendezvous!
    //
    // An alternative way of accomplishing this goal would be to implement
    // RefcountingHashMap::erase() and call it during SubmitParticipant.  But
    // erase() is deceptively complex to implement correctly.
    std::shared_ptr<tsl::BlockingCounter> blocking_counter = p.second;
    rendezvous.reset();
    blocking_counter->DecrementCount();
    xla::WaitAndLogIfStuck(blocking_counter.get(), [&] {
      return absl::StrFormat(
          "participant waiting for all threads to drop their reference to the "
          "rendezvous: %p",
          rendezvous.get());
    });
    return std::move(p.first);
  }

 protected:
  // Returns domain-specific output O and whether this replica is primary.
  virtual absl::StatusOr<O> RunCollectiveOp(const I& participant) = 0;

  // Adding participants_ requires holding mu_.
  // Not annotated with ABSL_GUARDED_BY(mu_) because we do not require the lock
  // to be held during CollectiveOp(), since at that point all the data is known
  // to be present due to the global barrier.
  std::vector<std::optional<I>> participants_;

 private:
  absl::Mutex mu_;

  // Runs the all-reduce on the given thread.  If successful, returns
  //  - a handle to the clique that was used, so that the caller may keep the
  //    clique alive if it chooses.
  //  - a BlockingCounter initialized to the number of participants, so that
  //    the caller can coordinate with the participants one last time if it
  //    chooses.  This is useful for coordinating destruction of the Rendezvous.
  absl::StatusOr<std::pair<O, std::shared_ptr<tsl::BlockingCounter>>>
  SubmitParticipant(const I& participant) {
    {
      absl::MutexLock lock(&mu_);
      CHECK(!participants_[participant.local_rank].has_value());
      participants_[participant.local_rank] = participant;
    }

    // Wait for all participants to arrive.
    all_participants_present_.DecrementCount();
    WaitAndLogIfStuck(&all_participants_present_, [&] {
      return absl::StrFormat(
          "participant %s waiting for all participants to arrive at rendezvous "
          "%s",
          participant.ToString(), key_.ToString());
    });

    TF_ASSIGN_OR_RETURN(O output, RunCollectiveOp(participant));
    return std::make_pair(std::move(output), returned_blocking_counter_);
  }

  const RendezvousKey key_;

  tsl::BlockingCounter all_participants_present_{key_.num_local_participants};

  // tsl::BlockingCounter returned by SubmitParticipant.
  std::shared_ptr<tsl::BlockingCounter> returned_blocking_counter_{
      std::make_shared<tsl::BlockingCounter>(key_.num_local_participants)};
};

// We only pipeline Send-Recv chains with channel_id > 0, where each chain
// has a unique channel_id, and allows multiple Send-Recv chains using
// channel_id 0.
inline bool MayPipelineSendRecvChannel(int64_t channel_id) {
  return channel_id > 0;
}

constexpr char kSendRecvSourceTargetPairsAttr[] =
    "_xla_send_recv_source_target_pairs";

// When a Send or Recv is annotated with frontend attribute
// _xla_send_recv_pipeline="1", asynchronous stream kP2P1 is used to execute the
// Send or Recv. For all other cases, asynchronous stream kP2P0 is used.
constexpr char kSendRecvPipelineAttr[] = "_xla_send_recv_pipeline";

// This frontend attribute conveys the following information:
// (1) _xla_send_recv_validation="invalid": the runtime should skip sending or
// receiving data when the instruction is executed.
// (2) the absent of the attribute: the runtime should faithfully perform the
// Send or Recv operation when the instruction is executed.
// (3) _xla_send_recv_validation={list-of-bounds}: the list-of-bounds
// corresponds to the value of _xla_send_recv_source_target_pairs, and specifies
// the execution instances for which the runtime should faithfully perform the
// Send or Recv operation. Here is an example:
//   _xla_send_recv_source_target_pairs={{0,1}, {1,2}}
//   _xla_send_recv_validation={{2,3}, {5,7}}
// The Send or Recv instruction with the above two attributes have the
// following semantics:
// The communication between device 0 and 1 will only send or receive data
// for execution instances 2 and 3 of the instruction on devices 0 and 1.
// For execution instances 0, 1, and beyond 3, the runtime should skip sending
// or receiving any data.
// Similarly, the communication between device 1 and 2 will only send or
// receive data on execution instances 5 and 7.
constexpr char kSendRecvValidationAttr[] = "_xla_send_recv_validation";

}  // end namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
