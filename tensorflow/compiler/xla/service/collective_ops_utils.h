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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/blocking_counter.h"

namespace xla {

enum class ReductionKind { SUM, PRODUCT, MIN, MAX };

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

// Figures out which IDs are participating in the collective subgroup.
// An empty `groups` indicates that all [0, total_participant_count) IDs
// are participating. Note that for CollectiveOpGroupMode::kFlattenedID,
// groups cannot be empty, so `total_participant_count` is an optional.
StatusOr<std::vector<int>> GetParticipatingIDs(
    int current_id, std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups);

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

absl::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode);

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids);

// Figures out subgroups of participating devices from given replica_groups and
// group_mode.
//
// Returns list of participants, where each participant is a list of
// GlobalDeviceIds.
//
// For example:
//   device_assignment={{33, 34}, {44, 45}, {55, 56}}  3 replicas 2 partitions
//   group_mode=CollectiveOpGroupMode::kCrossReplica
//   replica_groups={{0}, {1, 2}}
//
//   This functions returns {{33, 34}, {44, 45, 55, 56}}
//   There are 2 subgroups of participating devices {33, 34}, {44, 45, 55, 56}.
StatusOr<std::vector<std::vector<GlobalDeviceId>>>
GetParticipatingDevicesGroups(const DeviceAssignment& device_assignment,
                              absl::Span<const ReplicaGroup> replica_groups,
                              CollectiveOpGroupMode group_mode);

// Same as above, except that it returns the flattened id in the replica groups
// instead of device id.
StatusOr<std::vector<ReplicaGroup>> GetParticipatingFlattenedIdGroups(
    const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Figures out which devices are participating in the collective subgroup.
StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Returns true if the two replica group are orthogonal.
bool ReplicaGroupsOrthogonal(absl::Span<const ReplicaGroup> first,
                             absl::Span<const ReplicaGroup> second);

// A custom call target that can be used to create a nop that can legally
// replace a collective op.
constexpr char kNopCustomCallTarget[] = "AllocateBuffer";

// Returns true if instruction is a collective op or a collective fusion.
bool IsCollective(const HloInstruction* instruction);

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
  explicit ParticipantData(const RendezvousKey& rendezvous_key)
      : rendezvous_key(rendezvous_key) {}

  virtual ~ParticipantData() {}

  RendezvousKey rendezvous_key;

  virtual std::string ToString() const = 0;
};

// Encapsulates parameters to Rendezvous::SubmitParticipant.
struct AllReduceParticipantData : ParticipantData {
  AllReduceParticipantData(const RendezvousKey& rendezvous_key_p,
                           int64_t device_ordinal_p, se::Stream* stream_p)
      : ParticipantData(rendezvous_key_p),
        device_ordinal(device_ordinal_p),
        stream(stream_p) {}

  // TODO(b/125951860): We should vet that we're buffer allocating such that
  // source_buffer == destination_buffer if that avoids a NCCL copy (will depend
  // on how well the NCCL in-place implementation performs vs the out-of-place
  // implementation).
  struct Buffer {
    int64_t element_count;
    se::DeviceMemoryBase source_data;
    se::DeviceMemoryBase destination_data;
    PrimitiveType primitive_type;
  };
  int64_t device_ordinal;
  se::Stream* stream;
  std::vector<Buffer> buffers;

  ReductionKind reduction_kind;

  // For each local all-reduce participant a (global ID, local device ordinal)
  // pair for the participant. Participants are in no particular order.
  std::vector<std::pair<GlobalDeviceId, int64_t>> local_devices;

  std::string ToString() const override {
    std::vector<std::string> buffer_strs;
    buffer_strs.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      buffer_strs.push_back(
          absl::StrFormat("{element_count=%d}", buffer.element_count));
    }
    return absl::StrFormat(
        "AllReduceParticipantData{buffers=[%s], rendezvous_key=%s, "
        "device_ordinal=%d, stream=%p}",
        absl::StrJoin(buffer_strs, ","), rendezvous_key.ToString(),
        device_ordinal, stream);
  }
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
  explicit Rendezvous(const RendezvousKey& k) : key_(k) {}

  // Submit a participant to the rendezvous. We get the rendezvous from
  // `rendezvous_getter`, which we can then use to drop the existing reference.
  static StatusOr<O> SubmitParticipant(
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
  virtual StatusOr<O> RunCollectiveOp(const I& participant) = 0;

  // Initialize the rendezvous by the first ("primary") thread which reaches the
  // barrier. Returns whether this thread is primary.
  bool InitializationBarrier() {
    absl::MutexLock lock(&mu_);
    if (!initialized_) {
      initialized_ = true;
      return true;
    }
    return false;
  }

  absl::Mutex mu_;

  bool initialized_ ABSL_GUARDED_BY(mu_) = false;

  std::vector<I> participants_ ABSL_GUARDED_BY(mu_);

 private:
  // Runs the all-reduce on the given thread.  If successful, returns
  //  - a handle to the clique that was used, so that the caller may keep the
  //    clique alive if it chooses.
  //  - a BlockingCounter initialized to the number of participants, so that
  //    the caller can coordinate with the participants one last time if it
  //    chooses.  This is useful for coordinating destruction of the Rendezvous.
  StatusOr<std::pair<O, std::shared_ptr<tsl::BlockingCounter>>>
  SubmitParticipant(const I& participant) {
    {
      absl::MutexLock lock(&mu_);
      CHECK(!initialized_);

      // Spot check for consistent replica counts among submitting threads.
      if (!participants_.empty() &&
          participants_.back().rendezvous_key != participant.rendezvous_key) {
        return InvalidArgument(
            "Mismatch among all-reduce participants. Expected same "
            "replica-count, element-count, and rendezvous-key but were %s and "
            "%s",
            participants_.back().ToString(), participant.ToString());
      }
      participants_.push_back(participant);
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

}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
