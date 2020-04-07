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
#include <vector>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

enum class ReductionKind { SUM, PRODUCT, MIN, MAX };

// Attempts to match computation to one of the possible cases in ReductionKind.
absl::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation);

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_id.
StatusOr<std::vector<int64>> GetParticipatingReplicas(
    GlobalDeviceId device_id, absl::Span<const ReplicaGroup> replica_groups,
    int64 total_replica_count, const DeviceAssignment& device_assn);

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
                         CollectiveOpKind collective_op_kind, int64 op_id)
      : run_id(run_id),
        global_devices(std::move(global_devices)),
        num_local_participants(num_local_participants),
        collective_op_kind(collective_op_kind),
        op_id(op_id) {}

  static RendezvousKey FromInstruction(
      const RunId& run_id, std::vector<GlobalDeviceId> global_devices,
      int num_local_participants, const HloInstruction* instr) {
    CollectiveOpKind collective_op_kind;
    int64 op_id;

    std::tie(collective_op_kind, op_id) =
        instr->channel_id().has_value()
            ? std::make_pair(kCrossModule, instr->channel_id().value())
            : std::make_pair(
                  kCrossReplica,
                  static_cast<int64>(instr->GetModule()->unique_id()));
    return RendezvousKey(run_id, std::move(global_devices),
                         num_local_participants, collective_op_kind, op_id);
  }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.global_devices,
                      k.num_local_participants,
                      static_cast<int>(k.collective_op_kind), k.op_id);
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

  string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, global_devices=[%s], "
        "num_local_participants=%d, collective_op_kind=%d, op_id=%d}",
        run_id.ToString(), GlobalDeviceIdsToString(global_devices),
        num_local_participants, static_cast<int>(collective_op_kind), op_id);
  }

  RunId run_id;
  std::vector<GlobalDeviceId> global_devices;
  int num_local_participants;
  CollectiveOpKind collective_op_kind;
  int64 op_id;
};

template <typename DescFn>
void WaitAndLogIfStuck(tensorflow::BlockingCounter* counter,
                       const DescFn& desc_fn) {
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
  LOG(ERROR) << "Thread is unstuck!  Warning above was a false-positive.  "
                "Perhaps the timeout is too short: "
             << desc_fn();
}

// Encapsulates parameters to Rendezvous::SubmitParticipant.
struct AllReduceParticipantData {
  explicit AllReduceParticipantData(RendezvousKey rendezvous_key)
      : rendezvous_key(rendezvous_key) {}

  int64 device_ordinal;
  RendezvousKey rendezvous_key;

  // TODO(b/125951860): We should vet that we're buffer allocating such that
  // source_buffer == destination_buffer if that avoids a NCCL copy (will depend
  // on how well the NCCL in-place implementation performs vs the out-of-place
  // implementation).
  struct Buffer {
    int64 element_count;
    se::DeviceMemoryBase source_data;
    se::DeviceMemoryBase destination_data;
    PrimitiveType primitive_type;
  };
  std::vector<Buffer> buffers;
  se::Stream* stream;
  const NcclUniqueIdCallback* nccl_unique_id_callback = nullptr;

  ReductionKind reduction_kind;

  // For each local all-reduce participant a (global ID, local device ordinal)
  // pair for the participant. Participants are in no particular order.
  std::vector<std::pair<GlobalDeviceId, int64>> local_devices;

  string ToString() const {
    std::vector<std::string> buffer_strs;
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
// O: Participant output.
template <typename O>
class Rendezvous {
 public:
  virtual ~Rendezvous() {}
  explicit Rendezvous(const RendezvousKey& k) : key_(k) {}

  // Submit a participant to the rendezvous. We get the rendezvous from
  // `rendezvous_getter`, which we can then use to drop the existing reference.
  static StatusOr<O> SubmitParticipant(
      std::function<std::shared_ptr<Rendezvous<O>>()> rendezvous_getter,
      AllReduceParticipantData participant) {
    std::shared_ptr<Rendezvous<O>> rendezvous = rendezvous_getter();
    TF_ASSIGN_OR_RETURN(auto p, rendezvous->SubmitParticipant(participant));

    // Drop our reference to the Rendezvous and wait for all other threads to do
    // the same.  If we didn't do this, one of the threads could run past this
    // point, reenter ExecuteOnStream for another all-reduce, and attempt to
    // reuse the Rendezvous!
    //
    // An alternative way of accomplishing this goal would be to implement
    // RefcountingHashMap::erase() and call it during SubmitParticipant.  But
    // erase() is deceptively complex to implement correctly.
    std::shared_ptr<tensorflow::BlockingCounter> blocking_counter = p.second;
    rendezvous.reset();
    blocking_counter->DecrementCount();
    xla::WaitAndLogIfStuck(blocking_counter.get(), [&] {
      return absl::StrFormat(
          "participant waiting for all threads to drop their reference to the "
          "rendezvous: %p",
          rendezvous.get());
    });
    return p.first;
  }

 protected:
  // Returns domain-specific output O and whether this replica is primary.
  virtual StatusOr<std::pair<O, bool>> SubmitParticipantImpl(
      AllReduceParticipantData participant) = 0;

  virtual void CleanupImpl(O handle, bool is_primary) {}

  tensorflow::mutex mu_;

  bool initialized_ TF_GUARDED_BY(mu_) = false;

  std::vector<AllReduceParticipantData> participants_ TF_GUARDED_BY(mu_);

 private:
  // Runs the all-reduce on the given thread.  If successful, returns
  //  - a handle to the clique that was used, so that the caller may keep the
  //    clique alive if it chooses.
  //  - a BlockingCounter initialized to the number of participants, so that
  //    the caller can coordinate with the participants one last time if it
  //    chooses.  This is useful for coordinating destruction of the Rendezvous.
  StatusOr<std::pair<O, std::shared_ptr<tensorflow::BlockingCounter>>>
  SubmitParticipant(AllReduceParticipantData participant) {
    {
      tensorflow::mutex_lock lock(mu_);
      CHECK(!initialized_);

      // Spot check for consistent replica counts among submitting threads.
      if (!participants_.empty() &&
          (participants_.back().buffers.size() != participant.buffers.size() ||
           participants_.back().rendezvous_key != participant.rendezvous_key)) {
        return InvalidArgument(
            "Mismatch among all-reduce participants.  Expected same "
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
          "participant for device ordinal %d, stream %p waiting for all "
          "participants to arrive at rendezvous %s",
          participant.device_ordinal, participant.stream, key_.ToString());
    });

    StatusOr<std::pair<O, bool>> p_or = SubmitParticipantImpl(participant);

    done_.DecrementCount();
    if (!p_or.ok()) {
      return p_or.status();
    }
    std::pair<O, bool> p = p_or.ValueOrDie();

    O handle = p.first;
    bool is_primary = p.second;

    // The primary owns the lock on the NCCL clique.  Hold it until all threads
    // are done.  (We'll release it when we return from this function.)
    if (is_primary) {
      WaitAndLogIfStuck(&done_, [&] {
        return absl::StrFormat(
            "primary participant waiting for all other participants to "
            "complete all-reduce %s",
            key_.ToString());
      });
    }

    CleanupImpl(handle, is_primary);

    return std::make_pair(handle, returned_blocking_counter_);
  }
  const RendezvousKey key_;

  tensorflow::BlockingCounter all_participants_present_{
      key_.num_local_participants};
  tensorflow::BlockingCounter done_{key_.num_local_participants};

  // tensorflow::BlockingCounter returned by SubmitParticipant.
  std::shared_ptr<tensorflow::BlockingCounter> returned_blocking_counter_{
      std::make_shared<tensorflow::BlockingCounter>(
          key_.num_local_participants)};
};

}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
