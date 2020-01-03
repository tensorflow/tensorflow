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

#include "tensorflow/compiler/xla/service/gpu/collective_permute_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <map>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"

// This thunk's implementation is somewhat similar to our implementation of
// AllReduce using NCCL.  One reason it's separate is that, because this doesn't
// use NCCL, it can work even without a CUDA compiler.

namespace xla {
namespace gpu {

namespace {

using tensorflow::BlockingCounter;

// This same function appears in nccl_all_reduce_thunk.  I've copy/pasted it
// here primarily because I want the VLOGs to work.
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

// Key for looking up a Rendezvous object in our global map.
//
// Morally, the key is just a RunId.  num_participants is in this struct only
// because we use that information when constructing the Rendezvous.
struct RendezvousKey {
  RunId run_id;
  int num_participants;  // int, not int64, to match BlockingCounter's counter.

  string ToString() const {
    return absl::StrFormat("RendezvousKey{run_id=%s, num_participants=%d}",
                           run_id.ToString(), num_participants);
  }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }
};

// Information about a thread that's participating in a collective-permute
// operation.
struct ParticipantData {
  int64 replica_id;
  se::Stream* stream;

  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;

  // ReplicaIds to which we will copy the data in src.
  std::vector<int64> dest_replicas;
};

// The set of threads that want to do a collective permute together all pick the
// same Rendezvous object out of the global cache and call SubmitParticipant.
//
// The Rendezvous instance handles waiting for all threads to join and then
// doing the actual work of the collective permute.
//
// Rendezvous objects can only be used once.
class Rendezvous {
 public:
  explicit Rendezvous(const RendezvousKey& key) : key_(key) {}

  // Runs the collective permute on the given thread.
  //
  // If successful, returns a BlockingCounter initialized to the number of
  // participants, so that the caller can coordinate with the participants one
  // last time if it chooses.  This is useful for coordinating destruction of
  // the Rendezvous.
  StatusOr<std::shared_ptr<BlockingCounter>> SubmitParticipant(
      ParticipantData participant);

 private:
  const RendezvousKey key_;
  BlockingCounter all_arrived_{key_.num_participants};

  // BlockingCounter returned by SubmitParticipant.
  std::shared_ptr<BlockingCounter> returned_blocking_counter_{
      std::make_shared<BlockingCounter>(key_.num_participants)};

  tensorflow::mutex mu_;
  bool initialized_ GUARDED_BY(mu_) = false;

  // We use an std::map so that we can iterate over it below in a guaranteed
  // order.  The order shouldn't actually matter, but why be nondeterministic if
  // we don't have to be?
  std::map<int64, ParticipantData> participants_ GUARDED_BY(mu_);
};

void EnqueueCopy(se::DeviceMemoryBase src, se::Stream* src_stream,
                 se::DeviceMemoryBase dest, se::Stream* dest_stream) {
  CHECK_EQ(src.size(), dest.size());

  // If src_stream == dest_stream, we're doing a plain memcpy from one GPU back
  // to the same GPU.  x->ThenWaitFor(x) is illegal, so this has to be a special
  // case.
  if (src_stream == dest_stream) {
    dest_stream->ThenMemcpyD2D(&dest, src, src.size());
    return;
  }

  // We (arbitrarily) choose to use the dest stream do perform the copy.  This
  // means we need to make the dest stream wait until the src stream is ready
  // before it performs the copy, and then we need to make the src stream wait
  // until the dest stream has completed the copy.
  dest_stream->ThenWaitFor(src_stream).ThenMemcpyD2D(&dest, src, src.size());
  src_stream->ThenWaitFor(dest_stream);
}

StatusOr<std::shared_ptr<BlockingCounter>> Rendezvous::SubmitParticipant(
    ParticipantData participant) {
  bool primary;
  {
    tensorflow::mutex_lock lock(mu_);
    CHECK(participants_.emplace(participant.replica_id, participant).second);

    // The first thread to acquire the lock is designated as the primary.
    primary = !initialized_;

    if (primary) {
      initialized_ = true;
      returned_blocking_counter_ =
          std::make_shared<BlockingCounter>(key_.num_participants);
    }
  }

  // Wait for all participants to arrive.  Even though our copies are async and
  // enqueued by just one thread, this is not optional!  If we didn't wait for
  // everyone, then we wouldn't be able to enqueue the copies at the correct
  // point in their streams.
  all_arrived_.DecrementCount();
  WaitAndLogIfStuck(&all_arrived_, [&] {
    return absl::StrFormat(
        "participant for replica %d (stream %p, device %d) waiting for all "
        "other participants to arrive: %s",
        participant.replica_id, participant.stream,
        participant.stream->parent()->device_ordinal(), key_.ToString());
  });

  // Schedule the copies between the devices.  This is much easier to reason
  // about if we schedule all of the copies from just one thread.  The copies
  // are async anyway, so the number of host threads we use isn't important.
  if (primary) {
    tensorflow::mutex_lock lock(mu_);
    for (const auto& kv : participants_) {
      const ParticipantData& src_participant = kv.second;
      for (int64 dest_replica : src_participant.dest_replicas) {
        const ParticipantData& dest_participant =
            participants_.at(dest_replica);
        EnqueueCopy(src_participant.src, src_participant.stream,
                    dest_participant.dest, dest_participant.stream);
      }
    }
  }

  return returned_blocking_counter_;
}

// Global map of Rendezvous objects.  A thread participating in a collective op
// looks up its Rendezvous in this map to find the other threads that it's
// participating with.
//
// Rendezvous objects are one-time use, so they're removed from this map once
// we're through with them.
RefcountingHashMap<RendezvousKey, Rendezvous>& GlobalRendezvousMap() {
  static auto& m = *new RefcountingHashMap<RendezvousKey, Rendezvous>(
      [](const RendezvousKey& key) {
        return absl::make_unique<Rendezvous>(key);
      });
  return m;
}

}  // anonymous namespace

CollectivePermuteThunk::CollectivePermuteThunk(
    const BufferAllocation::Slice& src, const BufferAllocation::Slice& dest,
    const HloInstruction* instr)
    : Thunk(kCollectivePermute, instr), src_(src), dest_(dest) {}

Status CollectivePermuteThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto* instr = Cast<HloCollectivePermuteInstruction>(hlo_instruction());
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  // Rendezvous with the threads for all other devices that are participating in
  // this CollectivePermute.
  RendezvousKey key{params.run_id, params.device_assn->replica_count()};
  std::shared_ptr<Rendezvous> rendezvous = GlobalRendezvousMap()[key];

  TF_ASSIGN_OR_RETURN(int64 replica_id,
                      params.device_assn->ReplicaIdForDeviceOrdinal(
                          params.stream->parent()->device_ordinal()));

  // Figure out which replicas our data is copied to.
  std::vector<int64> dest_replicas;
  for (const auto& src_dest : instr->source_target_pairs()) {
    if (src_dest.first == replica_id) {
      dest_replicas.push_back(src_dest.second);
    }
  }

  auto src_addr = params.buffer_allocations->GetDeviceAddress(src_);
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  ParticipantData participant{replica_id, params.stream, src_addr, dest_addr,
                              dest_replicas};
  TF_ASSIGN_OR_RETURN(std::shared_ptr<BlockingCounter> final_sync,
                      rendezvous->SubmitParticipant(participant));

  // If no replica writes into us (i.e. we aren't the target of any copies), our
  // contract is that we zero our output.
  if (absl::c_none_of(instr->source_target_pairs(),
                      [&](std::pair<int64, int64> src_dest) {
                        return src_dest.second == replica_id;
                      })) {
    params.stream->ThenMemZero(&dest_addr, dest_addr.size());
  }

  // Drop our reference to the Rendezvous and wait for all other threads to do
  // the same.  If we didn't do this, one of the threads could run past this
  // point, reenter ExecuteOnStream for another collective-permute, and attempt
  // to reuse the Rendezvous!
  //
  // An alternative way of accomplishing this goal would be to implement
  // RefcountingHashMap::erase() and call it during SubmitParticipant.  But
  // erase() is deceptively complex to implement correctly.
  rendezvous.reset();
  final_sync->DecrementCount();
  WaitAndLogIfStuck(final_sync.get(), [&] {
    return absl::StrFormat(
        "participant for replica %d (stream %p, device ordinal %d) waiting for "
        "all threads to drop their reference to the rendezvous: %s",
        participant.replica_id, participant.stream,
        participant.stream->parent()->device_ordinal(), key.ToString());
  });
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
