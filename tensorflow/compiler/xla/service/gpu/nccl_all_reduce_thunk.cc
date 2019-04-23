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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"

#include "tensorflow/compiler/xla/util.h"

#if GOOGLE_CUDA
#include "absl/synchronization/blocking_counter.h"
#include "third_party/nccl/nccl.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif

namespace xla {
namespace gpu {

/* static */ bool NcclAllReduceThunk::NcclIsEnabled() {
#if GOOGLE_CUDA
  return true;
#else
  return false;
#endif
}

#if GOOGLE_CUDA
namespace {

// GPU-replica-driving host threads (i.e. the threads that call
// GpuExecutable::Execute) build up this structure to describe their
// participating replica, and then call to
// GlobalRendezvousManager::SubmitParticipant.
struct ParticipantData {
  // Number of replicas particiating in the AllReduce.
  int64 replica_count;

  int64 element_count;
  int64 device_ordinal;
  int64 generation_counter;

  // TODO(b/125951860): We should vet that we're buffer allocating such that
  // source_buffer == destination_buffer if that avoids a NCCL copy (will depend
  // on how well the NCCL in-place implementation performs vs the out-of-place
  // implementation).
  se::DeviceMemoryBase source_data;
  se::DeviceMemoryBase destination_data;
  se::Stream* stream;

  NcclAllReduceThunk* originator;

  string ToString() const {
    return absl::StrFormat(
        "ParticipantData{replica_count=%d, element_count=%d, "
        "device_ordinal=%d, generation_counter=%d, stream=%p, originator=%p}",
        replica_count, element_count, device_ordinal, generation_counter,
        stream, originator);
  }
};

// Class that gets instantiated as a singleton in GetGlobalRendezvous() to
// coordinate participating threads in performing an AllReduce operation.
//
// This manager is responsible for establishing communication channels and
// ultimately enqueueing the NCCL library operation onto the participating
// streams.
class GlobalRendezvousManager {
 public:
  // The GpuExecutable-executing threads call this in order to a) establish the
  // all-reduce rendezvous and b) enqueue the AllReduce operation on the caller
  // thread's associated stream (given in "participant").
  //
  // Implementation note: since the rendezvous we're creating here is global, we
  // try to be paranoid about the fact that the *correct* one is happening.  In
  // an ideal world we'd have some StreamExecutor se::Platform level construct
  // that we could use for cross-device networking primitives (e.g. via a
  // NetworkSupport interface) that could be shared between TensorFlow and XLA,
  // but this is a reasonable stopgap measure to get multi-GPU-replica up and
  // running properly for single-host, single-concurrent-XLA-module usage.
  Status SubmitParticipant(ParticipantData participant);

  // Returns the current generation number of AllReduce operations.
  // (Currently one AllReduce operation occurs per generation.)
  int64 GetCurrentGeneration() {
    tensorflow::mutex_lock lock(mutex_);
    return current_generation_;
  }

 private:
  // Called by the primary thread to set up the communication links.
  //
  // TODO(b/125951860): This performs lots of (presumably) unnecessary host-side
  // synchronization so that we can be paranoid about semantics in the earliest
  // implementation. In the limit we should only need to synchronize host
  // replica threads when the "number of replicas" or "participating device
  // ordinals" change, to set up a new NCCL "communication" context, at which
  // point we can enqueue onto device streams without host synchronization in
  // our code -- this will likely be helpful for "lots of little AllReduce"
  // cases.
  Status InitializeCommunicationChannels() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Called when all necessary participants are present, the functionality
  // that's implemented by all executing threads lives in here.
  Status DoAllReduce(ParticipantData data, ncclComm_t comm);

  // Puts all state back into a "reset" state for the next generation of
  // AllReduce requests.
  void DeinitializeGeneration() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    for (ncclComm_t& comm : comms_) {
      ncclCommDestroy(comm);
    }
    comms_.clear();
    participants_.clear();
    current_generation_++;
    initialized_ = false;
    done_ = absl::nullopt;
  }

  tensorflow::mutex mutex_;
  tensorflow::condition_variable all_participants_present_;
  tensorflow::condition_variable deinitialized_;

  // Communication handles that correspond to the participants below.
  std::vector<ncclComm_t> comms_ GUARDED_BY(mutex_);

  Status initialize_status_ GUARDED_BY(mutex_);
  std::vector<ParticipantData> participants_ GUARDED_BY(mutex_);
  int64 current_generation_ GUARDED_BY(mutex_) = 0;
  bool initialized_ GUARDED_BY(mutex_) = false;

  // The participating threads wait for this to count down in order to know we
  // can begin the teardown process.
  absl::optional<tensorflow::BlockingCounter> done_;
};

Status GlobalRendezvousManager::SubmitParticipant(ParticipantData participant) {
  auto all_participants_present = [this, &participant]()
                                      EXCLUSIVE_LOCKS_REQUIRED(mutex_) -> bool {
    return participants_.size() >= participant.replica_count;
  };

  // We remember the participant index at which we are inserted and use that
  // same index for referring to auxiliary metadata (e.g. the ncclComm_t handle
  // index) below.
  int64 index;

  {
    tensorflow::mutex_lock lock(mutex_);

    // Spot check for consistent replica counts among submitting threads.
    if (!participants_.empty() &&
        (participants_.back().replica_count != participant.replica_count ||
         participants_.back().originator != participant.originator)) {
      return InvalidArgument(
          "Running two XLA modules with AllReduces in parallel is not "
          "supported. It is possible this is due to a bug where were try to "
          "run two different AllReduces from the same module at once. "
          "(Attempted a rendezvous with a different replica count from other "
          "participants; existing: %s; submitted: %s)",
          participants_.back().ToString(), participant.ToString());
    }
    index = participants_.size();
    participants_.push_back(participant);

    if (all_participants_present()) {
      all_participants_present_.notify_all();
    }
  }

  // We pull into our thread a) the communication handle and b) whether we're
  // the "primary" thread for this rendezvous -- the "primary" thread has some
  // additional responsibilities for setup/teardown.
  ncclComm_t comm;
  bool primary;

  {
    tensorflow::mutex_lock lock(mutex_);
    while (!all_participants_present()) {
      // Once all the participants have arrived, all participating threads will
      // cross this barrier, though only (the first) one will be the "primary".
      all_participants_present_.wait(lock);
    }

    // Somebody will be the first -- that thread has some additional
    // responsibilities.
    primary = !initialized_;

    CHECK_EQ(participant.generation_counter, current_generation_);

    // Bump the generation counter so the other threads know we've completed the
    // global rendezvous and have set up the AllReduce.
    if (primary) {
      VLOG(3) << "Primary initializing accounting data.";
      initialized_ = true;
      done_.emplace(participant.replica_count);
      initialize_status_ = InitializeCommunicationChannels();
      VLOG(3) << "Done initializing communication channels; status: "
              << initialize_status_;
      if (!initialize_status_.ok()) {
        DeinitializeGeneration();
      }
    }

    if (!initialize_status_.ok()) {
      // TODO(b/125951860): If this fails once, it will fail forever.
      return initialize_status_;
    }

    comm = comms_[index];

    // Drop the lock at the end of scope so other participants may enter.
  }

  VLOG(3) << "Performing all reduce from device ordinal: "
          << participant.device_ordinal;

  Status all_reduce_status = DoAllReduce(participant, comm);

  VLOG(3) << "Waiting for all participants to complete enqueue.";

  done_->DecrementCount();

  if (primary) {
    // Primary thread clears out the AllReduce state when everybody is done to
    // make it clean-slate for any subsequent AllReduce request (e.g. number of
    // replicas may change in the next request).
    //
    // Note surrounding TODOs for only reinitializing this when the replica
    // count / participants actually change -- lots of "playing it safe"
    // happening in this first cut.
    done_->Wait();
    VLOG(3) << "All participants completed enqueue.";
    VLOG(3) << "Primary thread clearing.";
    tensorflow::mutex_lock lock(mutex_);
    DeinitializeGeneration();
    VLOG(3) << "Generation is now: " << current_generation_;
    deinitialized_.notify_all();
  } else {
    VLOG(3) << "Waiting to deinitialize.";
    tensorflow::mutex_lock lock(mutex_);
    while (initialized_) {
      deinitialized_.wait(lock);
    }
  }

  VLOG(3) << "Returning status: " << all_reduce_status;
  return all_reduce_status;
}

Status GlobalRendezvousManager::InitializeCommunicationChannels() {
  std::vector<int> ordinals;
  for (ParticipantData& data : participants_) {
    ordinals.push_back(data.device_ordinal);
  }
  comms_.resize(ordinals.size());
  VLOG(3) << "Participants: " << participants_.size()
          << "; initializing comms.";
  ncclResult_t result = ncclCommInitAll(comms_.data(), comms_.size(),
                                        /*devlist=*/ordinals.data());
  if (result != ncclSuccess) {
    comms_.clear();
    return InternalError(
        "Failed to initialize NCCL communication channels for %d participants: "
        "%s",
        participants_.size(), ncclGetErrorString(result));
  }
  return Status::OK();
}

Status GlobalRendezvousManager::DoAllReduce(ParticipantData participant,
                                            ncclComm_t comm) {
  se::StreamExecutor* executor = participant.stream->parent();
  se::cuda::ScopedActivateExecutorContext scoped_context(executor);
  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      participant.stream->implementation()->GpuStreamMemberHack());
  VLOG(3) << "Using stream pointer: " << cu_stream
          << " on device: " << participant.device_ordinal;
  void* send_buffer = participant.source_data.opaque();
  void* recv_buffer = participant.destination_data.opaque();
  ncclResult_t result = ncclAllReduce(send_buffer, recv_buffer,
                                      /*count=*/participant.element_count,
                                      /*datatype=*/ncclFloat,
                                      /*op=*/ncclSum,
                                      /*comm=*/comm,
                                      /*stream=*/*cu_stream);
  TF_RET_CHECK(ncclSuccess == result)
      << "Failed to perform all-reduce: " << ncclGetErrorString(result);

  VLOG(3) << "Done performing all reduce for ordinal: "
          << participant.device_ordinal;

  return Status::OK();
}

static GlobalRendezvousManager* GetGlobalRendezvous() {
  static auto* manager = new GlobalRendezvousManager;
  return manager;
}

}  // namespace

Status NcclAllReduceThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  auto* global_rendezvous = GetGlobalRendezvous();

  ParticipantData participant;
  participant.replica_count = replica_count_;
  participant.element_count = element_count_;
  participant.device_ordinal = stream->parent()->device_ordinal();
  participant.generation_counter = global_rendezvous->GetCurrentGeneration();
  participant.source_data = buffer_allocations.GetDeviceAddress(source_buffer_);
  participant.destination_data =
      buffer_allocations.GetDeviceAddress(destination_buffer_);
  participant.stream = stream;
  participant.originator = this;

  return GetGlobalRendezvous()->SubmitParticipant(std::move(participant));
}
#else

Status NcclAllReduceThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
}

#endif  // GOOGLE_CUDA

NcclAllReduceThunk::NcclAllReduceThunk(
    int64 replica_count, int64 element_count,
    const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer,
    const HloInstruction* all_reduce)
    : Thunk(Thunk::kNcclAllReduce, all_reduce),
      replica_count_(replica_count),
      element_count_(element_count),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer) {}

}  // namespace gpu
}  // namespace xla
