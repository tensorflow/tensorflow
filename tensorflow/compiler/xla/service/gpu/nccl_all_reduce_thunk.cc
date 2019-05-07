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
#include "absl/container/flat_hash_set.h"
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
//
// Implementation note: We make an effort to avoid initializing nccl
// communciation channels too often, as this is expensive.
//
// Ideally, we'd set up a nccl channel between each pair of devices that needs
// to communicate, and close each channel when the GPUs won't be communicating
// again "for a long time" (because channels hold memory on the GPU).  As a
// simplification to this ideal, we adopt the following policy.
//
//  - We maintain a set of GPUs that are "actively participating" in
//    cross-device communications.  That set of GPUs is always connected as a
//    clique, using ncclCommInitAll.
//
//  - When a NcclAllReduceThunk touches a new GPU, we tear down the old clique
//    and build a new, bigger one.
//
//  - All GPUs ever touched by a thunk are considered "actively in use" by that
//    thunk until the thunk is destroyed.  Destroying the thunk decrements the
//    refcount of the GPUs it's touched, and if that refcount goes to 0
//    (meaning, some GPUs are no longer in use by any thunk), we tear down the
//    clique and build a new, smaller one.
//
// This approximation is justified because:
//
//  - Currently the only collective operation we support is AllReduce, which
//    requires a clique.  When we support point-to-point operations, we may not
//    want to build a communication clique.
//
//  - Tearing down and creating a new thunk is tantamount to running the whole
//    XLA:GPU compiler.  This is expensive, so shouldn't happen "too often" to
//    cause thrashing here.
//
//  - XLA executables already keep resources on the GPU tied to the lifetime of
//    the executable (e.g. constants stored in GPU memory), so tying the
//    lifetime of the nccl communication channels to the lifetime of the
//    executable is consistent.
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

  // Increments the refcount of a GPU in our accounting of which devices are
  // "actively participating" in cross-device operations.
  //
  // This doesn't actually do anything other than increment the refcount.  If
  // the GPU added here is novel, we'll rebuild the nccl communication clique
  // when we actually go do the communication.
  void AddrefParticipatingDevice(int device_ordinal);

  // Decrements the refcount of a set of GPUs in our accounting of which devices
  // are "actively participating" in cross-device operations.
  //
  // If one or more GPUs' refcounts to go 0, we immediately destroy the whole
  // nccl communication clique.  We'll rebuild a new, smaller clique the next
  // time it's used.
  void DecrefParticipatingDevices(absl::Span<const int> device_ordinals);

  // Gets the set of devices that have a NCCL channel currently open.  This is
  // primarily for testing.
  absl::flat_hash_set<int> DevicesWithOpenNcclChannels() const {
    absl::flat_hash_set<int> devices;
    tensorflow::mutex_lock lock(mutex_);
    for (const auto& kv : comms_) {
      devices.insert(kv.first);
    }
    return devices;
  }

 private:
  // Destroys the current nccl communication clique and builds a new one
  // connecting the given devices.
  Status ReinitializeNcclClique(const absl::flat_hash_set<int>& device_ordinals)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Called when all necessary participants are present, the functionality
  // that's implemented by all executing threads lives in here.
  Status DoAllReduce(ParticipantData data, ncclComm_t comm);

  // Puts all state back into a "reset" state for the next generation of
  // AllReduce requests.
  void DeinitializeGeneration() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    participants_.clear();
    current_generation_++;
    initialized_ = false;
    done_ = absl::nullopt;
  }

  mutable tensorflow::mutex mutex_;
  tensorflow::condition_variable all_participants_present_;
  tensorflow::condition_variable deinitialized_;

  Status initialize_status_ GUARDED_BY(mutex_);
  std::vector<ParticipantData> participants_ GUARDED_BY(mutex_);
  int64 current_generation_ GUARDED_BY(mutex_) = 0;
  bool initialized_ GUARDED_BY(mutex_) = false;

  struct Comm {
    explicit Comm(ncclComm_t nccl_comm) : nccl_comm(nccl_comm) {}

    // Movable, but not copyable.
    Comm(Comm&& c) : nccl_comm(c.nccl_comm) { c.nccl_comm.reset(); }
    Comm& operator=(Comm&& c) {
      nccl_comm = c.nccl_comm;
      c.nccl_comm.reset();
      return *this;
    }
    Comm(const Comm&) = delete;
    Comm& operator=(const Comm&) = delete;

    absl::optional<ncclComm_t> nccl_comm;

    ~Comm() {
      if (nccl_comm.has_value()) {
        VLOG(3) << absl::StreamFormat("Destroying comm %p", *nccl_comm);
        ncclCommDestroy(*nccl_comm);
      }
    }
  };
  // Communication handles for our NCCL clique.  Key is device ordinal.
  absl::flat_hash_map<int, Comm> comms_ GUARDED_BY(mutex_);

  // Refcounts of which devices are "actively participating" in all-reduces.
  // These devices don't necessarily have an open comm, but the next time we run
  // an operation, we'll create a NCCL clique between all of them.
  absl::flat_hash_map<int, int64> device_refcounts_ GUARDED_BY(mutex_);

  // The participating threads wait for this to count down in order to know we
  // can begin the teardown process.
  absl::optional<tensorflow::BlockingCounter> done_;
};

Status GlobalRendezvousManager::SubmitParticipant(ParticipantData participant) {
  auto all_participants_present = [this, &participant]()
                                      EXCLUSIVE_LOCKS_REQUIRED(mutex_) -> bool {
    return participants_.size() >= participant.replica_count;
  };

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

      // Check if all participants_ are in comms_.  If not, we will rebuild the
      // clique to include them.  (This can't be spelled using absl::c_any_of
      // because it needs to touch comms_ and tensorflow::mutex lacks an
      // AssertHeld() function that would let us assert that the lambda is run
      // while holding the lock.)
      bool new_devices_found = false;
      for (const auto& p : participants_) {
        if (!comms_.contains(p.device_ordinal)) {
          new_devices_found = true;
          break;
        }
      }

      if (new_devices_found) {
        absl::flat_hash_set<int> new_clique_device_ordinals;
        for (const auto& kv : comms_) {
          new_clique_device_ordinals.insert(kv.first);
        }
        for (const auto& p : participants_) {
          new_clique_device_ordinals.insert(p.device_ordinal);
        }

        initialize_status_ = ReinitializeNcclClique(new_clique_device_ordinals);
        VLOG(3) << "Done initializing communication channels; status: "
                << initialize_status_;
        if (!initialize_status_.ok()) {
          DeinitializeGeneration();
        }
      }
    }

    if (!initialize_status_.ok()) {
      // TODO(b/125951860): If this fails once, it will fail forever.
      return initialize_status_;
    }

    comm = *comms_.at(participant.device_ordinal).nccl_comm;

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

Status GlobalRendezvousManager::ReinitializeNcclClique(
    const absl::flat_hash_set<int>& device_ordinals) {
  comms_.clear();

  std::vector<int> ordinals_vec(device_ordinals.begin(), device_ordinals.end());
  std::vector<ncclComm_t> comm_vec;
  comm_vec.resize(device_ordinals.size());

  VLOG(3) << absl::StreamFormat(
      "Initializing nccl comms for participant devices {%s}",
      absl::StrJoin(ordinals_vec, ", "));
  ncclResult_t result = ncclCommInitAll(comm_vec.data(), comm_vec.size(),
                                        /*devlist=*/ordinals_vec.data());
  if (result != ncclSuccess) {
    return InternalError(
        "Failed to initialize NCCL communication channels for %d participants: "
        "%s",
        ordinals_vec.size(), ncclGetErrorString(result));
  }

  for (int64 i = 0; i < ordinals_vec.size(); ++i) {
    VLOG(3) << absl::StreamFormat("Device ordinal %d assigned ncclComm %p",
                                  ordinals_vec[i], comm_vec[i]);
    CHECK(comms_.emplace(ordinals_vec[i], Comm{comm_vec[i]}).second);
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
  VLOG(3) << absl::StreamFormat(
      "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
      "datatype=ncclFloat, op=ncclSum, comm=%p, stream=%p)",
      send_buffer, recv_buffer, participant.element_count,
      static_cast<const void*>(comm), cu_stream);
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

void GlobalRendezvousManager::AddrefParticipatingDevice(int device_ordinal) {
  // Addref'ing a device doesn't do anything other than increment its refcount.
  // We'll update our nccl clique if necessary during the next call to
  // SubmitParticipant.
  tensorflow::mutex_lock lock(mutex_);
  device_refcounts_[device_ordinal]++;
}

void GlobalRendezvousManager::DecrefParticipatingDevices(
    absl::Span<const int> device_ordinals) {
  // Decref'ing devices causes us to destroy the nccl clique if any devices were
  // removed due to having refcount 0.  We'll rebuild the new, smaller clique
  // during the next call to SubmitParticipant.
  tensorflow::mutex_lock lock(mutex_);
  bool removed_device = false;
  for (int device_ordinal : device_ordinals) {
    auto it = device_refcounts_.find(device_ordinal);
    CHECK(it != device_refcounts_.end());
    it->second--;
    if (it->second == 0) {
      device_refcounts_.erase(it);
      removed_device = true;
    }
  }

  if (removed_device) {
    comms_.clear();
  }
}

static GlobalRendezvousManager* GetGlobalRendezvous() {
  static auto* manager = new GlobalRendezvousManager;
  return manager;
}

}  // namespace

/*static*/ absl::flat_hash_set<int>
NcclAllReduceThunk::DevicesWithOpenNcclChannels() {
  return GetGlobalRendezvous()->DevicesWithOpenNcclChannels();
}

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

  // We currently say that that all GPUs this thunk has ever touched are
  // "actively participating" in cross-device operations, until the thunk itself
  // is destroyed.
  //
  // This policy is an attempt to avoid thrashing the GPU (ncclCommInitAll is
  // very expensive) while also freeing resources on the GPUs when we can.  The
  // idea is, creating new thunks is tantamount to running the whole XLA:GPU
  // compiler stack, so that shouldn't happen terribly often.
  bool new_device;
  {
    tensorflow::mutex_lock lock(mu_);
    new_device = devices_seen_.insert(participant.device_ordinal).second;
  }
  if (new_device) {
    GetGlobalRendezvous()->AddrefParticipatingDevice(
        participant.device_ordinal);
  }

  return GetGlobalRendezvous()->SubmitParticipant(std::move(participant));
}

NcclAllReduceThunk::~NcclAllReduceThunk() {
  GetGlobalRendezvous()->DecrefParticipatingDevices(
      std::vector<int>(devices_seen_.begin(), devices_seen_.end()));
}

#else

Status NcclAllReduceThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
}

NcclAllReduceThunk::~NcclAllReduceThunk() = default;

/*static*/ absl::flat_hash_set<int>
NcclAllReduceThunk::DevicesWithOpenNcclChannels() {
  return {};
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
