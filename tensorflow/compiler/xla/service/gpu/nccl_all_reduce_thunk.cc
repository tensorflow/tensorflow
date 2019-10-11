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

#include <chrono>  // NOLINT (required by TF interfaces)
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "third_party/nccl/nccl.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace xla {
namespace gpu {

// This file runs collective ops (i.e. ops that communicate between multiple
// GPUs) using NCCL.  Currently only kAllReduce is implemented.
//
// Here's a high-level overview of how running an op works.
//
//  - Multiple threads call NcclAllReduceThunk::ExecuteOnStream.
//  - All threads that "go together" (i.e. are participating in the "same"
//    collective op) choose the same Rendezvous object from a global map.
//  - Once all threads have arrived at the Rendezvous, we know exactly which
//    GPUs are participating in the op, so we get or create a NcclClique
//    containing those GPUs.
//  - We perform the NCCL operation using the clique, then destroy the
//    Rendezvous.  The clique is cached, see below.
//
// Creating NCCL cliques is expensive, so we cache them.  Our policy is, a thunk
// keeps alive all cliques it's ever used.  When the thunk is destroyed, it
// releases its handle on the cliques, and cliques whose refcounts go to 0 are
// destroyed.

/* static */ bool NcclAllReduceThunk::NcclIsEnabled() {
  return true;  // Skylark selects this source file if NCCL is enabled.
}

namespace {

using tensorflow::BlockingCounter;

// Functions to translate an ncclResult_t/cudaError_t to a Status object.  Used
// by the macros below.
Status TranslateStatus(ncclResult_t s, const char* file, int64 line,
                       const char* expr) {
  if (s == ncclSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: NCCL operation %s failed: %s", file, line, expr,
                      ncclGetErrorString(s)));
}

Status TranslateStatus(cudaError_t s, const char* file, int64 line,
                       const char* expr) {
  if (s == cudaSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: CUDA operation %s failed: %s", file, line, expr,
                      cudaGetErrorString(s)));
}

// Macros to return or warn on CUDA/NCCL errors.  (The same macro works for both
// NCCL and CUDA errors.)
//
// It's tempting to say these macros belong in an XLA header somewhere, but in
// practice we don't do much direct-to-CUDA-API stuff outside of this file.
#define XLA_CUDA_RETURN_IF_ERROR(expr)                                       \
  do {                                                                       \
    Status s = ::xla::gpu::TranslateStatus(expr, __FILE__, __LINE__, #expr); \
    if (!s.ok()) {                                                           \
      return s;                                                              \
    }                                                                        \
  } while (0)

#define XLA_CUDA_WARN_IF_ERROR(expr)                                         \
  do {                                                                       \
    Status s = ::xla::gpu::TranslateStatus(expr, __FILE__, __LINE__, #expr); \
    if (!s.ok()) {                                                           \
      LOG(ERROR) << s.ToString();                                            \
    }                                                                        \
  } while (0)

template <typename DescFn>
void WaitAndLogIfStuck(BlockingCounter* counter, const DescFn& desc_fn) {
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

// RAII class owning a ncclComm_t, ensuring it doesn't leak.
class NcclComm {
 public:
  explicit NcclComm(ncclComm_t comm) : comm_(comm) {}

  // Movable, but not copyable.
  NcclComm(NcclComm&& c) noexcept : comm_(c.comm_) { c.comm_.reset(); }
  NcclComm& operator=(NcclComm&& c) noexcept {
    comm_ = c.comm_;
    c.comm_.reset();
    return *this;
  }
  NcclComm(const NcclComm&) = delete;
  NcclComm& operator=(const NcclComm&) = delete;

  ~NcclComm() {
    if (comm_.has_value() && *comm_ != nullptr) {
      VLOG(3) << absl::StreamFormat("Destroying comm %p", *comm_);
      XLA_CUDA_WARN_IF_ERROR(ncclCommDestroy(*comm_));
    }
  }

  ncclComm_t comm() { return *comm_; }

 private:
  absl::optional<ncclComm_t> comm_;
};

absl::optional<ncclRedOp_t> MatchAllReduceComputation(
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
    return ncclSum;
  } else if (match_opcode(HloOpcode::kMultiply)) {
    return ncclProd;
  } else if (match_opcode(HloOpcode::kMinimum)) {
    return ncclMin;
  } else if (match_opcode(HloOpcode::kMaximum)) {
    return ncclMax;
  } else {
    return absl::nullopt;
  }
}

absl::optional<ncclDataType_t> MatchNcclDataType(const HloInstruction* crs) {
  switch (crs->operand(0)->shape().element_type()) {
    case S8:
      return ncclInt8;
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
      return ncclFloat32;
    case F64:
      return ncclFloat64;
    default:
      return absl::nullopt;
  }
}

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
  enum AllReduceKind {
    kCrossModule,
    kCrossReplica,
  };

  explicit RendezvousKey(const RunId& run_id,
                         std::vector<int64> participating_replicas,
                         const HloAllReduceInstruction* instr)
      : run_id(run_id), participating_replicas(participating_replicas) {
    std::tie(all_reduce_kind, op_id) =
        instr->channel_id().has_value()
            ? std::make_pair(kCrossModule, instr->channel_id().value())
            : std::make_pair(
                  kCrossReplica,
                  static_cast<int64>(instr->GetModule()->unique_id()));
    absl::optional<ncclRedOp_t> computation =
        MatchAllReduceComputation(instr->to_apply());
    CHECK(computation.has_value());
    computation_kind = *computation;
    absl::optional<ncclDataType_t> allreduce_datatype =
        MatchNcclDataType(instr);
    CHECK(allreduce_datatype.has_value());
    datatype = *allreduce_datatype;
  }

  int num_participants() const { return participating_replicas.size(); }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.participating_replicas,
                      static_cast<int>(k.all_reduce_kind), k.op_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id &&
           a.participating_replicas == b.participating_replicas &&
           a.all_reduce_kind == b.all_reduce_kind &&  //
           a.op_id == b.op_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }

  string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, participating_replicas=[%s], "
        "all_reduce_kind=%d, op_id=%d}",
        run_id.ToString(), absl::StrJoin(participating_replicas, ","),
        static_cast<int>(all_reduce_kind), op_id);
  }

  RunId run_id;
  std::vector<int64> participating_replicas;
  AllReduceKind all_reduce_kind;
  ncclRedOp_t computation_kind;
  ncclDataType_t datatype;
  int64 op_id;
};

// Encapsulates parameters to Rendezvous::SubmitParticipant.
struct ParticipantData {
  explicit ParticipantData(RendezvousKey rendezvous_key)
      : rendezvous_key(rendezvous_key) {}

  int64 element_count;
  int64 device_ordinal;
  RendezvousKey rendezvous_key;

  // TODO(b/125951860): We should vet that we're buffer allocating such that
  // source_buffer == destination_buffer if that avoids a NCCL copy (will depend
  // on how well the NCCL in-place implementation performs vs the out-of-place
  // implementation).
  se::DeviceMemoryBase source_data;
  se::DeviceMemoryBase destination_data;
  se::Stream* stream;

  int num_participants() const { return rendezvous_key.num_participants(); }

  string ToString() const {
    return absl::StrFormat(
        "ParticipantData{element_count=%d, rendezvous_key=%s, "
        "device_ordinal=%d, stream=%p}",
        element_count, rendezvous_key.ToString(), device_ordinal, stream);
  }
};

// Key for looking up a particular NCCL clique.  This is just a set of unique
// device ordinals (i.e. GPU IDs).
struct NcclCliqueKey {
  explicit NcclCliqueKey(absl::Span<const int64> devices)
      : devices(devices.begin(), devices.end()) {
    absl::c_sort(this->devices);
    CHECK(absl::c_adjacent_find(devices) == devices.end())
        << "Duplicate devices are not allowed: "
        << absl::StrJoin(devices, ", ");
  }

  template <typename H>
  friend H AbslHashValue(H h, const NcclCliqueKey& k) {
    return H::combine(std::move(h), k.devices);
  }
  friend bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b) {
    return a.devices == b.devices;
  }

  std::vector<int64> devices;
};

// Owns a clique of NCCL comms which can be used for collective operations among
// a particular set of GPUs.
//
// You must ensure this is not in an error state (i.e. status() is OK) before
// touching any other methods.
//
// (Usually allowing objects to be in a constructed-but-uninitialized state is
// an antipattern.  We do it here because it allows us to have a
// RefcountingHashMap which contains and automatically constructs NcclCliques.
// This greatly simplifies the rest of this file.)
//
// Note that if you want to do a collective operation among a subset of these
// GPUs, you'll need a different clique.
class NcclClique {
 public:
  explicit NcclClique(absl::Span<const int64> devices)
      : devices_(devices.begin(), devices.end()) {
    absl::c_sort(devices_);
    status_ = Init();
  }

  Status status() { return status_; }

  absl::Span<const int64> devices() {
    TF_CHECK_OK(status_);
    return devices_;
  }
  ncclComm_t comm(int64 device) {
    int64 idx = std::distance(devices_.begin(), absl::c_find(devices_, device));
    return comms_.at(idx).comm();
  }

  // These methods let you acquire exclusive access to a NCCL clique, ensuring
  // no other NCCL operations are taking place on the clique's comms.
  //
  // We disable thread-safety analysis because in common use, only the primary
  // thread in a Rendezvous acquires this lock, and that makes thread-safety
  // analysis unhappy.  Tread carefully, you are playing with fire.
  void Lock() NO_THREAD_SAFETY_ANALYSIS {
    TF_CHECK_OK(status_);
    mu_->lock();
  }
  void Unlock() NO_THREAD_SAFETY_ANALYSIS {
    TF_CHECK_OK(status_);
    mu_->unlock();
  }

 private:
  Status Init() {
    VLOG(3) << absl::StreamFormat(
        "Initializing nccl comms for participant devices {%s}",
        absl::StrJoin(devices_, ", "));

    // Restore CUDA device after running this.  XLA shouldn't care, but maybe
    // another consumer does.
    int initial_cuda_device;
    XLA_CUDA_RETURN_IF_ERROR(cudaGetDevice(&initial_cuda_device));
    auto cuda_device_restorer = MakeCleanup(
        [&] { XLA_CUDA_WARN_IF_ERROR(cudaSetDevice(initial_cuda_device)); });

    // When using ncclGroupStart/End it seems that the ncclComm_t's are not
    // populated until the End() call.  This unfortunately makes error handling
    // tricky.
    std::vector<ncclComm_t> raw_comms(devices_.size(), nullptr);
    ncclUniqueId nccl_id;
    XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
    Status status = [&] {
      for (int i = 0; i < devices_.size(); ++i) {
        XLA_CUDA_RETURN_IF_ERROR(cudaSetDevice(devices_[i]));
        XLA_CUDA_RETURN_IF_ERROR(
            ncclCommInitRank(&raw_comms[i], devices_.size(), nccl_id, i));
      }
      return Status::OK();
    }();
    // Always call ncclGroupEnd().
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

    // Populate comms_ from the raw comms we created above.  If we encountered
    // an error above we'll later clear comms_ thus destroying any raw comms
    // that were created before the error.
    for (int i = 0; i < devices_.size(); ++i) {
      VLOG(3) << absl::StreamFormat("Device %d assigned ncclComm %p",
                                    devices_[i], raw_comms[i]);
      CHECK(raw_comms[i] != nullptr || !status.ok());
      comms_.emplace_back(raw_comms[i]);
    }
    if (!status.ok()) {
      comms_.clear();
    }

    return status;
  }

  Status status_;
  std::vector<int64> devices_;
  std::vector<NcclComm> comms_;

  // This mutex is in a unique_ptr so NcclClique can be movable.
  std::unique_ptr<tensorflow::mutex> mu_ =
      absl::make_unique<tensorflow::mutex>();
};

// Global cache of NCCL cliques.  An entry in this map is kept alive as long as
// there's a reference to it somewhere.  A Thunk holds a reference to each
// Clique it's ever used.
//
// A consequence of the fact that this is process-global is that we'll only ever
// have one clique alive for a given set of GPUs.  This means that a process
// will never do two collective operations concurrently on the same set of GPUs.
RefcountingHashMap<NcclCliqueKey, NcclClique>& GlobalNcclCliqueMap() {
  static auto& m = *new RefcountingHashMap<NcclCliqueKey, NcclClique>(
      [](const NcclCliqueKey& key) {
        return absl::make_unique<NcclClique>(key.devices);
      });
  return m;
}

// The set of threads that want to do a collective op together all pick the same
// Rendezvous object out of the global cache and call SubmitParticipant.
//
// The Rendezvous instance handles waiting for all threads to join, ensuring
// that a clique exists for the desired set of GPUs, etc.
//
// Rendezvous objects can only be used once.
class Rendezvous {
 public:
  explicit Rendezvous(const RendezvousKey& k) : key_(k) {}

  // Runs the all-reduce on the given thread.  If successful, returns
  //  - a handle to the clique that was used, so that the caller may keep the
  //    clique alive if it chooses.
  //  - a BlockingCounter initialized to the number of participants, so that
  //    the caller can coordinate with the participants one last time if it
  //    chooses.  This is useful for coordinating destruction of the Rendezvous.
  StatusOr<
      std::pair<std::shared_ptr<NcclClique>, std::shared_ptr<BlockingCounter>>>
  SubmitParticipant(ParticipantData participant);

 private:
  Status DoAllReduce(ParticipantData participant, ncclComm_t comm);

  const RendezvousKey key_;

  BlockingCounter all_participants_present_{key_.num_participants()};
  BlockingCounter done_{key_.num_participants()};
  // BlockingCounter returned by SubmitParticipant.
  std::shared_ptr<BlockingCounter> returned_blocking_counter_{
      std::make_shared<BlockingCounter>(key_.num_participants())};

  tensorflow::mutex mu_;

  bool initialized_ GUARDED_BY(mu_) = false;

  std::vector<ParticipantData> participants_ GUARDED_BY(mu_);
};

// Global map of Rendezvous objects.  A thread participating in a collective op
// looks up its Rendezvous in this map to find the other threads that it's
// participating with.
//
// Rendezvous objects are one-time use, so they're removed from this map once
// we're through with them.
RefcountingHashMap<RendezvousKey, Rendezvous>& GlobalRendezvousMap() {
  static auto& m = *new RefcountingHashMap<RendezvousKey, Rendezvous>(
      [](const RendezvousKey& k) { return absl::make_unique<Rendezvous>(k); });
  return m;
}

StatusOr<
    std::pair<std::shared_ptr<NcclClique>, std::shared_ptr<BlockingCounter>>>
Rendezvous::SubmitParticipant(ParticipantData participant) {
  {
    tensorflow::mutex_lock lock(mu_);
    CHECK(!initialized_);

    // Spot check for consistent replica counts among submitting threads.
    if (!participants_.empty() &&
        (participants_.back().element_count != participant.element_count ||
         participants_.back().rendezvous_key != participant.rendezvous_key)) {
      return InvalidArgument(
          "Mismatch among all-reduce participants.  Expected same "
          "replica-count, element-count, and rendezvous-key but were %s and %s",
          participants_.back().ToString(), participant.ToString());
    }
    participants_.push_back(participant);
  }

  // Wait for all participants to arrive.
  all_participants_present_.DecrementCount();
  WaitAndLogIfStuck(&all_participants_present_, [&] {
    return absl::StrFormat(
        "participant for device ordinal %d, stream %p waiting for all "
        "participants to be arrive at rendezvous %s",
        participant.device_ordinal, participant.stream, key_.ToString());
  });

  // We pull into our thread a) the communication handle and b) whether we're
  // the "primary" thread for this rendezvous -- the "primary" thread has some
  // additional responsibilities for setup/teardown.
  ncclComm_t comm;
  bool primary;
  std::shared_ptr<NcclClique> clique;

  // Releases the lock on the clique (held only by the primary thread).
  Cleanup<std::function<void()>> clique_lock_releaser;

  {
    tensorflow::mutex_lock lock(mu_);

    // The first thread to get here has additional responsibilities, such as
    // ensuring that there's a NCCL clique available for us to use.
    primary = !initialized_;

    // Look up or create the NCCL clique for this set of devices.
    std::vector<int64> devices;
    for (const auto& p : participants_) {
      devices.push_back(p.device_ordinal);
    }
    clique = GlobalNcclCliqueMap()[NcclCliqueKey(devices)];

    if (primary) {
      VLOG(3) << "Primary initializing accounting data.";
      initialized_ = true;

      // Acquire exclusive access to the NCCL clique itself so that two
      // unrelated collective operations won't try to use the clique
      // concurrently.
      clique->Lock();
      clique_lock_releaser = MakeCleanup([clique] { clique->Unlock(); });
    }

    if (!clique->status().ok()) {
      VLOG(1)
          << "SubmitParticipant failing because clique failed to initialize: "
          << clique->status().ToString();
      return clique->status();
    }

    comm = clique->comm(participant.device_ordinal);

    // Drop the lock at the end of scope so other participants may enter.
  }

  VLOG(3) << "Performing all reduce from device ordinal: "
          << participant.device_ordinal;
  Status all_reduce_status = DoAllReduce(participant, comm);
  VLOG(3) << "This thread done with all-reduce op.";

  done_.DecrementCount();

  // The primary owns the lock on the NCCL clique.  Hold it until all threads
  // are done.  (We'll release it when we return from this function.)
  if (primary) {
    WaitAndLogIfStuck(&done_, [&] {
      return absl::StrFormat(
          "primary participant (device ordinal %d, stream %p) waiting for all "
          "other participants to complete all-reduce %s",
          participant.device_ordinal, participant.stream, key_.ToString());
    });
  }

  VLOG(3) << "Returning status: " << all_reduce_status;
  if (!all_reduce_status.ok()) {
    return all_reduce_status;
  }
  return std::make_pair(clique, returned_blocking_counter_);
}

Status Rendezvous::DoAllReduce(ParticipantData participant, ncclComm_t comm) {
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
  XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                         /*count=*/participant.element_count,
                                         /*datatype=*/key_.datatype,
                                         /*op=*/key_.computation_kind,
                                         /*comm=*/comm,
                                         /*stream=*/*cu_stream));

  VLOG(3) << "Done performing all reduce for ordinal: "
          << participant.device_ordinal;

  return Status::OK();
}

}  // namespace

// Extra data stored in NcclAllReduceThunk that we didn't want to expose in the
// header.  In particular, this stores the thunk's cache of all NcclCliques it's
// ever used.  This causes those cliques to stay alive as long as the thunk
// lives, which is how we avoid expensive reinitialization of NCCL cliques.
struct NcclAllReduceThunk::AuxData {
  tensorflow::mutex mu;
  absl::flat_hash_set<std::shared_ptr<NcclClique>> cliques GUARDED_BY(mu);
};

/*static*/ bool NcclAllReduceThunk::CanImplement(const HloInstruction* crs) {
  return MatchAllReduceComputation(crs->to_apply()).has_value() &&
         MatchNcclDataType(crs).has_value() && crs->IsCrossReplicaAllReduce() &&
         crs->operand_count() == 1;  // One array to reduce.
}

/*static*/ absl::flat_hash_set<int>
NcclAllReduceThunk::DevicesWithOpenNcclChannels() {
  absl::flat_hash_set<int> devices;
  GlobalNcclCliqueMap().ForEach(
      [&](const NcclCliqueKey& k, const std::shared_ptr<NcclClique>&) {
        devices.insert(k.devices.begin(), k.devices.end());
      });
  return devices;
}

NcclAllReduceThunk::NcclAllReduceThunk(
    int64 replica_count, int64 element_count,
    const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer,
    const HloInstruction* all_reduce)
    : Thunk(Thunk::kNcclAllReduce, all_reduce),
      replica_count_(replica_count),
      element_count_(element_count),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      aux_data_(absl::make_unique<AuxData>()) {}

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_ordinal.
static StatusOr<std::vector<int64>> GetParticipatingReplicas(
    int64 device_ordinal, const HloAllReduceInstruction* instr,
    int64 total_replica_count, const DeviceAssignment& device_assn) {
  std::vector<int64> participating_replicas;

  // Empty replica_groups() means that all replicas participate in one big
  // group.
  if (instr->replica_groups().empty()) {
    participating_replicas.resize(total_replica_count);
    absl::c_iota(participating_replicas, 0);
    return participating_replicas;
  }

  // Use the DeviceAssignment to figure out our replica-id.
  TF_ASSIGN_OR_RETURN(int replica_id,
                      device_assn.ReplicaIdForDeviceOrdinal(device_ordinal));

  // Figure out the other replicas that go together with this one.
  absl::optional<ReplicaGroup> replica_group;
  for (const ReplicaGroup& g : instr->replica_groups()) {
    if (absl::c_linear_search(g.replica_ids(), replica_id)) {
      CHECK(!replica_group.has_value())
          << "Replica appears twice in replica groups? " << instr->ToString();
      replica_group = g;
    }
  }
  CHECK(replica_group.has_value())
      << "Replica " << replica_id << " doesn't appear in replica groups? "
      << instr->ToString();

  participating_replicas.insert(participating_replicas.begin(),
                                replica_group->replica_ids().begin(),
                                replica_group->replica_ids().end());
  return participating_replicas;
}

Status NcclAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << "Starting NcclAllReduceThunk.";
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  auto* instr = Cast<HloAllReduceInstruction>(hlo_instruction());
  int64 device_ordinal = params.stream->parent()->device_ordinal();

  TF_ASSIGN_OR_RETURN(
      std::vector<int64> participating_replicas,
      GetParticipatingReplicas(device_ordinal, instr, replica_count_,
                               *params.device_assn));

  // Find or create the rendezvous for this collective operation.
  RendezvousKey rendezvous_key(
      params.run_id, participating_replicas,
      Cast<HloAllReduceInstruction>(hlo_instruction()));
  std::shared_ptr<Rendezvous> rendezvous =
      GlobalRendezvousMap()[rendezvous_key];

  VLOG(2) << "Rendezvous key: " << rendezvous_key.ToString()
          << ", rendezvous: " << rendezvous.get()
          << ", participating replicas: "
          << absl::StrJoin(participating_replicas, ", ");

  ParticipantData participant(rendezvous_key);
  participant.element_count = element_count_;
  participant.device_ordinal = device_ordinal;
  participant.source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  participant.destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);
  participant.stream = params.stream;

  // Do the operation.
  StatusOr<std::pair<std::shared_ptr<NcclClique>,
                     std::shared_ptr<tensorflow::BlockingCounter>>>
      result = rendezvous->SubmitParticipant(participant);
  if (!result.ok()) {
    VLOG(1) << "NcclAllReduceThunk::ExecuteOnStream failed: "
            << result.status().ToString();
    return result.status();
  }

  std::shared_ptr<NcclClique> clique;
  std::shared_ptr<tensorflow::BlockingCounter> blocking_counter;
  std::tie(clique, blocking_counter) = std::move(result).ValueOrDie();

  // Keep the clique we used alive for as long as this Thunk lives.  Creating
  // new NCCL cliques is expensive, and this is how we avoid thrashing them.
  {
    tensorflow::mutex_lock lock(aux_data_->mu);
    aux_data_->cliques.insert(std::move(clique));
  }

  // Drop our reference to the Rendezvous and wait for all other threads to do
  // the same.  If we didn't do this, one of the threads could run past this
  // point, reenter ExecuteOnStream for another all-reduce, and attempt to reuse
  // the Rendezvous!
  //
  // An alternative way of accomplishing this goal would be to implement
  // RefcountingHashMap::erase() and call it during SubmitParticipant.  But
  // erase() is deceptively complex to implement correctly.
  rendezvous.reset();
  blocking_counter->DecrementCount();
  WaitAndLogIfStuck(blocking_counter.get(), [&] {
    return absl::StrFormat(
        "participant for device ordinal %d, stream %p waiting for "
        "all threads to drop their reference to the rendezvous: %s",
        device_ordinal, params.stream, rendezvous_key.ToString());
  });

  return Status::OK();
}

NcclAllReduceThunk::~NcclAllReduceThunk() {}

}  // namespace gpu
}  // namespace xla
