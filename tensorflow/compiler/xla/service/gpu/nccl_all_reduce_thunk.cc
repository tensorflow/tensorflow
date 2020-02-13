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
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "third_party/nccl/nccl.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
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

ncclRedOp_t ReductionKindToNccl(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

PrimitiveType AllReducePrimitiveType(const HloInstruction* instr) {
  return instr->operand(0)->shape().element_type();
}

absl::optional<ncclDataType_t> DatatypeToNccl(PrimitiveType element_type) {
  switch (element_type) {
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
  void Lock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    TF_CHECK_OK(status_);
    mu_->lock();
  }
  void Unlock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
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

class RendezvousNcclAllReduce : public Rendezvous<std::shared_ptr<NcclClique>> {
 public:
  explicit RendezvousNcclAllReduce(const RendezvousKey& k)
      : Rendezvous<std::shared_ptr<NcclClique>>(k) {}

 protected:
  StatusOr<std::pair<std::shared_ptr<NcclClique>, bool>> SubmitParticipantImpl(
      AllReduceParticipantData participant) override;

  void CleanupImpl(std::shared_ptr<NcclClique> handle,
                   bool is_primary) override;
};

// Global map of Rendezvous objects.  A thread participating in a collective op
// looks up its Rendezvous in this map to find the other threads that it's
// participating with.
//
// Rendezvous objects are one-time use, so they're removed from this map once
// we're through with them.
RefcountingHashMap<RendezvousKey, RendezvousNcclAllReduce>&
GlobalRendezvousMap() {
  static auto& m =
      *new RefcountingHashMap<RendezvousKey, RendezvousNcclAllReduce>(
          [](const RendezvousKey& k) {
            return absl::make_unique<RendezvousNcclAllReduce>(k);
          });
  return m;
}

StatusOr<std::pair<std::shared_ptr<NcclClique>, bool>>
RendezvousNcclAllReduce::SubmitParticipantImpl(
    AllReduceParticipantData participant) {
  // We pull into our thread a) the communication handle and b) whether we're
  // the "primary" thread for this rendezvous -- the "primary" thread has some
  // additional responsibilities for setup/teardown.
  ncclComm_t comm;
  bool primary;
  std::shared_ptr<NcclClique> clique;

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
      // We'll unlock it in CleanupImpl.
      clique->Lock();
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
  ncclRedOp_t computation = ReductionKindToNccl(participant.reduction_kind);
  absl::optional<ncclDataType_t> allreduce_datatype =
      DatatypeToNccl(participant.primitive_type);
  CHECK(allreduce_datatype.has_value());

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
      "comm=%p, stream=%p)",
      send_buffer, recv_buffer, participant.element_count,
      static_cast<const void*>(comm), cu_stream);
  XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                         /*count=*/participant.element_count,
                                         /*datatype=*/*allreduce_datatype,
                                         /*op=*/computation,
                                         /*comm=*/comm,
                                         /*stream=*/*cu_stream));

  VLOG(3) << "Done performing all reduce for ordinal: "
          << participant.device_ordinal;
  VLOG(3) << "This thread done with all-reduce op.";

  return std::make_pair(clique, primary);
}

void RendezvousNcclAllReduce::CleanupImpl(std::shared_ptr<NcclClique> handle,
                                          bool is_primary) {
  // Releases the lock on the clique (held only by the primary thread).
  if (is_primary) {
    handle->Unlock();
  }
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
  return MatchReductionComputation(crs->to_apply()).has_value() &&
         DatatypeToNccl(AllReducePrimitiveType(crs)).has_value() &&
         crs->IsCrossReplicaAllReduce() &&
         crs->operand_count() == 1 &&  // One array to reduce.
         LayoutUtil::IsDenseArray(crs->operand(0)->shape());
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
Status NcclAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << "Starting NcclAllReduceThunk.";
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  auto* instr = Cast<HloAllReduceInstruction>(hlo_instruction());
  int64 device_ordinal = params.stream->parent()->device_ordinal();

  TF_ASSIGN_OR_RETURN(
      std::vector<int64> participating_replicas,
      GetParticipatingReplicas(device_ordinal, instr->replica_groups(),
                               replica_count_, *params.device_assn));

  // Find or create the rendezvous for this collective operation.
  RendezvousKey rendezvous_key = RendezvousKey::FromInstruction(
      params.run_id, participating_replicas, hlo_instruction());

  VLOG(2) << "Rendezvous key: " << rendezvous_key.ToString()
          << ", participating replicas: "
          << absl::StrJoin(participating_replicas, ", ");

  AllReduceParticipantData participant(rendezvous_key);
  participant.element_count = element_count_;
  participant.device_ordinal = device_ordinal;
  participant.source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  participant.destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);
  participant.stream = params.stream;
  auto reduction_kind =
      MatchReductionComputation(hlo_instruction()->to_apply());
  CHECK(reduction_kind.has_value());
  participant.reduction_kind = *reduction_kind;
  participant.primitive_type = AllReducePrimitiveType(hlo_instruction());

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<NcclClique> clique,
      RendezvousNcclAllReduce::SubmitParticipant(
          [&] { return GlobalRendezvousMap()[rendezvous_key]; }, participant));

  // Keep the clique we used alive for as long as this Thunk lives.  Creating
  // new NCCL cliques is expensive, and this is how we avoid thrashing them.
  {
    tensorflow::mutex_lock lock(aux_data_->mu);
    aux_data_->cliques.insert(std::move(clique));
  }
  return Status::OK();
}

NcclAllReduceThunk::~NcclAllReduceThunk() {}

}  // namespace gpu
}  // namespace xla
