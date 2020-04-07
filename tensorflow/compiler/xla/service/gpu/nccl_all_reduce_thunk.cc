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
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#endif
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

#if TENSORFLOW_USE_ROCM
// Local hipify of cuda symbols
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
#endif

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

absl::optional<ncclDataType_t> DatatypeToNccl(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return ncclInt8;
    case PRED:
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

Status StringToNcclUniqueId(const std::string& str_id, ncclUniqueId* nccl_id) {
  if (str_id.size() != NCCL_UNIQUE_ID_BYTES) {
    return InvalidArgument(
        "ncclUniqueId string must have %d bytes, got %d bytes", str_id.size(),
        NCCL_UNIQUE_ID_BYTES);
  }
  // NcclUniqueId is internally just a char[].
  static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
                "NCCL_UNIQUE_ID_BYTES");
  std::memcpy(static_cast<void*>(nccl_id), str_id.data(), NCCL_UNIQUE_ID_BYTES);
  return Status::OK();
}

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
  explicit NcclClique(
      int64 num_global_devices, std::vector<int64> local_device_ordinals,
      std::vector<int64> local_device_ranks,
      const StatusOr<absl::optional<std::string>>& nccl_unique_id)
      : num_global_devices_(num_global_devices),
        local_device_ordinals_(std::move(local_device_ordinals)),
        local_device_ranks_(std::move(local_device_ranks)) {
    CHECK_EQ(local_device_ordinals_.size(), local_device_ranks_.size());
    // It's unusual to pass a StatusOr<> into a class, but since this class
    // already has a erroneous state, it turns out to be a little easier to
    // implement this way than to change RefcountingHashMap.
    status_ = Init(nccl_unique_id);
  }

  Status status() { return status_; }

  // A NCCL communicator is the NCCL state associated with a participant (rank)
  // in a reduction. This method returns the state associated with a particular
  // local device ordinal.
  ncclComm_t comm(int64 device_ordinal) {
    int64 idx =
        std::distance(local_device_ordinals_.begin(),
                      absl::c_find(local_device_ordinals_, device_ordinal));
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
  Status Init(
      const StatusOr<absl::optional<std::string>>& maybe_nccl_unique_id) {
    VLOG(3) << absl::StreamFormat(
        "Initializing nccl comms for participant device ordinals %s ranks {%s}",
        absl::StrJoin(local_device_ordinals_, ", "),
        absl::StrJoin(local_device_ranks_, ", "));

    // Restore CUDA device after running this.  XLA shouldn't care, but maybe
    // another consumer does.
    int initial_cuda_device;
    XLA_CUDA_RETURN_IF_ERROR(cudaGetDevice(&initial_cuda_device));
    auto cuda_device_restorer = MakeCleanup(
        [&] { XLA_CUDA_WARN_IF_ERROR(cudaSetDevice(initial_cuda_device)); });

    // When using ncclGroupStart/End it seems that the ncclComm_t's are not
    // populated until the End() call.  This unfortunately makes error handling
    // tricky.
    std::vector<ncclComm_t> raw_comms(local_device_ordinals_.size(), nullptr);
    TF_ASSIGN_OR_RETURN(const absl::optional<std::string>& nccl_id_string,
                        maybe_nccl_unique_id);

    ncclUniqueId nccl_id;
    if (nccl_id_string) {
      TF_RETURN_IF_ERROR(StringToNcclUniqueId(*nccl_id_string, &nccl_id));
    } else {
      XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
    }
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
    Status status = [&] {
      for (int i = 0; i < local_device_ordinals_.size(); ++i) {
        XLA_CUDA_RETURN_IF_ERROR(cudaSetDevice(local_device_ordinals_[i]));
        XLA_CUDA_RETURN_IF_ERROR(ncclCommInitRank(&raw_comms[i],
                                                  num_global_devices_, nccl_id,
                                                  local_device_ranks_.at(i)));
      }
      return Status::OK();
    }();
    // Always call ncclGroupEnd().
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

    // Populate comms_ from the raw comms we created above.  If we encountered
    // an error above we'll later clear comms_ thus destroying any raw comms
    // that were created before the error.
    for (int i = 0; i < local_device_ordinals_.size(); ++i) {
      VLOG(3) << absl::StreamFormat("Device ordinal %d assigned ncclComm %p",
                                    local_device_ordinals_[i], raw_comms[i]);
      CHECK(raw_comms[i] != nullptr || !status.ok());
      comms_.emplace_back(raw_comms[i]);
    }
    if (!status.ok()) {
      comms_.clear();
    }

    return status;
  }

  Status status_;
  int64 num_global_devices_;
  std::vector<int64> local_device_ordinals_;
  // NCCL communicator rank for each local device. The rank of a device is equal
  // to the offset of the local device in the global device set.
  std::vector<int64> local_device_ranks_;
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
  static auto& m = *new RefcountingHashMap<NcclCliqueKey, NcclClique>();
  return m;
}

using RendezvousBase =
    Rendezvous<AllReduceParticipantData, std::shared_ptr<NcclClique>>;
class RendezvousNcclAllReduce : public RendezvousBase {
 public:
  explicit RendezvousNcclAllReduce(const RendezvousKey& k)
      : RendezvousBase(k) {}

 protected:
  StatusOr<ParticipantImplOutput> SubmitParticipantImpl(
      const AllReduceParticipantData& participant) override;

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
      *new RefcountingHashMap<RendezvousKey, RendezvousNcclAllReduce>();
  return m;
}

StatusOr<RendezvousNcclAllReduce::ParticipantImplOutput>
RendezvousNcclAllReduce::SubmitParticipantImpl(
    const AllReduceParticipantData& participant) {
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

    TF_RET_CHECK(participant.local_devices.size() ==
                 participant.rendezvous_key.num_local_participants);

    // Look up or create the NCCL clique for this set of devices.
    NcclCliqueKey clique_key(participant.rendezvous_key.global_devices);

    auto clique_factory =
        [&](const NcclCliqueKey& key) -> std::unique_ptr<NcclClique> {
      std::vector<int64> local_device_ranks;
      std::vector<int64> local_device_ordinals;
      local_device_ranks.reserve(participant.local_devices.size());
      local_device_ordinals.reserve(participant.local_devices.size());
      for (const auto& l : participant.local_devices) {
        auto it =
            absl::c_find(participant.rendezvous_key.global_devices, l.first);
        CHECK(it != participant.rendezvous_key.global_devices.end()) << l.first;
        local_device_ranks.push_back(std::distance(
            participant.rendezvous_key.global_devices.begin(), it));
        local_device_ordinals.push_back(l.second);
      }
      StatusOr<absl::optional<std::string>> nccl_unique_id;
      if (participant.nccl_unique_id_callback) {
        nccl_unique_id = (*participant.nccl_unique_id_callback)(clique_key);
      } else {
        if (participant.rendezvous_key.global_devices.size() !=
            participant.rendezvous_key.num_local_participants) {
          nccl_unique_id = InvalidArgument(
              "Multihost AllReduce on GPU requires a nccl_unique_id_callback "
              "to be provided by the client.");
        } else {
          nccl_unique_id = absl::optional<std::string>();
        }
      }
      return absl::make_unique<NcclClique>(
          participant.rendezvous_key.global_devices.size(),
          std::move(local_device_ordinals), std::move(local_device_ranks),
          nccl_unique_id);
    };
    clique =
        GlobalNcclCliqueMap().GetOrCreateIfAbsent(clique_key, clique_factory);

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

  se::StreamExecutor* executor = participant.stream->parent();
  se::gpu::ScopedActivateExecutorContext scoped_context(executor);
  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      participant.stream->implementation()->GpuStreamMemberHack());
  VLOG(3) << "Using stream pointer: " << cu_stream
          << " on device: " << participant.device_ordinal;
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (auto& buffer : participant.buffers) {
    void* send_buffer = const_cast<void*>(buffer.source_data.opaque());
    void* recv_buffer = const_cast<void*>(buffer.destination_data.opaque());
    absl::optional<ncclDataType_t> allreduce_datatype =
        DatatypeToNccl(buffer.primitive_type);
    CHECK(allreduce_datatype.has_value());
    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, buffer.element_count,
        static_cast<const void*>(comm), cu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           /*count=*/buffer.element_count,
                                           /*datatype=*/*allreduce_datatype,
                                           /*op=*/computation,
                                           /*comm=*/comm,
                                           /*stream=*/*cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all reduce for ordinal: "
          << participant.device_ordinal;
  VLOG(3) << "This thread done with all-reduce op.";

  return ParticipantImplOutput{primary, clique};
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
  absl::flat_hash_set<std::shared_ptr<NcclClique>> cliques TF_GUARDED_BY(mu);
};

/*static*/ bool NcclAllReduceThunk::CanImplement(const HloInstruction* crs) {
  auto operands_are_supported = [crs]() {
    return absl::c_all_of(crs->operands(), [](HloInstruction* operand) {
      return LayoutUtil::IsDenseArray(operand->shape()) &&
             DatatypeToNccl(operand->shape().element_type()).has_value();
    });
  };
  return MatchReductionComputation(crs->to_apply()).has_value() &&
         crs->IsCrossReplicaAllReduce() && operands_are_supported();
}

/*static*/ absl::flat_hash_set<GlobalDeviceId>
NcclAllReduceThunk::DevicesWithOpenNcclChannels() {
  absl::flat_hash_set<GlobalDeviceId> devices;
  GlobalNcclCliqueMap().ForEach(
      [&](const NcclCliqueKey& k, const std::shared_ptr<NcclClique>&) {
        devices.insert(k.devices().begin(), k.devices().end());
      });
  return devices;
}

NcclAllReduceThunk::NcclAllReduceThunk(
    int64 replica_count, std::vector<NcclAllReduceThunk::Buffer> buffers,
    const HloInstruction* all_reduce)
    : Thunk(Thunk::kNcclAllReduce, all_reduce),
      replica_count_(replica_count),
      buffers_(std::move(buffers)),
      aux_data_(absl::make_unique<AuxData>()) {
  CHECK_EQ(hlo_instruction()->operand_count(), buffers_.size());
}

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_ordinal.
Status NcclAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << "Starting NcclAllReduceThunk.";
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  auto* instr = Cast<HloAllReduceInstruction>(hlo_instruction());
  int64 local_device_ordinal = params.stream->parent()->device_ordinal();
  GlobalDeviceId global_device_id;
  if (params.gpu_global_device_ids) {
    TF_RET_CHECK(0 <= local_device_ordinal &&
                 local_device_ordinal < params.gpu_global_device_ids->size());
    global_device_id = (*params.gpu_global_device_ids)[local_device_ordinal];
  } else {
    // No local -> global mapping was provided; assume the identity mapping.
    global_device_id = GlobalDeviceId(local_device_ordinal);
  }

  // Determines the set of global and local devices that are participating in
  // the same collective group as the caller.
  TF_ASSIGN_OR_RETURN(
      std::vector<int64> global_participating_replicas,
      GetParticipatingReplicas(global_device_id, instr->replica_groups(),
                               replica_count_, *params.device_assn));
  std::vector<GlobalDeviceId> global_devices;
  std::vector<std::pair<GlobalDeviceId, int64>> local_devices;
  local_devices.reserve(global_participating_replicas.size());
  global_devices.reserve(global_participating_replicas.size());
  TF_RET_CHECK(params.device_assn->computation_count() == 1)
      << params.device_assn->ToString();
  for (int64 replica : global_participating_replicas) {
    GlobalDeviceId global_device(
        (*params.device_assn)(replica, /*computation=*/0));
    global_devices.push_back(global_device);
    if (!params.gpu_global_device_ids) {
      local_devices.emplace_back(global_device, global_device.value());
    } else {
      auto it = absl::c_find(*params.gpu_global_device_ids, global_device);
      if (it != params.gpu_global_device_ids->end()) {
        local_devices.emplace_back(
            *it, std::distance(params.gpu_global_device_ids->begin(), it));
      }
    }
  }
  absl::c_sort(global_devices);

  // Find or create the rendezvous for this collective operation.
  RendezvousKey rendezvous_key = RendezvousKey::FromInstruction(
      params.run_id, global_devices, local_devices.size(), hlo_instruction());

  if (VLOG_IS_ON(2)) {
    std::vector<std::string> local_participants;
    for (const auto& entry : local_devices) {
      local_participants.push_back(absl::StrFormat(
          "global=%d/local=%d", entry.first.value(), entry.second));
    }
    VLOG(2) << "Rendezvous key: " << rendezvous_key.ToString()
            << ", global participating replicas: "
            << absl::StrJoin(global_participating_replicas, ", ")
            << ", global participating devices: "
            << GlobalDeviceIdsToString(global_devices)
            << ", local participants: "
            << absl::StrJoin(local_participants, ",");
  }
  AllReduceParticipantData participant(rendezvous_key);
  participant.device_ordinal = local_device_ordinal;
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const NcclAllReduceThunk::Buffer& buffer = buffers_[i];
    AllReduceParticipantData::Buffer pbuffer;
    pbuffer.element_count = buffer.element_count;
    pbuffer.source_data =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer);
    pbuffer.destination_data =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer);
    pbuffer.primitive_type =
        hlo_instruction()->operand(i)->shape().element_type();
    participant.buffers.push_back(pbuffer);
  }
  participant.stream = params.stream;
  participant.local_devices = std::move(local_devices);
  participant.nccl_unique_id_callback = params.nccl_unique_id_callback;
  auto reduction_kind =
      MatchReductionComputation(hlo_instruction()->to_apply());
  CHECK(reduction_kind.has_value());
  participant.reduction_kind = *reduction_kind;

  auto rendezvous_factory = [](const RendezvousKey& k) {
    return absl::make_unique<RendezvousNcclAllReduce>(k);
  };

  TF_ASSIGN_OR_RETURN(std::shared_ptr<NcclClique> clique,
                      RendezvousNcclAllReduce::SubmitParticipant(
                          [&] {
                            return GlobalRendezvousMap().GetOrCreateIfAbsent(
                                rendezvous_key, rendezvous_factory);
                          },
                          participant));

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
