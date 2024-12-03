/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/nccl_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

#if (defined(PLATFORM_GOOGLE) && !defined(TENSORFLOW_USE_ROCM))
#define WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT true
#else
#define WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT false
#endif

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NCCL errors.
//==-----------------------------------------------------------------------===//

static absl::Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                             const char* expr) {
  if (s == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "%s:%d: NCCL operation %s failed: %s."
      " Last NCCL warning(error) log entry (may be unrelated) '%s'.",
      file, line, expr, ncclGetErrorString(s), ncclGetLastError(nullptr)));
}

#define XLA_NCCL_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NCCL_RETURN_IF_ERROR(expr)      \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      return s;                             \
    }                                       \
  } while (0)

#define XLA_NCCL_LOG_IF_ERROR(expr)         \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      LOG(ERROR) << s.ToString();           \
    }                                       \
  } while (0)

#define XLA_NCCL_CHECK(expr) CHECK(XLA_NCCL_STATUS(expr).ok())

//==-----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType dtype,
                                                     bool is_reduction_op) {
  switch (dtype) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
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
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type for reduction operation: %s",
                            primitive_util::LowercasePrimitiveTypeName(dtype)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %s",
                          primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

static ncclRedOp_t ToNcclReduction(ReductionKind kind) {
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

//==-----------------------------------------------------------------------===//
// Casting between opaque API structs and NCCL types.
//==-----------------------------------------------------------------------===//

static ncclComm_t Cast(const Communicator* comm) {
  auto* nccl_communicator = tsl::down_cast<const NcclCommunicator*>(comm);
  CHECK(nccl_communicator != nullptr) << "Unsupported XLA communicator";
  return nccl_communicator->comm();
}

#if WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
static ncclPersistentPlanAllocator* Cast(
    NcclApi::NcclPersistentPlanAllocatorHandle handle) {
  return reinterpret_cast<ncclPersistentPlanAllocator*>(handle);
}

static ncclPersistentPlanAllocator** Cast(
    NcclApi::NcclPersistentPlanAllocatorHandle* handle) {
  return reinterpret_cast<ncclPersistentPlanAllocator**>(handle);
}

static NcclApi::NcclPersistentPlanAllocatorHandle Cast(
    ncclPersistentPlanAllocator* ptr) {
  return reinterpret_cast<NcclApi::NcclPersistentPlanAllocatorHandle>(ptr);
}
#endif  // WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT

//==-----------------------------------------------------------------------===//
// NcclApi::PersistentPlanAllocator
//==-----------------------------------------------------------------------===//

using PersistentPlanAllocator = NcclApi::PersistentPlanAllocator;
using ScopedPersistentPlanAllocator = NcclApi::ScopedPersistentPlanAllocator;

PersistentPlanAllocator::PersistentPlanAllocator(
    int64_t device_ordinal, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream)
    : handle_(nullptr),
      device_ordinal_(device_ordinal),
      allocator_(allocator),
      stream_(stream) {
  // NCCL persistent plan allocator is implemented as NCCL patch that is not yet
  // open sourced and can't be used from OSS XLA.
#if WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
  auto* nccl_allocator = new ncclPersistentPlanAllocator;
  nccl_allocator->ctl = this;

  nccl_allocator->alloc = +[](void** ptr, void* src, size_t size, void* ctl) {
    auto allocator = reinterpret_cast<PersistentPlanAllocator*>(ctl);
    auto allocated = allocator->AllocateAndInitialize(src, size);
    if (!allocated.ok()) return ncclInternalError;
    *ptr = allocated->opaque();
    allocator->AddRef();
    return ncclSuccess;
  };

  nccl_allocator->free = +[](void* ptr, void* ctl) -> ncclResult_t {
    auto allocator = reinterpret_cast<PersistentPlanAllocator*>(ctl);
    auto status = allocator->Deallocate(se::DeviceMemoryBase(ptr));
    allocator->DropRef();
    return status.ok() ? ncclSuccess : ncclInternalError;
  };

  handle_ = Cast(nccl_allocator);
#endif  // WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
}

PersistentPlanAllocator::~PersistentPlanAllocator() {
#if WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
  delete Cast(handle_);
#endif  // WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
}

absl::StatusOr<se::DeviceMemoryBase>
PersistentPlanAllocator::AllocateAndInitialize(void* src, size_t size) {
  TF_ASSIGN_OR_RETURN(auto owned_mem,
                      allocator_->Allocate(device_ordinal_, size));
  VLOG(5) << "Allocate and initialize NCCL persistent plan; mem="
          << owned_mem->opaque() << "; size=" << size;
  se::DeviceMemoryBase mem = owned_mem.Release();
  TF_RETURN_IF_ERROR(stream_->Memcpy(&mem, src, size));
  return mem;
}

absl::Status PersistentPlanAllocator::Deallocate(se::DeviceMemoryBase mem) {
  VLOG(5) << "Deallocate NCCL persistent plan; mem=" << mem.opaque();
  return allocator_->Deallocate(device_ordinal_, mem);
}

ScopedPersistentPlanAllocator::ScopedPersistentPlanAllocator(
    Communicator* comm, tsl::RCReference<PersistentPlanAllocator> allocator)
    : comm_(comm), allocator_(std::move(allocator)) {
#if WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
  XLA_NCCL_CHECK(
      ncclCommGetPersistentPlanAllocator(Cast(comm_), Cast(&recover_)))
      << "Failed to get NCCL persistent plan allocator";
  XLA_NCCL_CHECK(ncclCommSetPersistentPlanAllocator(Cast(comm_),
                                                    Cast(allocator_->handle())))
      << "Failed to set NCCL persistent plan allocator";
#endif  // WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
}

ScopedPersistentPlanAllocator::~ScopedPersistentPlanAllocator() {
#if WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
  XLA_NCCL_CHECK(
      ncclCommSetPersistentPlanAllocator(Cast(comm_), Cast(recover_)))
      << "Failed to set NCCL persistent plan allocator";
#endif  // WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT
}

//==-----------------------------------------------------------------------===//
// NcclApi
//==-----------------------------------------------------------------------===//

// This a default NCCL API implementation that forwards all API calls to NCCL
// itself. It is available only if NCCL + CUDA are configured at compile time.
class DefaultNcclApi final : public NcclApi {
 public:
  absl::StatusOr<NcclCliqueId> GetUniqueId() final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> CommInitRanks(
      int32_t nranks, const NcclCliqueId& clique_id,
      absl::Span<const DeviceRank> ranks, const Config& config) final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> CommSplit(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, std::optional<Config> config) final;

  absl::Status CommAbort(Communicator* comm) final;
  absl::Status CommFinalize(Communicator* comm) final;

  absl::StatusOr<int32_t> CommCount(Communicator* comm) final;

  absl::Status CommGetAsyncError(Communicator* comm) final;

  absl::Status GroupStart() final;
  absl::Status GroupEnd() final;

  absl::Status AllReduce(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         Communicator* comm, se::Stream* stream) final;

  absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, size_t root, Communicator* comm,
                         se::Stream* stream) final;

  absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             ReductionKind reduction_kind, Communicator* comm,
                             se::Stream* stream) final;

  absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, Communicator* comm,
                         se::Stream* stream) final;

  absl::Status Send(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                    size_t count, int32_t peer, Communicator* comm,
                    se::Stream* stream) final;
  absl::Status SendPtrToPeer(void* ptr, int32_t peer, Communicator* comm,
                             se::Stream* stream) final;

  absl::Status Recv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                    size_t count, int32_t peer, Communicator* comm,
                    se::Stream* stream) final;
  absl::Status RecvPtrFromPeer(void* ptr, int32_t peer, Communicator* comm,
                               se::Stream* stream) final;

  absl::StatusOr<NcclRegisteredBufferHandle> RegisterBuffer(
      Communicator* comm, se::DeviceMemoryBase buffer) final;

  absl::StatusOr<NcclRegisteredBufferHandle> DeregisterBuffer(
      Communicator* comm, NcclRegisteredBufferHandle handle) final;
};

NcclApi* NcclApi::Default() {
  static auto* nccl_api = new DefaultNcclApi();
  return nccl_api;
}

bool NcclApi::HasNcclSupport() { return true; }

static absl::StatusOr<ncclUniqueId> AsNcclUniqueId(
    const NcclCliqueId& clique_id) {
  if (clique_id.size() != NCCL_UNIQUE_ID_BYTES) {
    return Internal(
        "CliqueId size is not equal to NCCL_UNIQUE_ID_BYTES: %d vs %d",
        clique_id.size(), NCCL_UNIQUE_ID_BYTES);
  }
  ncclUniqueId id;
  absl::c_copy(clique_id.data(), id.internal);
  return id;
}

absl::StatusOr<NcclCliqueId> DefaultNcclApi::GetUniqueId() {
  VLOG(3) << "Get NCCL unique id";
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return NcclCliqueId(std::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
DefaultNcclApi::CommInitRanks(int32_t nranks, const NcclCliqueId& clique_id,
                              absl::Span<const DeviceRank> ranks,
                              const Config& config) {
  VLOG(1) << "Initialize NCCL communicator for " << ranks.size()
          << " devices; fingerprint(id)=" << clique_id.fingerprint();

  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION > 50700
  comm_config.splitShare = config.split_share;
#endif
  if (config.max_nchannels > 0) {
    comm_config.maxCTAs = config.max_nchannels;
    VLOG(1) << "Maximum number of channels for fingerprint(id)="
            << clique_id.fingerprint() << " is set to: " << comm_config.maxCTAs;
  }

  std::vector<ncclComm_t> comm_handles;
  std::vector<std::unique_ptr<Communicator>> comms;

  comm_handles.resize(ranks.size(), nullptr);
  comms.reserve(ranks.size());

  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < ranks.size(); ++i) {
    VLOG(1) << "Initialize NCCL communicator for rank #" << ranks[i].rank
            << " of " << nranks
            << "; fingerprint(id)=" << clique_id.fingerprint();
    auto activate_context = ranks[i].device->Activate();

    TF_ASSIGN_OR_RETURN(auto nccl_unique_id, AsNcclUniqueId(clique_id));
    XLA_NCCL_RETURN_IF_ERROR(
        ncclCommInitRankConfig(&comm_handles[i], nranks, nccl_unique_id,
                               ranks[i].rank.value(), &comm_config));
  }
  TF_RETURN_IF_ERROR(GroupEnd());

  for (ncclComm_t comm_handle : comm_handles) {
    comms.emplace_back(std::make_unique<NcclCommunicator>(comm_handle));
  }

  return comms;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
DefaultNcclApi::CommSplit(absl::Span<const Communicator* const> comms,
                          int32_t color, absl::Span<const RankId> keys,
                          std::optional<Config> config) {
  auto rank_formatter = [](std::string* str, RankId rank) {
    absl::StrAppend(str, rank.value());
  };

  VLOG(1) << absl::StreamFormat(
      "Split %d NCCL communicators using color %d and keys: [%s]", comms.size(),
      color, absl::StrJoin(keys, ",", rank_formatter));

#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
  if (keys.size() != comms.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Comms and keys must have the same size, but %d != %d",
                        comms.size(), keys.size()));
  }

  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
  if (config.has_value()) {
    comm_config.splitShare = config.value().split_share;
    // If max_nchannels is set, then we don't want to
    // inherit from parent comm.
    if (config.value().max_nchannels > 0) {
      comm_config.maxCTAs = config.value().max_nchannels;
      VLOG(1) << "CommSplit maximum number of channels "
              << " is set to: " << comm_config.maxCTAs;
    }
  }

  // In contrast to grouped initialization communicator splitting initializes
  // communicators only after a successful call to `GroupEnd`, so we keep a
  // vector of handles and after successful splitting convert to RAII wrappers.
  std::vector<ncclComm_t> split_comms_handles;
  split_comms_handles.resize(comms.size(), nullptr);

  ncclConfig_t* comm_config_ptr = config.has_value() ? &comm_config : nullptr;
  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < comms.size(); ++i) {
    VLOG(1) << "Split NCCL communicator " << comms[i] << " with color " << color
            << " and key " << keys[i];
    XLA_NCCL_RETURN_IF_ERROR(ncclCommSplit(
        Cast(comms[i]), color, keys[i].value(), &split_comms_handles[i],
        /*config=*/comm_config_ptr));
  }
  TF_RETURN_IF_ERROR(GroupEnd());

  std::vector<std::unique_ptr<Communicator>> split_comms;
  split_comms.reserve(split_comms_handles.size());
  for (size_t i = 0; i < split_comms_handles.size(); ++i) {
    split_comms.emplace_back(
        std::make_unique<NcclCommunicator>(split_comms_handles[i]));
  }
  return split_comms;
#else
  return absl::UnimplementedError(
      absl::StrFormat("%s:%d: NCCL operation ncclCommSplit not implemented",
                      __FILE__, __LINE__));
#endif  // !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
}

absl::Status DefaultNcclApi::CommAbort(Communicator* comm) {
  VLOG(1) << "Abort NCCL communicator: " << comm;
  return XLA_NCCL_STATUS(ncclCommAbort(Cast(comm)));
}

absl::Status DefaultNcclApi::CommFinalize(Communicator* comm) {
  VLOG(1) << "Finalize NCCL communicator: " << comm;
  return XLA_NCCL_STATUS(ncclCommFinalize(Cast(comm)));
}

absl::StatusOr<int32_t> DefaultNcclApi::CommCount(Communicator* comm) {
  VLOG(5) << "Get the number of ranks in NCCL communicator: " << comm;
  int32_t count;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(Cast(comm), &count));
  return count;
}

absl::Status DefaultNcclApi::CommGetAsyncError(Communicator* comm) {
  VLOG(5) << "Get last async error for NCCL communicator: " << comm;

  ncclResult_t async_err;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(Cast(comm), &async_err));
  if (async_err == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrCat(
      ncclGetErrorString(async_err),
      ". Last NCCL error (maybe unrelated): ", ncclGetLastError(Cast(comm))));
}

absl::Status DefaultNcclApi::GroupStart() {
  VLOG(5) << "Start NCCL group";
  return XLA_NCCL_STATUS(ncclGroupStart());
}

absl::Status DefaultNcclApi::GroupEnd() {
  VLOG(5) << "End NCCL group";
  return XLA_NCCL_STATUS(ncclGroupEnd());
}

absl::Status DefaultNcclApi::AllReduce(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       Communicator* comm, se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), Cast(comm),
      se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::Broadcast(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       size_t root, Communicator* comm,
                                       se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Broadcast operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root, comm, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, root, Cast(comm), se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                           se::DeviceMemoryBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           Communicator* comm,
                                           se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL ReduceScatter operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), Cast(comm),
      se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::AllGather(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       Communicator* comm, se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllGather operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, Cast(comm), se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::Send(se::DeviceMemoryBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  int32_t peer, Communicator* comm,
                                  se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Send operation on device #%d; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer, comm,
      stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(
      ncclSend(send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer, Cast(comm), se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::SendPtrToPeer(void* ptr, int32_t peer,
                                           Communicator* comm,
                                           se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL RecvPtrFromPeer operation on device #%d;  "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), peer, comm, stream);
  return XLA_NCCL_STATUS(ncclSend(ptr, 1, ncclUint64, peer, Cast(comm),
                                  se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::Recv(se::DeviceMemoryBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  int32_t peer, Communicator* comm,
                                  se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Recv operation on device #%d; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer, comm,
      stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(
      ncclRecv(recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
               peer, Cast(comm), se::gpu::AsGpuStreamValue(stream)));
}

absl::Status DefaultNcclApi::RecvPtrFromPeer(void* ptr, int32_t peer,
                                             Communicator* comm,
                                             se::Stream* stream) {
  VLOG(3) << absl::StreamFormat(
      "Launch NCCL RecvPtrFromPeer operation on device #%d; "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), peer, comm, stream);

  return XLA_NCCL_STATUS(ncclRecv(ptr, 1, ncclUint64, peer, Cast(comm),
                                  se::gpu::AsGpuStreamValue(stream)));
}

absl::StatusOr<NcclApi::NcclRegisteredBufferHandle>
DefaultNcclApi::RegisterBuffer(Communicator* comm,
                               se::DeviceMemoryBase buffer) {
  VLOG(3) << absl::StreamFormat(
      "Register buffer for NCCL communicator; buffer=%p; size=%d; comm=%p",
      buffer.opaque(), buffer.size(), comm);
  void* handle = nullptr;
#if (NCCL_VERSION_CODE >= 21901)
  XLA_NCCL_RETURN_IF_ERROR(
      ncclCommRegister(Cast(comm), buffer.opaque(), buffer.size(), &handle));
#endif  // NCCL_VERSION_CODE >= 21901
  return reinterpret_cast<NcclRegisteredBufferHandle>(handle);
}

absl::StatusOr<NcclApi::NcclRegisteredBufferHandle>
DefaultNcclApi::DeregisterBuffer(Communicator* comm,
                                 NcclRegisteredBufferHandle handle) {
  VLOG(3) << absl::StreamFormat(
      "Deregister buffer for NCCL communicator; handle=%p; comm=%p", handle,
      comm);
#if (NCCL_VERSION_CODE >= 21901)
  return XLA_NCCL_STATUS(
      ncclCommDeregister(Cast(comm), reinterpret_cast<void*>(handle)));
#else
  return absl::UnimplementedError(
      "ncclCommDeregister is unavailable in this build.");
#endif  // NCCL_VERSION_CODE >= 21901
}
}  // namespace xla::gpu
