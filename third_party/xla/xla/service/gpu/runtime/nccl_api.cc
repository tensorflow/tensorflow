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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/nccl_collectives.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
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
class DefaultNcclApi final : public NcclCollectives {
 public:
};

NcclApi* NcclApi::Default() {
  static auto* nccl_api = new DefaultNcclApi();
  return nccl_api;
}

bool NcclApi::HasNcclSupport() { return true; }

}  // namespace xla::gpu
