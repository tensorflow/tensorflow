/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_utils.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique.h"
#include "xla/service/gpu/nccl_types.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

absl::StatusOr<NcclRedOp> ToNcclReduction(ReductionKind kind) {
#if XLA_ENABLE_XCCL
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
#endif  // XLA_ENABLE_XCCL

  return absl::InternalError("XLA compiled without NCCL");
}

static absl::StatusOr<NcclDataType> ToNcclDataType(PrimitiveType element_type,
                                                   Thunk::Kind reduction_op) {
#if XLA_ENABLE_XCCL
  switch (element_type) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
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
      // For all-reduce and reduce-scatter, we expect 16 bit integer types to be
      // promoted to 32-bit.
      if (reduction_op == Thunk::kNcclAllReduce ||
          reduction_op == Thunk::kNcclAllReduceStart ||
          reduction_op == Thunk::kNcclReduceScatter) {
        return tsl::errors::InvalidArgument(absl::StrFormat(
            "Unsupported data type: %s", PrimitiveType_Name(element_type)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
#if defined(__CUDA_BF16_TYPES_EXIST__) || TENSORFLOW_USE_ROCM
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tsl::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
#endif  // XLA_ENABLE_XCCL

  return absl::InternalError("XLA compiled without NCCL");
}

absl::StatusOr<std::pair<NcclDataType, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op) {
  TF_ASSIGN_OR_RETURN(NcclDataType dtype,
                      ToNcclDataType(element_type, reduction_op));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

#ifdef PLATFORM_GOOGLE
//==-----------------------------------------------------------------------===//
// RAII helper to set NCCL persistent plan allocator.
//==-----------------------------------------------------------------------===//

NcclPersistentPlanAllocator::NcclPersistentPlanAllocator(
    int64_t device_ordinal, stream_executor::DeviceMemoryAllocator* allocator,
    stream_executor::Stream* stream)
    : device_ordinal_(device_ordinal), allocator_(allocator), stream_(stream) {
  nccl_allocator_.ctl = this;

  nccl_allocator_.alloc = +[](void** ptr, void* src, size_t size, void* ctl) {
    auto allocator = reinterpret_cast<NcclPersistentPlanAllocator*>(ctl);
    auto allocated = allocator->AllocateAndInitialize(src, size);
    if (!allocated.ok()) return ncclInternalError;
    allocator->AddRef();
    *ptr = allocated->opaque();
    return ncclSuccess;
  };

  nccl_allocator_.free = +[](void* ptr, void* ctl) -> ncclResult_t {
    auto allocator = reinterpret_cast<NcclPersistentPlanAllocator*>(ctl);
    auto status = allocator->Deallocate(stream_executor::DeviceMemoryBase(ptr));
    allocator->DropRef();
    return status.ok() ? ncclSuccess : ncclInternalError;
  };
}

NcclPersistentPlanAllocator::~NcclPersistentPlanAllocator() = default;

absl::StatusOr<stream_executor::DeviceMemoryBase>
NcclPersistentPlanAllocator::AllocateAndInitialize(void* src, size_t size) {
  TF_ASSIGN_OR_RETURN(auto owned_mem,
                      allocator_->Allocate(device_ordinal_, size));
  stream_executor::DeviceMemoryBase mem = owned_mem.Release();
  stream_->ThenMemcpy(&mem, src, size);
  return mem;
}

absl::Status NcclPersistentPlanAllocator::Deallocate(
    stream_executor::DeviceMemoryBase mem) {
  return allocator_->Deallocate(device_ordinal_, mem);
}

ScopedNcclPersistentPlanAllocator::ScopedNcclPersistentPlanAllocator(
    NcclComm::Lock* comm, ncclPersistentPlanAllocator* allocator)
    : comm_(comm) {
  CHECK(ncclCommGetPersistentPlanAllocator(**comm_, &recover_) == ncclSuccess)
      << "Failed to get NCCL persistent plan allocator";

  CHECK(ncclCommSetPersistentPlanAllocator(**comm, allocator) == ncclSuccess)
      << "Faield to set NCCL persistent plan allocator";
}

ScopedNcclPersistentPlanAllocator::~ScopedNcclPersistentPlanAllocator() {
  CHECK(ncclCommSetPersistentPlanAllocator(**comm_, recover_) == ncclSuccess)
      << "Faield to set NCCL persistent plan allocator";
}
#endif
}  // namespace xla::gpu
