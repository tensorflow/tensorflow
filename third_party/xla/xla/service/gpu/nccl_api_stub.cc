/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "tsl/concurrency/ref_count.h"

namespace xla::gpu {

using PersistentPlanAllocator = NcclApi::PersistentPlanAllocator;
using ScopedPersistentPlanAllocator = NcclApi::ScopedPersistentPlanAllocator;

PersistentPlanAllocator::PersistentPlanAllocator(int64_t,
                                                 se::DeviceMemoryAllocator*,
                                                 se::Stream*) {
  // Suppress clang unused private field warnings.
  (void)device_ordinal_;
  (void)allocator_;
  (void)stream_;
}

PersistentPlanAllocator::~PersistentPlanAllocator() = default;

absl::StatusOr<se::DeviceMemoryBase>
PersistentPlanAllocator::AllocateAndInitialize(void*, size_t) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status PersistentPlanAllocator::Deallocate(se::DeviceMemoryBase mem) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

ScopedPersistentPlanAllocator::ScopedPersistentPlanAllocator(
    NcclCommHandle, tsl::RCReference<PersistentPlanAllocator>) {
  // Suppress clang unused private field warnings.
  (void)comm_;
  (void)recover_;
  (void)allocator_;
}

ScopedPersistentPlanAllocator::~ScopedPersistentPlanAllocator() = default;

absl::StatusOr<se::DeviceMemoryBase> NcclApi::Slice(se::DeviceMemoryBase,
                                                    PrimitiveType, size_t,
                                                    size_t) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<NcclCliqueId> NcclApi::GetUniqueId() {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<NcclCommHandle> NcclApi::CommInitRank(int32_t,
                                                     const NcclCliqueId&,
                                                     int32_t) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::CommAbort(NcclCommHandle) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<int32_t> NcclApi::CommCount(NcclCommHandle) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::CommGetAsyncError(NcclCommHandle) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::GroupStart() {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::GroupEnd() {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::AllReduce(se::DeviceMemoryBase, se::DeviceMemoryBase,
                                PrimitiveType, size_t, ReductionKind,
                                NcclCommHandle, se::Stream*) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::ReduceScatter(se::DeviceMemoryBase, se::DeviceMemoryBase,
                                    PrimitiveType, size_t, ReductionKind,
                                    NcclCommHandle, se::Stream*) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::AllGather(se::DeviceMemoryBase, se::DeviceMemoryBase,
                                PrimitiveType, size_t, NcclCommHandle,
                                se::Stream*) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::Send(se::DeviceMemoryBase, PrimitiveType, size_t, int32_t,
                           NcclCommHandle, se::Stream*) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::Recv(se::DeviceMemoryBase, PrimitiveType, size_t, int32_t,
                           NcclCommHandle, se::Stream*) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<NcclApi::NcclRegisteredBufferHandle> NcclApi::RegisterBuffer(
    NcclCommHandle, se::DeviceMemoryBase) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<NcclApi::NcclRegisteredBufferHandle> NcclApi::DeregisterBuffer(
    NcclCommHandle, NcclRegisteredBufferHandle) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

}  // namespace xla::gpu
