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

// This is a NCCL API stub that is linked into the process when XLA compiled
// without NCCL or CUDA support. It returns errors from all API calls. This stub
// makes it always safe to include NCCL API headers everywhere in XLA without
// #ifdefs or complex build rules magic. All magic handled by `:nccl_api`.

//==-----------------------------------------------------------------------===//
// NcclApi::PersistentPlanAllocator
//==-----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// NcclApiStub
//===----------------------------------------------------------------------===//

static absl::Status UnimplementedError() {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

class NcclApiStub final : public NcclApi {
 public:
  absl::StatusOr<NcclCliqueId> GetUniqueId() final {
    return UnimplementedError();
  }

  absl::StatusOr<NcclCommHandle> CommInitRank(int32_t nranks,
                                              const NcclCliqueId& clique_id,
                                              int32_t rank) final {
    return UnimplementedError();
  }

  absl::Status CommAbort(NcclCommHandle) final { return UnimplementedError(); }

  absl::StatusOr<int32_t> CommCount(NcclCommHandle) final {
    return UnimplementedError();
  }

  absl::Status CommGetAsyncError(NcclCommHandle) final {
    return UnimplementedError();
  }

  absl::Status GroupStart() final { return UnimplementedError(); }
  absl::Status GroupEnd() final { return UnimplementedError(); }

  absl::Status AllReduce(se::DeviceMemoryBase, se::DeviceMemoryBase,
                         PrimitiveType, size_t, ReductionKind, NcclCommHandle,
                         se::Stream*) final {
    return UnimplementedError();
  }

  absl::Status ReduceScatter(se::DeviceMemoryBase, se::DeviceMemoryBase,
                             PrimitiveType, size_t, ReductionKind,
                             NcclCommHandle, se::Stream*) final {
    return UnimplementedError();
  }

  absl::Status AllGather(se::DeviceMemoryBase, se::DeviceMemoryBase,
                         PrimitiveType, size_t, NcclCommHandle,
                         se::Stream*) final {
    return UnimplementedError();
  }

  absl::Status Send(se::DeviceMemoryBase, PrimitiveType, size_t, int32_t,
                    NcclCommHandle, se::Stream*) final {
    return UnimplementedError();
  }

  absl::Status Recv(se::DeviceMemoryBase, PrimitiveType, size_t, int32_t,
                    NcclCommHandle, se::Stream*) final {
    return UnimplementedError();
  }

  absl::StatusOr<NcclRegisteredBufferHandle> RegisterBuffer(
      NcclCommHandle, se::DeviceMemoryBase) final {
    return UnimplementedError();
  }

  absl::StatusOr<NcclRegisteredBufferHandle> DeregisterBuffer(
      NcclCommHandle, NcclRegisteredBufferHandle) final {
    return UnimplementedError();
  }
};

NcclApi* NcclApi::Default() {
  static auto* nccl_api = new NcclApiStub();
  return nccl_api;
}

}  // namespace xla::gpu
