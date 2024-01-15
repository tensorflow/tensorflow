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

#ifndef XLA_SERVICE_GPU_NCCL_UTILS_H_
#define XLA_SERVICE_GPU_NCCL_UTILS_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique.h"
#include "xla/service/gpu/nccl_types.h"
#include "xla/service/gpu/thunk.h"
#include "xla/xla_data.pb.h"
#include "tsl/concurrency/ref_count.h"

namespace xla {
namespace gpu {

absl::StatusOr<NcclRedOp> ToNcclReduction(ReductionKind kind);

absl::StatusOr<std::pair<NcclDataType, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op);

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices);  // may be null

#ifdef PLATFORM_GOOGLE
//==-----------------------------------------------------------------------===//
// RAII helper to set NCCL persistent plan allocator.
//==-----------------------------------------------------------------------===//

class NcclPersistentPlanAllocator
    : public tsl::ReferenceCounted<NcclPersistentPlanAllocator> {
 public:
  NcclPersistentPlanAllocator(int64_t device_ordinal,
                              stream_executor::DeviceMemoryAllocator* allocator,
                              stream_executor::Stream* stream);
  ~NcclPersistentPlanAllocator();

  absl::StatusOr<stream_executor::DeviceMemoryBase> AllocateAndInitialize(
      void* src, size_t size);
  absl::Status Deallocate(stream_executor::DeviceMemoryBase mem);

  ncclPersistentPlanAllocator* nccl_allocator() { return &nccl_allocator_; }

 private:
  int64_t device_ordinal_;
  stream_executor::DeviceMemoryAllocator* allocator_;
  stream_executor::Stream* stream_;

  ncclPersistentPlanAllocator nccl_allocator_;
};

class ScopedNcclPersistentPlanAllocator {
 public:
  ScopedNcclPersistentPlanAllocator(
      NcclComm::Lock* comm, ncclPersistentPlanAllocator* nccl_allocator);
  ~ScopedNcclPersistentPlanAllocator();

 private:
  NcclComm::Lock* comm_;
  ncclPersistentPlanAllocator* recover_;
};
#endif
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NCCL_UTILS_H_
