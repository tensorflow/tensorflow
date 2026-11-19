/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Device memory allocator using ROCm/HIP Virtual Memory Management (VMM) APIs.
// Returned memory allocations are always mapped to a valid device address range
// and accessible from the device and its peers.
class RocmVmmAllocator : public MemoryAllocator {
 public:
  // Shared by all the ROCm VMM sites, not just this allocator.
  struct Options {
    // Request hipMemHandleTypePosixFileDescriptor rather than
    // hipMemHandleTypeNone, which permits no export mechanism. Memory the
    // collective layer may register has to be allocated this way. Defaults on,
    // as CudaDeviceAllocator::Options does; honored on ROCm 10.0 and newer, and
    // only where fd export really works.
    bool enable_posix_fd_handle = true;
  };

  // Options is a member of this class, so its default member initializer is
  // not available in a default argument here - hence the overload.
  explicit RocmVmmAllocator(StreamExecutor* executor);
  RocmVmmAllocator(StreamExecutor* executor, const Options& options);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

 private:
  StreamExecutor* executor_;
  Options options_;
};

struct RocmVmmAllocationParams {
  // Zero-initialized, so the fields GetRocmVmmAllocationParams() leaves alone
  // do not reach the driver as stack garbage.
  hipMemAllocationProp prop = {};

  // Allocation sizes and reservation alignments must be multiples of this.
  size_t granularity = 0;
};

// hipMemGetAllocationGranularity() takes the same properties hipMemCreate()
// does, so every VMM site goes through here rather than risk querying
// granularity with properties it does not allocate with.
//
// Exportable handles are best-effort: if the granularity query rejects them,
// prop.requestedHandleTypes falls back to hipMemHandleTypeNone.
//
// The caller must have activated `executor`'s context.
absl::StatusOr<RocmVmmAllocationParams> GetRocmVmmAllocationParams(
    StreamExecutor* executor, const RocmVmmAllocator::Options& options);

// Creates a physical VMM allocation of `padded_size` bytes with `prop`. Only
// hipMemCreate() knows whether the driver, container and device can really
// export handles, so a rejected request is retried with hipMemHandleTypeNone.
//
// The caller must have activated the context of the device `prop` names.
absl::StatusOr<hipMemGenericAllocationHandle_t> CreateRocmVmmPhysicalAllocation(
    hipMemAllocationProp prop, uint64_t padded_size);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_
