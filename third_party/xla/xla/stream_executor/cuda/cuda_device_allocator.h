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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// CUDA device memory allocator. The current implementation uses CUDA Virtual
// Memory Management (VMM) APIs (cuMemCreate/cuMemAddressReserve/cuMemMap).
class CudaDeviceAllocator : public MemoryAllocator {
 public:
  struct Options {
    // Minimum alignment for allocations. The actual alignment is the maximum
    // of this value and the device-reported allocation granularity.
    size_t alignment = 4096;

    // Whether to enable peer access from all accessible devices.
    bool enable_peer_access = false;

    // Whether to request POSIX_FILE_DESCRIPTOR handle type.
    bool enable_posix_fd_handle = true;

    // Whether to request FABRIC handle type.
    bool enable_fabric_handle = false;

    // Whether to mark allocations as GPUDirect RDMA capable.
    bool enable_rdma = false;

    // Whether to use CUDA Virtual Memory Management (VMM) APIs. If false,
    // falls back to legacy cuMemAlloc / cuMemFree APIs.
    bool use_vmm = true;
  };

  explicit CudaDeviceAllocator(StreamExecutor* executor);
  CudaDeviceAllocator(StreamExecutor* executor, Options options);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

  const Options& options() const { return options_; }

  static void EnterStreamCapture(StreamExecutor* executor);
  static void ExitStreamCapture(StreamExecutor* executor);

 private:
  StreamExecutor* executor_;
  Options options_;
};

class CudaDeviceMemoryAllocation : public MemoryAllocation {
 public:
  CudaDeviceMemoryAllocation(StreamExecutor* executor, void* ptr,
                             uint64_t requested_size, uint64_t padded_size,
                             CUmemGenericAllocationHandle handle);

  ~CudaDeviceMemoryAllocation() final;

  DeviceAddressBase address() const final;

  std::string ToString() const final;

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t requested_size_;
  uint64_t padded_size_;
  CUmemGenericAllocationHandle handle_;
};

CUmemAllocationProp BuildVmmAllocationProp(
    CUdevice device, const CudaDeviceAllocator::Options& options);

// Probes device allocator options supported by CUDA VMM; falls back through
// simpler handle-type combinations if the device rejects the strongest one.
absl::StatusOr<CudaDeviceAllocator::Options> QueryDeviceAllocatorOptions(
    CUdevice device);

// The handle types the driver accepted for a granularity query, and the
// recommended mapping granularity for them.
struct VmmGranularityProbe {
  CudaDeviceAllocator::Options options;
  size_t granularity = 0;
};

// Queries the recommended VMM allocation granularity for `options`. Drivers can
// reject the query itself for unsupported handle types, so this falls back the
// same way CreateVmmPhysicalAllocation does (FABRIC+POSIX_FD -> POSIX_FD ->
// NONE) and returns `options` with the handle types that were accepted. The
// caller must have activated the device context.
absl::StatusOr<VmmGranularityProbe> ProbeVmmGranularity(
    CUdevice device, CudaDeviceAllocator::Options options);

// Creates a physical VMM allocation of `padded_size` bytes with cuMemCreate,
// falling back through simpler handle types (FABRIC+POSIX_FD -> POSIX_FD ->
// NONE) when the driver reports NOT_PERMITTED, NOT_SUPPORTED or INVALID_VALUE.
// The returned handle may therefore carry fewer handle types than `properties`
// requested; callers that computed `padded_size` from the original properties
// accept that. The caller must have activated the device context.
absl::StatusOr<CUmemGenericAllocationHandle> CreateVmmPhysicalAllocation(
    CUmemAllocationProp properties, uint64_t padded_size);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_
