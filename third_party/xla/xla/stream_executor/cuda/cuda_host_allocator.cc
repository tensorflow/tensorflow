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

#include "xla/stream_executor/cuda/cuda_host_allocator.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/numbers.h"

namespace stream_executor::gpu {

// Allocates pinned host memory. For NUMA-aware allocation, uses NUMAMalloc
// and registers the memory with CUDA. Otherwise uses cuMemHostAlloc.
static absl::StatusOr<void*> HostAllocate(StreamExecutor* executor,
                                          int32_t numa_node, uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    // CUDA programming guide: "Any address of a variable ... returned by one
    // of the memory allocation routines from the driver ... API is always
    // aligned to at least 256 bytes."
    auto* buffer =
        tsl::port::NUMAMalloc(numa_node, size, /*minimum_alignment=*/256);
    if (buffer == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "%sFailed to allocate %s (%llu bytes) of host memory "
          "pinned to NUMA node %d",
          xla::XlaFormatDevice(executor->device_ordinal()),
          tsl::strings::HumanReadableNumBytes(size), size, numa_node));
    }

    std::unique_ptr<ActivateContext> activate = executor->Activate();
    absl::Status status = cuda::ToStatus(
        cuMemHostRegister(buffer, size, CU_MEMHOSTREGISTER_PORTABLE));
    if (!status.ok()) {
      tsl::port::NUMAFree(buffer, size);
      return absl::InternalError(absl::StrFormat(
          "%sFailed to register %s (%llu bytes) of host memory: %s",
          xla::XlaFormatDevice(executor->device_ordinal()),
          tsl::strings::HumanReadableNumBytes(size), size, status.message()));
    }

    XLA_VLOG_DEVICE(2, executor->device_ordinal())
        << "Allocated NUMA host memory " << buffer << " of " << size
        << " bytes on node " << numa_node;
    return buffer;
  }

  std::unique_ptr<ActivateContext> activate = executor->Activate();
  void* buffer = nullptr;
  // "Portable" memory is visible to all CUDA contexts.
  absl::Status status =
      cuda::ToStatus(cuMemHostAlloc(&buffer, size, CU_MEMHOSTALLOC_PORTABLE));
  if (!status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "%sFailed to allocate %s (%llu bytes) of pinned host memory: %s",
        xla::XlaFormatDevice(executor->device_ordinal()),
        tsl::strings::HumanReadableNumBytes(size), size, status.message()));
  }

  XLA_VLOG_DEVICE(2, executor->device_ordinal())
      << "Allocated pinned host memory " << buffer << " of " << size
      << " bytes";
  return buffer;
}

// Frees pinned host memory previously allocated by HostAllocate.
static void HostDeallocate(StreamExecutor* executor, int32_t numa_node,
                           void* ptr, uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    if (size > 0) {
      std::unique_ptr<ActivateContext> activate = executor->Activate();
      absl::Status status = cuda::ToStatus(cuMemHostUnregister(ptr));
      if (!status.ok()) {
        XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
            << "Failed to unregister host memory at " << ptr << ": " << status;
      }
    }
    tsl::port::NUMAFree(ptr, size);
  } else {
    std::unique_ptr<ActivateContext> activate = executor->Activate();
    absl::Status status = cuda::ToStatus(cuMemFreeHost(ptr));
    if (!status.ok()) {
      XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
          << "Failed to free host memory at " << ptr << ": " << status;
    }
  }
  XLA_VLOG_DEVICE(2, executor->device_ordinal())
      << "Freed host memory " << ptr << " of " << size << " bytes";
}

namespace {

// A pinned host memory allocation backed by cuMemHostAlloc or NUMAMalloc.
class CudaHostMemoryAllocation : public MemoryAllocation {
 public:
  CudaHostMemoryAllocation(StreamExecutor* executor, void* ptr, uint64_t size,
                           int32_t numa_node)
      : executor_(executor), ptr_(ptr), size_(size), numa_node_(numa_node) {}

  ~CudaHostMemoryAllocation() final {
    if (ptr_ != nullptr) {
      HostDeallocate(executor_, numa_node_, ptr_, size_);
    }
  }

  DeviceAddressBase address() const final {
    return DeviceAddressBase(ptr_, size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "CudaHostMemoryAllocation[device=%d, numa=%d, ptr=%p, size=%d]",
        executor_->device_ordinal(), numa_node_, ptr_, size_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t size_;
  int32_t numa_node_;
};

}  // namespace

CudaHostAllocator::CudaHostAllocator(StreamExecutor* executor,
                                     int32_t numa_node)
    : executor_(executor), numa_node_(numa_node) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> CudaHostAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<CudaHostMemoryAllocation>(executor_, nullptr, 0,
                                                      numa_node_);
  }

  auto result = HostAllocate(executor_, numa_node_, size);
  if (!result.ok()) return result.status();

  return std::make_unique<CudaHostMemoryAllocation>(executor_, *result, size,
                                                    numa_node_);
}

}  // namespace stream_executor::gpu
