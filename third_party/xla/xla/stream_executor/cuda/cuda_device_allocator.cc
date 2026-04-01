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

#include "xla/stream_executor/cuda/cuda_device_allocator.h"

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
#include "tsl/platform/numbers.h"

namespace stream_executor::gpu {

// Allocates device memory using cuMemAlloc.
static absl::StatusOr<void*> DeviceAllocate(StreamExecutor* executor,
                                            uint64_t size) {
  std::unique_ptr<ActivateContext> activate = executor->Activate();
  CUdeviceptr result = 0;
  absl::Status status = cuda::ToStatus(cuMemAlloc(&result, size));
  if (!status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "%sFailed to allocate %s (%llu bytes) of device memory: %s",
        xla::XlaFormatDevice(executor->device_ordinal()),
        tsl::strings::HumanReadableNumBytes(size), size, status.message()));
  }
  void* ptr = reinterpret_cast<void*>(result);
  XLA_VLOG_DEVICE(2, executor->device_ordinal())
      << "Allocated device memory " << ptr << " of " << size << " bytes";
  return ptr;
}

// Frees device memory previously allocated by DeviceAllocate.
static void DeviceDeallocate(StreamExecutor* executor, void* ptr,
                             uint64_t size) {
  std::unique_ptr<ActivateContext> activate = executor->Activate();
  CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(ptr);
  absl::Status status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
        << "Failed to free device memory at " << ptr << ": " << status;
  } else {
    XLA_VLOG_DEVICE(2, executor->device_ordinal())
        << "Freed device memory " << ptr << " of " << size << " bytes";
  }
}

namespace {

// A device memory allocation backed by cuMemAlloc.
class CudaDeviceMemoryAllocation : public MemoryAllocation {
 public:
  CudaDeviceMemoryAllocation(StreamExecutor* executor, void* ptr, uint64_t size)
      : executor_(executor), ptr_(ptr), size_(size) {}

  ~CudaDeviceMemoryAllocation() final {
    if (ptr_ != nullptr) {
      DeviceDeallocate(executor_, ptr_, size_);
    }
  }

  DeviceAddressBase address() const final {
    return DeviceAddressBase(ptr_, size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "CudaDeviceMemoryAllocation[device=%d, ptr=%p, size=%d]",
        executor_->device_ordinal(), ptr_, size_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t size_;
};

}  // namespace

CudaDeviceAllocator::CudaDeviceAllocator(StreamExecutor* executor)
    : executor_(executor) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> CudaDeviceAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<CudaDeviceMemoryAllocation>(executor_, nullptr, 0);
  }

  auto result = DeviceAllocate(executor_, size);
  if (!result.ok()) return result.status();

  return std::make_unique<CudaDeviceMemoryAllocation>(executor_, *result, size);
}

}  // namespace stream_executor::gpu
