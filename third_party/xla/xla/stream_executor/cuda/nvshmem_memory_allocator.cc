/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/nvshmem_memory_allocator.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/nvshmem/nvshmem.h"   // IWYU pragma: keep
#include "third_party/nvshmem/nvshmemx.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cuda_memory_allocator.h"
#include "xla/stream_executor/cuda/nvshmem.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numbers.h"

namespace stream_executor::gpu {
namespace {

absl::StatusOr<void*> NvshmemAllocate(uint64_t size) {
  TF_RETURN_IF_ERROR(nvshmem::InitializeOnce());
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for NVSHMEM",
      tsl::strings::HumanReadableNumBytes(size), size);
  void* buffer = nvshmem_malloc(size);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from NVSHMEM memory",
        tsl::strings::HumanReadableNumBytes(size), size));
  }
  return buffer;
}

absl::Status NvshmemFree(void* ptr) {
  TF_RETURN_IF_ERROR(nvshmem::InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                ptr);
  nvshmem_free(ptr);
  return absl::OkStatus();
}

// A memory allocated from NVSHMEM on the given executor.
class NvshmemMemoryAllocation : public MemoryAllocation {
 public:
  NvshmemMemoryAllocation(void* ptr, uint64_t size);

  ~NvshmemMemoryAllocation() final;
  DeviceAddressBase address() const final;

 private:
  void* ptr_;
  uint64_t size_;
};

}  // namespace

NvshmemMemoryAllocation::NvshmemMemoryAllocation(void* ptr, uint64_t size)
    : ptr_(ptr), size_(size) {}

NvshmemMemoryAllocation::~NvshmemMemoryAllocation() {
  CHECK_OK(NvshmemFree(ptr_));  // Crash OK
}

DeviceAddressBase NvshmemMemoryAllocation::address() const {
  return DeviceAddressBase(ptr_, size_);
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
NvshmemMemoryAllocator::Allocate(uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, NvshmemAllocate(size));
  return std::make_unique<NvshmemMemoryAllocation>(ptr, size);
}

}  // namespace stream_executor::gpu

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    nvshmem_memory_allocator,
    stream_executor::gpu::RegisterCollectiveAllocatorFactory(
        stream_executor::gpu::CollectiveAllocatorType::kNvshmem,
        [](stream_executor::StreamExecutor* executor) {
          return std::make_unique<
              stream_executor::gpu::NvshmemMemoryAllocator>();
        }));
