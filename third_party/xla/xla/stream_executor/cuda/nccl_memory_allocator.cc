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

#include "xla/stream_executor/cuda/nccl_memory_allocator.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/nccl/nccl.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_memory_allocator.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"

namespace stream_executor::gpu {
namespace {

absl::StatusOr<void*> NcclAllocate(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activate = executor->Activate();

  void* ptr = nullptr;
  ncclResult_t res = ncclMemAlloc(&ptr, size);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from NCCL: %s. Last "
        "NCCL warning(error) log entry (may be unrelated): %s",
        tsl::strings::HumanReadableNumBytes(size), size,
        ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }
  XLA_VLOG_DEVICE(2, executor->device_ordinal())
      << "Allocated memory " << ptr << " of " << size << " bytes from NCCL";
  return ptr;
}

absl::Status NcclFree(StreamExecutor* executor, void* ptr, uint64_t size) {
  std::unique_ptr<ActivateContext> activate = executor->Activate();

  ncclResult_t res = ncclMemFree(ptr);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "Failed to free NCCL memory at %p; result: %s. Last "
        "NCCL warning(error) log entry (may be unrelated): %s",
        ptr, ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }

  XLA_VLOG_DEVICE(2, executor->device_ordinal())
      << "Freed NCCL memory " << ptr << " of " << size << " bytes";
  return absl::OkStatus();
}

// A memory allocated from NCCL on the given executor.
class NcclMemoryAllocation : public MemoryAllocation {
 public:
  NcclMemoryAllocation(StreamExecutor* executor, void* ptr, uint64_t size);

  ~NcclMemoryAllocation() final;
  DeviceAddressBase address() const final;

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t size_;
};

}  // namespace

NcclMemoryAllocation::NcclMemoryAllocation(StreamExecutor* executor, void* ptr,
                                           uint64_t size)
    : executor_(executor), ptr_(ptr), size_(size) {}

NcclMemoryAllocation::~NcclMemoryAllocation() {
  CHECK_OK(NcclFree(executor_, ptr_, size_));  // Crash OK
}

DeviceAddressBase NcclMemoryAllocation::address() const {
  return DeviceAddressBase(ptr_, size_);
}

NcclMemoryAllocator::NcclMemoryAllocator(StreamExecutor* executor)
    : executor_(executor) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> NcclMemoryAllocator::Allocate(
    uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, NcclAllocate(executor_, size));
  return std::make_unique<NcclMemoryAllocation>(executor_, ptr, size);
}

}  // namespace stream_executor::gpu

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    nccl_memory_allocator,
    stream_executor::gpu::RegisterCollectiveAllocatorFactory(
        stream_executor::gpu::CollectiveAllocatorType::kNccl,
        [](stream_executor::StreamExecutor* executor) {
          return std::make_unique<stream_executor::gpu::NcclMemoryAllocator>(
              executor);
        }));
