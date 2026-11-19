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

#include "xla/stream_executor/rocm/rocm_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

// Distinguishes "these properties are not supported here", which is worth a
// retry with a weaker handle type, from a failure the retry cannot fix.
static bool IsUnsupportedPropertyError(hipError_t result) {
  return result == hipErrorNotSupported || result == hipErrorInvalidValue ||
         result == hipErrorOperatingSystem;
}

absl::StatusOr<RocmVmmAllocationParams> GetRocmVmmAllocationParams(
    StreamExecutor* executor, const RocmVmmAllocator::Options& options) {
  RocmVmmAllocationParams params;
  params.prop.type = hipMemAllocationTypePinned;
  params.prop.location.type = hipMemLocationTypeDevice;
  params.prop.location.id = executor->device_ordinal();
  params.prop.requestedHandleTypes = hipMemHandleTypeNone;

  if (options.enable_posix_fd_handle) {
    // ROCm before 10.0 cannot export and import these handles correctly. This
    // is the HIP version, numbered separately: ROCm 10.0 ships HIP 7.15.
    SemanticVersion hip_version =
        executor->GetDeviceDescription().runtime_version();
    if (hip_version >= SemanticVersion{7, 15, 0}) {
      params.prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "Exportable ROCm VMM handles require ROCm 10.0 or newer; HIP "
             "runtime is "
          << hip_version << ". Requesting non-exportable handles.";
    }
  }

  auto query_granularity = [&params]() {
    return ToStatus(
        hipMemGetAllocationGranularity(&params.granularity, &params.prop,
                                       hipMemAllocationGranularityRecommended),
        "hipMemGetAllocationGranularity");
  };

  absl::Status status = query_granularity();
  if (!status.ok() &&
      params.prop.requestedHandleTypes == hipMemHandleTypePosixFileDescriptor) {
    // runtime_version() describes the userspace runtime only: the driver, the
    // container or the device may still refuse fd export.
    LOG_FIRST_N(WARNING, 1)
        << "Granularity query for exportable ROCm VMM handles failed: "
        << status << ". Requesting non-exportable handles.";
    params.prop.requestedHandleTypes = hipMemHandleTypeNone;
    status = query_granularity();
  }
  ABSL_RETURN_IF_ERROR(status);
  return params;
}

absl::StatusOr<hipMemGenericAllocationHandle_t> CreateRocmVmmPhysicalAllocation(
    hipMemAllocationProp prop, uint64_t padded_size) {
  hipMemGenericAllocationHandle_t handle = nullptr;
  hipError_t result = hipMemCreate(&handle, padded_size, &prop, 0ULL);
  if (result == hipSuccess) {
    return handle;
  }
  if (prop.requestedHandleTypes != hipMemHandleTypePosixFileDescriptor ||
      !IsUnsupportedPropertyError(result)) {
    return ToStatus(result, "hipMemCreate");
  }

  // `padded_size` is a multiple of the exportable granularity, hence of the
  // non-exportable one too; were it not, the retry below reports the error.
  LOG_FIRST_N(WARNING, 1)
      << "hipMemCreate with exportable ROCm VMM handles failed: "
      << ToStatus(result)
      << ". Retrying with non-exportable handles; collectives sharing this "
         "memory will not work.";
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  ABSL_RETURN_IF_ERROR(ToStatus(hipMemCreate(&handle, padded_size, &prop, 0ULL),
                           "hipMemCreate"));
  return handle;
}

static hipMemAccessDesc GetVmmAccessDescriptor(int device) {
  hipMemAccessDesc descriptor = {};
  descriptor.location.type = hipMemLocationTypeDevice;
  descriptor.location.id = device;
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  return descriptor;
}

// Allocates device memory using HIP VMM APIs. Returns the mapped pointer,
// the padded size, and the allocation handle.
static absl::StatusOr<
    std::tuple<void*, uint64_t, hipMemGenericAllocationHandle_t>>
VmmAllocate(StreamExecutor* executor, uint64_t size,
            const RocmVmmAllocator::Options& options) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  ABSL_ASSIGN_OR_RETURN(RocmVmmAllocationParams params,
                   GetRocmVmmAllocationParams(executor, options));
  size_t granularity = params.granularity;

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  ABSL_ASSIGN_OR_RETURN(hipMemGenericAllocationHandle_t handle,
                   CreateRocmVmmPhysicalAllocation(params.prop, padded_size));

  hipDeviceptr_t ptr = nullptr;
  absl::Status status = ToStatus(
      hipMemAddressReserve(&ptr, padded_size, granularity, nullptr, 0ULL));
  if (!status.ok()) {
    ToStatus(hipMemRelease(handle)).IgnoreError();
    return status;
  }

  status = ToStatus(hipMemMap(ptr, padded_size, 0, handle, 0ULL));
  if (!status.ok()) {
    ToStatus(hipMemAddressFree(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemRelease(handle)).IgnoreError();
    return status;
  }

  // Cleanup on error: unmap + free VA + release physical memory.
  absl::Cleanup cleanup = [&]() {
    ToStatus(hipMemUnmap(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemAddressFree(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemRelease(handle)).IgnoreError();
  };

  VLOG(3) << "VMM allocated " << ptr << " requested size: " << size
          << " padded size: " << padded_size << " granularity: " << granularity
          << " device: " << executor->device_ordinal();

  // Set access for this device and all P2P-capable peers, matching the CUDA
  // pattern that gates on CanEnablePeerAccessTo().
  int device_count = 0;
  ABSL_RETURN_IF_ERROR(ToStatus(hipGetDeviceCount(&device_count)));
  for (int peer = 0; peer < device_count; peer++) {
    if (peer != executor->device_ordinal()) {
      auto peer_executor_or = const_cast<Platform*>(executor->GetPlatform())
                                  ->ExecutorForDevice(peer);
      if (!peer_executor_or.ok() ||
          !executor->CanEnablePeerAccessTo(peer_executor_or.value())) {
        continue;
      }
    }
    hipMemAccessDesc access_desc = GetVmmAccessDescriptor(peer);
    ABSL_RETURN_IF_ERROR(
        ToStatus(hipMemSetAccess(ptr, padded_size, &access_desc, 1)));
  }

  std::move(cleanup).Cancel();
  return std::make_tuple(ptr, padded_size, handle);
}

static void VmmDeallocate(StreamExecutor* executor, void* ptr,
                          uint64_t padded_size,
                          hipMemGenericAllocationHandle_t handle) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  VLOG(3) << "VMM deallocating " << ptr << " padded size: " << padded_size
          << " device: " << executor->device_ordinal();

  absl::Status status = ToStatus(hipMemUnmap(ptr, padded_size));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to unmap VMM memory at " << ptr << ": " << status;
  }
  status = ToStatus(hipMemRelease(handle));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to release VMM handle for " << ptr << ": " << status;
  }
  status = ToStatus(hipMemAddressFree(ptr, padded_size));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to free VMM address at " << ptr << ": " << status;
  }
}

namespace {

class RocmVmmMemoryAllocation final : public MemoryAllocation {
 public:
  RocmVmmMemoryAllocation(StreamExecutor* executor, void* ptr,
                          uint64_t requested_size, uint64_t padded_size,
                          hipMemGenericAllocationHandle_t handle)
      : executor_(executor),
        ptr_(ptr),
        requested_size_(requested_size),
        padded_size_(padded_size),
        handle_(handle) {}

  ~RocmVmmMemoryAllocation() final {
    if (ptr_ != nullptr) {
      VmmDeallocate(executor_, ptr_, padded_size_, handle_);
    }
  }

  DeviceAddressBase address() const final {
    return DeviceAddressBase(ptr_, requested_size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "RocmVmmMemoryAllocation[device=%d, ptr=%p, size=%d, padded_size=%d]",
        executor_->device_ordinal(), ptr_, requested_size_, padded_size_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t requested_size_;
  uint64_t padded_size_;
  hipMemGenericAllocationHandle_t handle_;
};

}  // namespace

RocmVmmAllocator::RocmVmmAllocator(StreamExecutor* executor)
    : RocmVmmAllocator(executor, Options()) {}

RocmVmmAllocator::RocmVmmAllocator(StreamExecutor* executor,
                                   const Options& options)
    : executor_(executor), options_(options) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> RocmVmmAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<RocmVmmMemoryAllocation>(executor_, nullptr, 0, 0,
                                                     nullptr);
  }

  ABSL_ASSIGN_OR_RETURN(auto result, VmmAllocate(executor_, size, options_));
  auto [ptr, padded_size, handle] = result;

  return std::make_unique<RocmVmmMemoryAllocation>(executor_, ptr, size,
                                                   padded_size, handle);
}

}  // namespace stream_executor::gpu
