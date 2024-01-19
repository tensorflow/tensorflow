/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"

#ifdef STREAM_EXECUTOR_GPU_ENABLE_XCCL
#include "third_party/nccl/nccl.h"
#endif  // STREAM_EXECUTOR_GPU_ENABLE_XCCL

namespace stream_executor::gpu {

/* static */ absl::StatusOr<void*> GpuCollectives::CollectiveMemoryAllocate(
    GpuContext* context, uint64_t bytes) {
  if (bytes == 0) return nullptr;

  ScopedActivateContext activated(context);

#ifdef STREAM_EXECUTOR_GPU_ENABLE_XCCL
  void* ptr = nullptr;
  ncclResult_t res = ncclMemAlloc(&ptr, bytes);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to allocate %s (%llu bytes) from device collective memory: %s, "
        "Last NCCL warning(error) log entry (may be unrelated): %s",
        tsl::strings::HumanReadableNumBytes(bytes), bytes,
        ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }
  VLOG(2) << "Allocated collective memory " << ptr << " for context "
          << context->context() << " of " << bytes << " bytes";
  return ptr;
#else
  return absl::FailedPreconditionError("XLA was compiled without NCCL support");
#endif
}

/* static */ absl::Status GpuCollectives::CollectiveMemoryDeallocate(
    GpuContext* context, void* location) {
  ScopedActivateContext activation(context);

#ifdef STREAM_EXECUTOR_GPU_ENABLE_XCCL
  ncclResult_t res = ncclMemFree(location);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to free device collective memory at %p; result: %s, Last NCCL "
        "warning(error) log entry (may be unrelated): %s",
        location, ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }

  VLOG(2) << "Deallocated collective memory " << location << " for context "
          << context->context();
  return absl::OkStatus();
#else
  return absl::FailedPreconditionError("XLA was compiled without NCCL support");
#endif
}

}  // namespace stream_executor::gpu
