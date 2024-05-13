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
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_driver.h"

namespace stream_executor::gpu {

absl::StatusOr<void*> GpuCollectives::CollectiveMemoryAllocate(
    GpuContext* context, uint64_t bytes) {
  return absl::UnimplementedError(
      "Feature not supported on ROCm platform (CollectiveMemoryAllocate)");
}

absl::Status GpuCollectives::CollectiveMemoryDeallocate(GpuContext* context,
                                                        void* location) {
  return absl::UnimplementedError(
      "Feature not supported on ROCm platform (CollectiveMemoryDeallocate)");
}

}  // namespace stream_executor::gpu
