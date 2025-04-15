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

#include "xla/stream_executor/cuda/cuda_collectives.h"

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

absl::StatusOr<xla::gpu::GpuCollectives*> GetGpuCollectives(
    StreamExecutor* executor) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  return tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);
}

/* static */ absl::StatusOr<void*> CudaCollectives::CollectiveMemoryAllocate(
    StreamExecutor* executor, uint64_t bytes) {
  if (bytes == 0) return nullptr;

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Allocate(bytes);
}

/* static */ absl::Status CudaCollectives::CollectiveMemoryDeallocate(
    StreamExecutor* executor, void* location) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Deallocate(location);
}

}  // namespace stream_executor::gpu
