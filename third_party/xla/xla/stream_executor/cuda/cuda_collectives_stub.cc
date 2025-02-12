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
#include "xla/stream_executor/cuda/cuda_collectives.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

/* static */ absl::StatusOr<void *> CudaCollectives::CollectiveMemoryAllocate(
    StreamExecutor *executor, uint64_t bytes) {
  if (bytes == 0) return nullptr;
  return absl::FailedPreconditionError("XLA was compiled without NCCL support");
}

/* static */ absl::Status CudaCollectives::CollectiveMemoryDeallocate(
    StreamExecutor *executor, void *location) {
  return absl::FailedPreconditionError("XLA was compiled without NCCL support");
}

}  // namespace stream_executor::gpu
