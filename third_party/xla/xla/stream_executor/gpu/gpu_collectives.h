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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COLLECTIVES_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COLLECTIVES_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/context.h"

namespace stream_executor::gpu {

struct GpuCollectives {
  // Allocates a collective device memory space of size bytes associated with
  // the given context.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclmemalloc
  static absl::StatusOr<void*> CollectiveMemoryAllocate(Context* context,
                                                        uint64_t bytes);

  // Deallocates a collective device memory space of size bytes associated with
  // the given context.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclmemfree
  static absl::Status CollectiveMemoryDeallocate(Context* context,
                                                 void* location);
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COLLECTIVES_H_
