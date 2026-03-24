/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

enum class MemorySpaceColor {
  // Corresponds to stream_executor::MemorySpace::kDefault or kUnified.
  // This memory can be allocated with any device allocation API.
  kDefault = 0,

  // Corresponds to stream_executor::MemorySpace::kCollective.
  // This memory should be allocated with ncclMemAlloc in the runtime.
  kCollective = 1,

  // Temp buffers can be allocated within separate memory space (if
  // xla_gpu_temp_buffer_use_separate_color is set). This improves cuda-graphs
  // performance. See more details in the corresponding flag description.
  kTempBuffer = 2,

  // Corresponds to stream_executor::MemorySpace::kUnified.
  // This memory should be allocated in a CPU/GPU unified memory space.
  kUnified = 3,
};

BufferAssigner::Colorer CreateColorer(const DebugOptions& option);
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
