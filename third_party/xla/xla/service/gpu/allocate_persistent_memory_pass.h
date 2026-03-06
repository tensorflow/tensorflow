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

#ifndef XLA_SERVICE_GPU_ALLOCATE_PERSISTENT_MEMORY_PASS_H_
#define XLA_SERVICE_GPU_ALLOCATE_PERSISTENT_MEMORY_PASS_H_

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

class ThunkPassBufferAllocator;

// A pass that runs on the Thunk sequence and calls AllocatePersistentBuffers
// on each Thunk, allowing them to allocate their own long-lived memory.
class AllocatePersistentMemoryPass {
 public:
  static absl::Status Run(Thunk& thunk, ThunkPassBufferAllocator& allocator);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ALLOCATE_PERSISTENT_MEMORY_PASS_H_
