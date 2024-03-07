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

#ifndef XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_ALLOCATIONS_H_
#define XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_ALLOCATIONS_H_

#include "absl/container/flat_hash_map.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::gpu {

// Command buffer allocations tracks external buffer allocations done via the
// CommandBuffer API and owned by the XLA executable (via instantiated command
// buffers and memory allocation Gpu graph nodes).
class CommandBufferAllocations : public BufferAllocations::ExternalAllocations {
 public:
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Index index) const override;

  // Adds an external allocation for a given buffer index. Returns error if
  // allocation already exists.
  absl::Status AddAllocation(BufferAllocation::Index index,
                             se::DeviceMemoryBase memory) override;

  // Erases an external allocation for a given buffer index. Returns error if
  // allocation does not exists.
  absl::Status EraseAllocation(BufferAllocation::Index index) override;

 private:
  absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase> allocs_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_ALLOCATIONS_H_
