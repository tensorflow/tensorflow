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

#include "xla/backends/cpu/runtime/thunk_testlib.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::cpu {

BufferAllocation CreateBufferAllocation(size_t index, const Literal& literal) {
  size_t size_in_bytes = literal.size_bytes();
  return BufferAllocation(index, size_in_bytes, 0);
}

BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation) {
  return CreateBufferAllocationSlice(allocation, 0, allocation.size());
}

BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation, int64_t offset, int64_t size) {
  return BufferAllocation::Slice(&allocation, offset, size);
}

BufferAllocations CreateBufferAllocations(absl::Span<Literal*> literals) {
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(literals.size());

  for (auto* literal : literals) {
    size_t size_in_bytes = literal->size_bytes();
    buffers.emplace_back(literal->untyped_data(), size_in_bytes);
  }

  return BufferAllocations(buffers);
}

}  // namespace xla::cpu
