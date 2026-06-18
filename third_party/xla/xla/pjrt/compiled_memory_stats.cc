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

#include "xla/pjrt/compiled_memory_stats.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_value.h"

namespace xla {

// Recomputes the memory stats from allocations. Why recompute?
// Firstly, there are cases in which gpu::Executable inherits its allocations
// from elsewhere, and no buffer assignment is available.
// Secondly, exec->buffer_assignment()->GetStats() provides the statistics we
// want, but does not distinguish between device and host memory, and does
// not account for aliased memory.
void CompiledMemoryStats::PopulateBufferStatsFromAllocations(
    absl::Span<const BufferAllocation* const> allocs) {
  argument_size_in_bytes = 0;
  output_size_in_bytes = 0;
  temp_size_in_bytes = 0;
  alias_size_in_bytes = 0;
  host_argument_size_in_bytes = 0;
  host_output_size_in_bytes = 0;
  host_temp_size_in_bytes = 0;
  host_alias_size_in_bytes = 0;

  for (const BufferAllocation* alloc : allocs) {
    // Backends may use BufferAssigner::CanUseAllocation to deliberately pack
    // multiple compatible colors into a single allocation. CompiledMemoryStats
    // only separates host from device memory, so do not require exact color
    // equality here. We still sanity-check that an allocation does not mix host
    // and device assignments.
    int64_t alloc_memory_kind = -1;
    for (const auto& [value, _] : alloc->assigned_buffers()) {
      const HloPosition& defining_position = value->defining_position();
      int64_t memory_space = Layout::kDefaultMemorySpace;
      if (defining_position.shape().has_layout()) {
        memory_space = defining_position.shape().layout().memory_space();
      }
      int64_t memory_kind = memory_space == Layout::kHostMemorySpace
                                ? Layout::kHostMemorySpace
                                : Layout::kDefaultMemorySpace;
      if (alloc_memory_kind == -1) {
        alloc_memory_kind = memory_kind;
      } else {
        CHECK(alloc_memory_kind == memory_kind &&
              "expected same host/device memory kind for all assignments in "
              "allocation");
      }
    }
    if (alloc_memory_kind == -1) {
      // But if assignments are not available, then we have to assume that the
      // default coloring behavior was used.
      alloc_memory_kind = alloc->color() == Layout::kHostMemorySpace
                              ? Layout::kHostMemorySpace
                              : Layout::kDefaultMemorySpace;
    }

    bool is_host = alloc_memory_kind == Layout::kHostMemorySpace;
    int64_t size = alloc->size();
    if (alloc->is_entry_computation_parameter()) {
      if (is_host) {
        host_argument_size_in_bytes += size;
      } else {
        argument_size_in_bytes += size;
      }
      if (alloc->is_parameter_aliased_with_output()) {
        if (is_host) {
          host_alias_size_in_bytes += size;
        } else {
          alias_size_in_bytes += size;
        }
      }
    }
    if (alloc->maybe_live_out()) {
      if (is_host) {
        host_output_size_in_bytes += size;
      } else {
        output_size_in_bytes += size;
      }
    }
    if (alloc->IsPreallocatedTempBuffer()) {
      if (is_host) {
        host_temp_size_in_bytes += size;
      } else {
        temp_size_in_bytes += size;
      }
    }
  }
}

}  // namespace xla
