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

#ifndef XLA_PJRT_COMPILED_MEMORY_STATS_H_
#define XLA_PJRT_COMPILED_MEMORY_STATS_H_

#include <cstdint>
#include <string>

#include "absl/types/span.h"
#include "xla/pjrt/proto/executable_metadata.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

// Static memory usage for a compiled program.
// The on-device memory needed to run an executable is at least
//   generated_code_size_in_bytes
//   + argument_size_in_bytes + output_size_in_bytes - alias_size_in_bytes
//   + temp_size_in_bytes.
struct CompiledMemoryStats {
  // Device default memory (e.g., HBM for GPU/TPU) usage stats.
  int64_t generated_code_size_in_bytes = 0;
  int64_t argument_size_in_bytes = 0;
  int64_t output_size_in_bytes = 0;
  int64_t peak_memory_in_bytes = 0;
  int64_t total_allocation_bytes = 0;
  int64_t indefinite_allocations = 0;
  int64_t peak_unpadded_heap_bytes = 0;
  // How much argument is reused for output.
  int64_t alias_size_in_bytes = 0;
  int64_t temp_size_in_bytes = 0;
  int64_t total_size_in_bytes = 0;

  // Host memory usage stats.
  int64_t host_generated_code_size_in_bytes = 0;
  int64_t host_argument_size_in_bytes = 0;
  int64_t host_output_size_in_bytes = 0;
  int64_t host_alias_size_in_bytes = 0;
  int64_t host_temp_size_in_bytes = 0;

  std::string serialized_buffer_assignment;

  std::string DebugString() const;

  CompiledMemoryStatsProto ToProto() const;

  static CompiledMemoryStats FromProto(const CompiledMemoryStatsProto& proto);

  void PopulateBufferStatsFromAllocations(
      absl::Span<const BufferAllocation* const> allocs);
};

}  // namespace xla

#endif  // XLA_PJRT_COMPILED_MEMORY_STATS_H_
