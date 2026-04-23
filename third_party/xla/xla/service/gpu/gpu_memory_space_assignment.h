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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Frontend attribute names for specifying memory spaces on custom call
// operands and results. The format is {index:memory_space,...}, e.g.
// "{0:1,2:1}" means operand/result 0 and 2 should be in memory space 1.
inline constexpr absl::string_view kOperandsMemorySpacesAttr =
    "operands_memory_spaces";
inline constexpr absl::string_view kResultsMemorySpacesAttr =
    "results_memory_spaces";

enum class MemorySpaceColor {
  // Corresponds to stream_executor::MemoryTypes::kDefault or kUnified.
  // This memory can be allocated with any device allocation API.
  kDefault = 0,

  // Corresponds to stream_executor::MemoryTypes::kCollective. This memory
  // should be compatible with symmetric memory requirements.
  kCollective = 1,

  // Temp buffers can be allocated within separate memory space (if
  // xla_gpu_temp_buffer_use_separate_color is set). This improves cuda-graphs
  // performance. See more details in the corresponding flag description.
  kTempBuffer = 2,
};

// Converts an integer to a MemorySpaceColor, returning an error if the value
// does not correspond to a known color.
absl::StatusOr<MemorySpaceColor> AsMemorySpaceColor(int64_t memory_space);

// Parses a string of the form "{index:memory_space,...}" into a vector of
// (index, memory_space) pairs. For example, "{0:1,2:1}" means index 0 and 2
// should be in memory space 1.
absl::StatusOr<std::vector<std::pair<int64_t, MemorySpaceColor>>>
ParseIndexMemorySpacePairs(absl::string_view str);

// Creates a buffer colorer that assigns memory space colors to HLO values
// during buffer assignment. It handles:
//  - Collective operations (all-reduce, all-gather, etc.) → kCollective
//  - Mosaic with NVSHMEM/multimem → kCollective
//  - Custom call `operands_memory_spaces` / `results_memory_spaces` frontend
//    attributes → requested memory space
//  - Everything else → kDefault
BufferAssigner::Colorer CreateColorer(const DebugOptions& option);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
