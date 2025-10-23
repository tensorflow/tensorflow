/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_
#define XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Creates and returns a list of `BufferAllocationInfo` instances containing
// relevant information from `buffer_assignment`.
std::vector<BufferAllocationInfo> CreateBufferAllocationInfos(
    const HloModule& module, const BufferAssignment& buffer_assignment);

// Creates and returns a table containing the mapping from entry computation
// parameters to buffer allocation indices:
//
//   vector[parameter_number] == allocation.index()
//   vector[result_number]    == allocation.index()
//
std::vector<int32_t> CreateArgIndexTable(
    absl::Span<const BufferAllocationInfo> allocations);
std::vector<int32_t> CreateResultIndexTable(
    absl::Span<const BufferAllocationInfo> allocations);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_UTIL_H_
