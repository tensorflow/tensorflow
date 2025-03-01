/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_
#define XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::cpu {

// A storage (or an alias) for constant allocations data.
struct ConstantAllocation {
  se::DeviceMemoryBase AsDeviceMemoryBase() const;

  BufferAllocation::Index index = -1;
  std::variant<std::monostate, std::unique_ptr<Literal>,
               absl::Span<const uint8_t>>
      data;
};

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<ConstantAllocation>> CreateConstantAllocations(
    const BufferAssignment& assignment);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_
