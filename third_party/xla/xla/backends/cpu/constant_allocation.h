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

#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_executable.h"

namespace xla::cpu {

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<CpuExecutable::ConstantAllocation>>
CreateConstantAllocations(const BufferAssignment& assignment);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_
