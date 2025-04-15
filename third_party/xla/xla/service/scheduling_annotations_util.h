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

#ifndef XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_
#define XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Returns the scheduling annotation id for the given instruction. If the
// instruction does not have a scheduling annotation, or the annotation is not
// an integer returns std::nullopt.
std::optional<int64_t> GetSchedulingAnnotation(
    const HloInstruction* instruction);

// Sets the scheduling annotation id for the given instruction.
void SetSchedulingAnnotation(HloInstruction* instruction, int64_t id);

// Returns the next available scheduling id for the given module. The next
// available id is the maximum scheduling id in the module plus one.
int64_t NextSchedulingId(const HloModule& module);

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_
