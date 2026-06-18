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

#ifndef XLA_SERVICE_DEBUG_UNSTABLE_REDUCTION_FINDER_H_
#define XLA_SERVICE_DEBUG_UNSTABLE_REDUCTION_FINDER_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Find all unstable reduction instructions in the given module.
//
// We define an unstable reduction instruction as a reduce instruction that
// accumulates something other than a maximum or a minimum, and whose
// accumulation type is a floating point type smaller than f32.
std::vector<const HloInstruction*> FindUnstableReductionInstructions(
    const HloModule* module);

}  // namespace xla

#endif  // XLA_SERVICE_DEBUG_UNSTABLE_REDUCTION_FINDER_H_
