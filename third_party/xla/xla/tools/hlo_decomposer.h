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

#ifndef XLA_TOOLS_HLO_DECOMPOSER_H_
#define XLA_TOOLS_HLO_DECOMPOSER_H_

#include <memory>
#include <vector>

#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Decomposes the `module` into individual ops and de-duplicates the decomposed
// op if `deduplicate_modules` is true. The modules are considered duplicate if
// if their computation graphs are isomorphic (i.e. computations and
// instructions are sorted, names are ignored etc).
absl::StatusOr<std::vector<std::unique_ptr<HloModule>>> DecomposeHloModule(
    const HloModule& module, bool deduplicate_modules);

// Extracts an HLO instruction into a new HLO module replacing its operands
// with parameter instructions.
std::unique_ptr<HloModule> ExtractInstructionIntoNewModule(
    const HloInstruction& hlo);

// Extracts an HLO computation into a new HLO module, using its clone as the
// root computation.
std::unique_ptr<HloModule> ExtractComputationIntoNewModule(
    const HloComputation& computation);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_DECOMPOSER_H_
