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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"

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

// Extracts HLO instructions into a new HLO module replacing all operands
// with parameter instructions even if the result of one instruction is used
// as a parameter to another. Combines results of all operations into the
// tuple and adds this tuple as a root instruction of the new module.
// Parameters:
//   instructions: HLO instructions to be extracted.
//   done_ops: Set of HLO opcodes that are done operations (e.g. AllReduceDone).
//   non_optimized_ops: Set of HLO opcodes that are not optimized (e.g.
//   AllReduce).
std::unique_ptr<HloModule> ExtractCollectiveOperationsIntoNewModule(
    const std::vector<HloInstruction*>& instructions,
    const absl::flat_hash_set<HloOpcode>& done_ops,
    const absl::flat_hash_set<HloOpcode>& non_optimized_ops);

// Extracts producer and consumer HLO instruction into a new HLO module
// replacing its operands with parameter instructions.
std::unique_ptr<HloModule> ExtractProducerConsumerIntoNewModule(
    const HloInstruction& producer, const HloInstruction& consumer);

// Extracts an HLO computation into a new HLO module, using its clone as the
// root computation.
std::unique_ptr<HloModule> ExtractComputationIntoNewModule(
    const HloComputation& computation);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_DECOMPOSER_H_
