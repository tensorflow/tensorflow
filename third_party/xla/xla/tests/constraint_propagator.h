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

#ifndef XLA_TESTS_CONSTRAINT_PROPAGATOR_H_
#define XLA_TESTS_CONSTRAINT_PROPAGATOR_H_

#include <cstdint>
#include <functional>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/constraint_state.h"

namespace xla {

enum class IdentityElementType { kUnknown, kZero, kOne };

// Returns the identity element type for the given reduction computation.
// Add => 0, Mul => 1, etc.. Returns kUnknown otherwise.
IdentityElementType GetReductionIdentityElementType(
    const HloComputation& computation);

// ConstraintPropagator implements a reverse dataflow analysis to determine
// valid input ranges for HLO instructions, ensuring execution without
// undefined behavior such as NaNs or Infs.
//
// The propagator follows a "Reverse Constraint Propagation" mechanism: it
// starts from the HLO graph's roots (outputs) and traverses the instructions
// in reverse topological order (use-before-def), propagating requirements
// backward to the parameters.
//
// Key Principles:
// - Soundness over Completeness: The goal is to identify a subset of the
//   input space that guarantees stable execution, rather than finding every
//   possible valid input.
// - Interval Arithmetic: Constraints are modeled as independent intervals
//   (min, max, and zero-exclusion) for each tensor use. Relational expressions
//   between operands are simplified into independent constraints to keep
//   computation scalable.
//
// Limitations:
// - Does not handle control flow.
// - Uses weak heuristics for accumulation-intensive operations.
class ConstraintPropagator {
 public:
  static absl::StatusOr<
      absl::flat_hash_map<const HloInstruction*, ConstraintState>>
  Run(const HloModule& module,
      std::function<std::optional<uint64_t>(const HloInstruction*, int64_t)>
          get_index_known_zeroes = nullptr);

 private:
  explicit ConstraintPropagator(
      std::function<std::optional<uint64_t>(const HloInstruction*, int64_t)>
          get_index_known_zeroes)
      : get_index_known_zeroes_(get_index_known_zeroes) {}

  // Propagates constraints in post-order throughout the given computation.
  absl::Status Propagate(const HloComputation* computation);

  // Seeds constraints from ops like sqrt, log, etc on their operands. eg.
  // sqrt(x) => x >= 0
  absl::Status SeedConstraints(const HloComputation* computation);

  // Propagates constraints from the seed constraints backwards with exact
  // propagation to avoid introducing approximations.
  absl::Status PropagateSeedConstraints(const HloComputation* computation);

  // Propagates constraints backwards with weak heuristics to make it
  // computationally tractable.
  absl::Status PropagateConstraints(const HloComputation* computation);

  // Propagates constraints from the output of an instruction to its operands.
  // This is exact and does not introduce any approximations as ops like data
  // formatting can simply propagate the exact constraints to their operands.
  absl::Status PropagateConstraintsExact(const HloInstruction* instruction);

  // Propagates constraints from the output of an instruction to its operands.
  // This is approximate and introduces approximations for ops like add, sub,
  // etc. These approximations can reduce the valid search space for an input
  // and also result in empty constraints.
  absl::Status PropagateConstraintsApprox(const HloInstruction* instruction);

  // Function that extracts known zeroes bitmask for the given dimension of a
  // DynamicSlice or DynamicUpdateSlice instruction.
  std::function<std::optional<uint64_t>(const HloInstruction*, int64_t)>
      get_index_known_zeroes_;

  // Constraint states for each instruction in the module.
  absl::flat_hash_map<const HloInstruction*, ConstraintState> states_;
};

}  // namespace xla

#endif  // XLA_TESTS_CONSTRAINT_PROPAGATOR_H_
