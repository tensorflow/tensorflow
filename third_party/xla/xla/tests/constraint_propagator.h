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

enum class ConstantType { kUnknown, kZero, kOne };

// Return the constant type required by this computation, if known.
ConstantType GetInitValue(const HloComputation& computation);

class ConstraintPropagator {
 public:
  // Runs the constraint propagation on the entry computation of the
  // module. Returns a map from a parameter instruction to its computed
  // ConstraintState.
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

  absl::Status Propagate(const HloComputation* computation);

  absl::Status SeedConstraints(const HloComputation* computation);
  absl::Status PropagateSeedConstraints(const HloComputation* computation);
  absl::Status PropagateConstraints(const HloComputation* computation);
  absl::Status PropagateConstraintsExact(const HloInstruction* instruction);
  absl::Status PropagateConstraintsApprox(const HloInstruction* instruction);

  std::function<std::optional<uint64_t>(const HloInstruction*, int64_t)>
      get_index_known_zeroes_;

  absl::flat_hash_map<const HloInstruction*, ConstraintState> states_;
};

}  // namespace xla

#endif  // XLA_TESTS_CONSTRAINT_PROPAGATOR_H_
