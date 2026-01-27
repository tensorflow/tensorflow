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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_PERMUTATION_SORT_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_PERMUTATION_SORT_EXPANDER_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/util.h"

namespace xla {

// Replaces key-value sorts where the key operand can be proved to contain
// unordered indices between 0 and sort dimension size - 1. Such patterns
// compute the inverse permutation, which can be done more efficiently using
// Scatter.
class PermutationSortExpander : public OpExpanderPass {
 public:
  explicit PermutationSortExpander(HloPredicate extra_filter = nullptr)
      : OpExpanderPass(std::move(extra_filter)) {}

  absl::string_view name() const override {
    return "permutation_sort_simplifier";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_PERMUTATION_SORT_EXPANDER_H_
