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

#ifndef XLA_SERVICE_SCATTER_EXPANDER_H_
#define XLA_SERVICE_SCATTER_EXPANDER_H_

#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// This pass rewrites scatter operations into (roughly) while loops of
// dynamic-update-slices.
//
// This pass can be used in three ways:
//
//   - kEliminateAllScatters: For backends that don't support scatter, this pass
//     can convert every scatter into a loop.
//
//   - kEliminateSimpleScatters: For backends that *do* support scatter, this
//     pass can strength-reduce "simple" scatters -- specifically, scatters that
//     can be represented without a loop -- to dynamic-update-slices.
//
//   - kEliminateIndeterministicScatters: For backends that *do* support
//     scatter, this pass converts scatters with potentially indeterministic
//     behavior, because of non-unique indices or non-associative combiner
//     functions. There may be false positives, but no false negatives, i.e.
//     some scatters are converted even when deterministic in practice.
//
// Note that even in kEliminateSimpleScatters mode, this pass may still expand a
// scatter into a loop (with a trip-count of 1).  It's up to other
// simplification passes to remove the loop.
class ScatterExpander : public OpExpanderPass {
 public:
  enum Mode {
    kEliminateAllScatters,
    kEliminateSimpleScatters,
    kEliminateIndeterministicScatters,
  };

  explicit ScatterExpander(Mode m) : mode_(m) {}

  absl::string_view name() const override { return "scatter_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* inst) override;
  static absl::StatusOr<std::vector<HloInstruction*>> ScatterLoopBody(
      HloInstruction* scatter, HloInstruction* induction_var,
      absl::Span<HloInstruction* const> loop_state);

 private:
  Mode mode_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCATTER_EXPANDER_H_
