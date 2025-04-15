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

#ifndef XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
#define XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/compile_time_cap.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// HLO pass that rewrites while loops to hoist loop invariant instructions in
// the while body into the computation that contains the while instruction.

class WhileLoopInvariantCodeMotion : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;
  // If `hoist_constants` is true then constants are always hoisted out of while
  // loop bodies.  Otherwise they are only hoisted out if they enable other
  // non-trivial computations to be hoisted out.
  //
  // Setting `hoist_constants` to false can be help if LICM is run in the mid
  // level HLO pipeline because hoisting constants out of while loop bodies can
  // break optimizations like constant folding.
  //
  // Setting `hoist_other` and `hoist_reshapes` to false can be used to hoist
  // only constants. If provided, `hoist_size_inflation_ratio` will forbid
  // hoisting instructions where the ratio of the size of the output(s) to the
  // input(s) is larger than hoist_size_inflation_ratio. This is useful on
  // platforms on which it's important to prevent blow-ups in memory size.
  //
  // If `hoist_reshapes` is true, then reshapes are allowed to be hoisted out of
  // while loop body by themselves. Otherwise, they are only hoisted out if they
  // enable other non-trivial computations to be hoisted out.
  //
  // Setting `hoist_reshapes` to false can be useful when LICM is run in the
  // mid level HLO pipeline because the reshapes will often get fused with
  // consumer instructions, and won't cost anything if not hoisted. However,
  // any stand alone reshapes after fusion will benefit from hoisting.
  explicit WhileLoopInvariantCodeMotion(
      bool hoist_constants = false, bool hoist_reshapes = false,
      bool hoist_other = true,
      std::optional<float> hoist_size_inflation_ratio = std::nullopt,
      ShapeSizeFunction shape_size_function = ShapeUtil::ByteSizeOfElements)
      : hoist_constants_(hoist_constants),
        hoist_reshapes_(hoist_reshapes),
        hoist_other_(hoist_other),
        hoist_size_inflation_ratio_(hoist_size_inflation_ratio),
        shape_size_function_(shape_size_function) {}
  ~WhileLoopInvariantCodeMotion() override = default;

  absl::string_view name() const override {
    return "while-loop-invariant-code-motion";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool NotWorthHoistingIndividually(const HloInstruction& instruction);
  absl::StatusOr<bool> TryHoistingInvariantInstructionsFromWhileBody(
      HloInstruction* while_instr, BoundNonLinearCompilerAnalysis* allowance);

  bool hoist_constants_;
  bool hoist_reshapes_;
  bool hoist_other_;
  std::optional<float> hoist_size_inflation_ratio_;
  ShapeSizeFunction shape_size_function_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
