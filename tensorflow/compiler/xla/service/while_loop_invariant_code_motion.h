/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass that rewrites while loops to hoist loop invariant instructions in
// the while body into the computation that contains the while instruction.

class WhileLoopInvariantCodeMotion : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64(const Shape&)>;
  // If `hoist_constants` is true then constants are always hoisted out of while
  // loop bodies.  Otherwise they are only hoisted out if they enable other
  // non-trivial computations to be hoisted out.
  //
  // Setting `hoist_constants` to false can be help if LICM is run in the mid
  // level HLO pipeline because hoisting constants out of while loop bodies can
  // break optimizations like constant folding.
  // Setting `hoist_non_constants` to false can be used to hoist only constants.
  // If provided, `hoist_size_inflation_ratio` will forbid hoisting instructions
  // where the ratio of the size of the output(s) to the input(s) is larger than
  // hoist_size_inflation_ratio. This is useful on platforms on which it's
  // important to prevent blow-ups in memory size.
  explicit WhileLoopInvariantCodeMotion(
      bool hoist_constants = false, bool hoist_non_constants = true,
      absl::optional<float> hoist_size_inflation_ratio = absl::nullopt,
      ShapeSizeFunction shape_size_function = ShapeUtil::ByteSizeOfElements)
      : hoist_constants_(hoist_constants),
        hoist_non_constants_(hoist_non_constants),
        hoist_size_inflation_ratio_(hoist_size_inflation_ratio),
        shape_size_function_(shape_size_function) {}
  ~WhileLoopInvariantCodeMotion() override = default;

  absl::string_view name() const override {
    return "while-loop-invariant-code-motion";
  }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool NotWorthHoistingIndividually(const HloInstruction& instruction);
  StatusOr<bool> TryHoistingInvariantInstructionsFromWhileBody(
      HloInstruction* while_instr);

  bool hoist_constants_;
  bool hoist_non_constants_;
  absl::optional<float> hoist_size_inflation_ratio_;
  ShapeSizeFunction shape_size_function_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
