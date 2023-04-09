/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_EXPENSIVE_INVARIANT_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_EXPENSIVE_INVARIANT_CODE_MOTION_H_

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass that rewrites while loops to hoist expensive and non-size-inflating
// groups of loop invariant instructions in the while body into the computation
// that contains the while instruction.
// Users can specify worth_hoisting_individually, and only the groups
// instructions with a root that returns true with it will be hoisted out.
class WhileLoopExpensiveInvariantCodeMotion : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;
  explicit WhileLoopExpensiveInvariantCodeMotion(
      HloPredicate worth_hoisting_individually,
      ShapeSizeFunction shape_size_function = ShapeUtil::ByteSizeOfElements)
      : shape_size_function_(std::move(shape_size_function)),
        worth_hoisting_individually_(std::move(worth_hoisting_individually)) {}
  ~WhileLoopExpensiveInvariantCodeMotion() override = default;

  absl::string_view name() const override {
    return "while-loop-expensive-invariant-code-motion";
  }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> TryHoistingInvariantInstructionsFromWhileBody(
      HloInstruction* while_instr);

  ShapeSizeFunction shape_size_function_;
  HloPredicate worth_hoisting_individually_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_EXPENSIVE_INVARIANT_CODE_MOTION_H_
