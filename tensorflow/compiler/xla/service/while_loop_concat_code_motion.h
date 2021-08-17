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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONCAT_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONCAT_CODE_MOTION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// A pass that tries to lift concatenation out of a while loop, and replace
// piece-wise subcomputations in the loop body with one on the concatenated
// shape.
//
// For example:
//
// loop = while (a, b, c, d) {
//   e = concat(a, b)
//   f = some-op(e) <with the same shape as e>
//   s0 = slice(f) first half
//   s1 = slice(f) second half
//   a_1 = add(a, s0)
//   b_1 = add(b, s1)
//   a_new = add(a_1, c)
//   b_new = add(b_1, d)
//   c_new = add(a_new, c)
//   d_new = add(b_new, d)
//   ROOT tuple(a_new, b_new, c_new, d_new)
// }
//
// will be transformed to
//
// ab = concat(a, b)
// cd = concat(c, d)
// while (ab, cd) {
//   f = some-op(ab)
//   ab_1 = add(ab, f)
//   ab_new = add(ab_1, cd)
//   cd_new = add(ab_new, cd)
//   ROOT tuple(ab_new, cd_new)
// }
// a_new = slice(ab_new) first half
// b_new = slice(ab_new) second half
// c_new = slice(cd_new) first half
// d_new = slice(cd_new) second half
class WhileLoopConcatCodeMotion : public HloModulePass {
 public:
  explicit WhileLoopConcatCodeMotion(int64_t min_operand_count_to_optimize)
      : min_operand_count_to_optimize_(min_operand_count_to_optimize) {}
  ~WhileLoopConcatCodeMotion() override = default;

  absl::string_view name() const override {
    static constexpr absl::string_view kName = "while-loop-concat-code-motion";
    return kName;
  }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  const int64_t min_operand_count_to_optimize_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONCAT_CODE_MOTION_H_
