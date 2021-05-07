/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_OPERAND_FOLDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_OPERAND_FOLDING_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

// Folds Convert operands to wider types into instructions that supports wider
// result accumulation than the shape inference type.
//
// e.g. s32 hlo(s32 convert(s8), s32 convert(s8)) -> s32 hlo(s8, s8)
class ConvertOperandFolding : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "convert_operand_folding"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_OPERAND_FOLDING_H_
