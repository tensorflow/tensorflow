/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convolution_pred_expander.h"

#include <iterator>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace m = match;

bool ConvolutionPredExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return Match(instruction, m::Convolution(m::Op().WithElementType(PRED),
                                           m::Op().WithElementType(PRED))
                                .WithElementType(PRED));
}

StatusOr<HloInstruction*> ConvolutionPredExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();

  // Convert convolution operands to F16.
  // The underlying convolution library (cuDNN) supports convolutions on FP and
  // integer (only S8) inputs. We cannot use S8, as the value could overflow to
  // zero, so we use F16 instead - it is not as efficient, but is correct.
  absl::InlinedVector<HloInstruction*, 2> new_operands;
  absl::c_transform(instruction->operands(), std::back_inserter(new_operands),
                    [&](HloInstruction* operand) {
                      CHECK_EQ(operand->shape().element_type(), PRED);
                      return MakeConvertToHlo(operand, F16);
                    });

  // Replace instruction with integer convolution and convert back to PRED.
  Shape new_shape = ShapeUtil::ChangeElementType(instruction->shape(), F16);
  HloInstruction* new_instruction = computation->AddInstruction(
      instruction->CloneWithNewOperands(new_shape, new_operands));
  return MakeConvertToHlo(new_instruction, PRED);
}

}  // namespace xla
