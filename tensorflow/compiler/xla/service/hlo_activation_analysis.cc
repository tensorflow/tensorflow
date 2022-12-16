/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_activation_analysis.h"

#include <memory>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {

void ActivationAnalysisOnComputation(const HloComputation* computation,
                                     ConstHloInstructionSet* activation_set) {
  for (const HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    // Dot or convolution create an "Activation".
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kConvolution) {
      activation_set->insert(hlo);
      continue;
    }

    // Don't mark tuples directly since we want to indirect through tuples.
    if (hlo->opcode() == HloOpcode::kTuple) {
      continue;
    }

    if (hlo->opcode() == HloOpcode::kGetTupleElement &&
        hlo->operand(0)->opcode() == HloOpcode::kTuple) {
      if (activation_set->count(hlo->operand(0)->operand(hlo->tuple_index()))) {
        activation_set->insert(hlo);
      }
      continue;
    }

    if (hlo->opcode() == HloOpcode::kGetTupleElement &&
        hlo->operand(0)->opcode() == HloOpcode::kWhile) {
      if (activation_set->count(
              hlo->operand(0)->while_body()->root_instruction()->operand(
                  hlo->tuple_index()))) {
        activation_set->insert(hlo);
      }
      continue;
    }

    if (hlo->opcode() == HloOpcode::kGetTupleElement &&
        hlo->operand(0)->opcode() == HloOpcode::kCall) {
      if (activation_set->count(DynCast<HloCallableInstruction>(hlo->operand(0))
                                    ->called_computation_root()
                                    ->operand(hlo->tuple_index()))) {
        activation_set->insert(hlo);
      }
      continue;
    }

    if (hlo->opcode() == HloOpcode::kGetTupleElement &&
        hlo->operand(0)->opcode() == HloOpcode::kConditional) {
      for (auto branch : hlo->operand(0)->branch_computations()) {
        if (activation_set->count(
                branch->root_instruction()->operand(hlo->tuple_index()))) {
          activation_set->insert(hlo);
        }
      }
      continue;
    }

    if (hlo->opcode() == HloOpcode::kWhile) {
      const HloInstruction* body_param =
          hlo->while_body()->parameter_instruction(0);
      if (!body_param->shape().IsTuple()) {
        if (activation_set->count(hlo->operand(0))) {
          activation_set->insert(body_param);
        }
      }
      for (const HloInstruction* use : body_param->users()) {
        if (use->opcode() == HloOpcode::kGetTupleElement &&
            activation_set->count(
                hlo->operand(0)->operand(use->tuple_index()))) {
          activation_set->insert(use);
        }
      }
      ActivationAnalysisOnComputation(hlo->while_body(), activation_set);
      continue;
    }

    // Skipping conditional and call for now.
    if (hlo->opcode() == HloOpcode::kConditional) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kCall) {
      continue;
    }

    for (const HloInstruction* operand : hlo->operands()) {
      if (activation_set->count(operand) ||
          (operand->opcode() == HloOpcode::kWhile &&
           activation_set->count(operand->while_body()->root_instruction()))) {
        activation_set->insert(hlo);
        break;
      }
    }
  }
}

ConstHloInstructionSet ComputeHloActivationAnalysis(const HloModule* module) {
  ConstHloInstructionSet activation_set;
  ActivationAnalysisOnComputation(module->entry_computation(), &activation_set);
  return activation_set;
}

}  // namespace xla
