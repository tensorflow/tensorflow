/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_parallelization_preparation.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace cpu {

StatusOr<bool> ParallelizationPreparation::Run(HloModule* module) {
  bool changed = false;
  HloComputation* entry_computation = module->entry_computation();
  std::unordered_set<HloInstruction*> outlined;
  std::vector<HloInstruction*> instructions_to_outline;
  for (HloInstruction* instruction :
       entry_computation->MakeInstructionPostOrder()) {
    // If the instruction has been outlined, it no longer exists and we must not
    // dereference it.
    if (outlined.count(instruction) > 0) {
      continue;
    }

    // Skip parameters and constants, there is nothing to parallelize.
    if (instruction->opcode() == HloOpcode::kParameter ||
        instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }
    instructions_to_outline.clear();
    HloInstruction* outline_candidate = instruction;
    instructions_to_outline.push_back(outline_candidate);
    bool all_bitcasts = outline_candidate->opcode() == HloOpcode::kBitcast;

    // Outline sole users with the current instruction.
    while (outline_candidate->users().size() == 1) {
      HloInstruction* prior_candidate = outline_candidate;
      outline_candidate = *outline_candidate->users().begin();
      all_bitcasts |= outline_candidate->opcode() == HloOpcode::kBitcast;
      if (std::any_of(outline_candidate->operands().begin(),
                      outline_candidate->operands().end(),
                      [&](const HloInstruction* operand) {
                        // Do not consider any candidates which have operands
                        // other than the prior candidate, constants or
                        // parameters. Otherwise, we'd increase the fan-in which
                        // would reduce parallelism.
                        return operand->opcode() != HloOpcode::kParameter &&
                               operand->opcode() != HloOpcode::kConstant &&
                               operand != prior_candidate;
                      })) {
        break;
      }
      instructions_to_outline.push_back(outline_candidate);
    }
    // If all instructions in the outline candidates are a bitcast, then create
    // a copy at the head of the bitcasts and include it in the outlined
    // instructions. The underlying problem is that a computation which forwards
    // a parameter buffer to the output is not properly handled by the backends
    // or analysis.
    //
    // This would be better handled by being smarter about choosing outline
    // candidates in the first place.
    if (all_bitcasts) {
      // 'head' is the first instruction in the chain of bitcasts.
      HloInstruction* head = instructions_to_outline[0];
      HloInstruction* head_operand = head->mutable_operand(0);
      HloInstruction* copy =
          entry_computation->AddInstruction(HloInstruction::CreateUnary(
              head_operand->shape(), HloOpcode::kCopy, head_operand));
      TF_RETURN_IF_ERROR(head->ReplaceOperandWith(0, copy));
      instructions_to_outline.insert(instructions_to_outline.begin(), copy);
    }

    outlined.insert(instructions_to_outline.begin(),
                    instructions_to_outline.end());

    // Optimization to avoid replacing a single existing kCall with another
    // kCall that just calls the first one.
    if (instructions_to_outline.size() == 1 &&
        instructions_to_outline[0]->opcode() == HloOpcode::kCall) {
      continue;
    }

    module->OutlineExpressionFromComputation(
        instructions_to_outline,
        tensorflow::strings::StrCat("pp_", instruction->name()),
        entry_computation);
    changed = true;
  }

  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  for (auto& computation : module->computations()) {
    HloInstruction* root = computation->root_instruction();
    // Copy root instruction if it does not define its own top-level buffer.
    // TODO(b/32885001) Remove these copies (at least for the unambiguous case).
    // TODO(b/32885001) Perform shallow copy if root value is a tuple.
    if (!points_to_analysis->InstructionDefinesBufferAtIndex(root,
                                                             /*index=*/{})) {
      HloInstruction* copy = computation->AddInstruction(
          HloInstruction::CreateUnary(root->shape(), HloOpcode::kCopy, root));
      computation->set_root_instruction(copy);
      changed = true;
    }
  }
  return changed;
}

}  // namespace cpu
}  // namespace xla
