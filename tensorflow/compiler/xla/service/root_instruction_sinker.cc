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

#include "tensorflow/compiler/xla/service/root_instruction_sinker.h"

#include "tensorflow/compiler/xla/service/tuple_util.h"
namespace xla {

namespace {

// Sinks the root of the given computation for tuple root types.
void SinkTupleRoot(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  CHECK(root->shape().IsTuple());
  HloInstruction* new_root = TupleUtil::Duplicate(root);
  // Add the new instructions to the schedule.
  HloInstructionSequence& sequence =
      computation->parent()->schedule().GetOrCreateSequence(computation);
  for (HloInstruction* operand : new_root->operands()) {
    sequence.push_back(operand);
  }
  sequence.push_back(new_root);
  computation->set_root_instruction(new_root);
}

// Sinks the root of the given computation for not-tuple root types.
void SinkNontupleRoot(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  CHECK(!root->shape().IsTuple());
  HloInstruction* new_root = computation->AddInstruction(
      HloInstruction::CreateBitcast(root->shape(), root));
  HloInstructionSequence& sequence =
      computation->parent()->schedule().GetOrCreateSequence(computation);
  sequence.push_back(new_root);
  computation->set_root_instruction(new_root);
}

}  // namespace

StatusOr<bool> RootInstructionSinker::Run(HloModule* module) {
  TF_RET_CHECK(module->has_schedule());

  bool modified = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    HloInstructionSequence& sequence =
        module->schedule().GetOrCreateSequence(computation);
    if (computation->root_instruction() ==
        sequence.instructions().at(sequence.size() - 1)) {
      continue;
    }
    if (computation->root_instruction()->shape().IsTuple()) {
      SinkTupleRoot(computation);
    } else {
      SinkNontupleRoot(computation);
    }
    modified = true;
  }
  return modified;
}

}  // namespace xla
