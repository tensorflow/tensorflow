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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  tensorflow::gtl::FlatMap<string, const HloInstruction*> instructions;

  for (auto& computation : module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      TF_RET_CHECK(instruction->parent() == computation.get());
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (const auto& fused : instruction->fused_instructions()) {
          TF_RET_CHECK(fused->parent() ==
                       instruction->fused_instructions_computation())
              << "Fused HLO was missing a parent: " << fused->ToString()
              << " parent: " << fused->parent()
              << " computation: " << computation.get();
        }
      }

      auto previous = instructions.find(instruction->name());
      TF_RET_CHECK(previous == instructions.end())
          << "HLO has name that is not unique within module:\n"
          << instruction->ToString()
          << " in computation: " << computation->name()
          << "\nPrevious HLO with same name:\n"
          << previous->second->ToString()
          << " in computation: " << previous->second->parent()->name();
      instructions[instruction->name()] = instruction.get();
    }
  }

  return false;
}

}  // namespace xla
