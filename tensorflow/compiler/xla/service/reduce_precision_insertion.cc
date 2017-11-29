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

#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> ReducePrecisionInsertion::Run(HloModule* module) {
  bool changed = false;
  VLOG(1) << "Running ReducePrecisionInsertion pass on " << module->name();

  for (auto& computation : module->computations()) {
    std::vector<HloInstruction*> instructions_to_suffix;

    for (auto& instruction : computation->instructions()) {
      VLOG(3) << "Visited instruction: " << instruction->ToString();

      // For now, ReducePrecision is only implemented for F32 data, so this
      // ignore instructions that produce other data.  In particular, this
      // currently ignores instructions producing tuples, even if those tuples
      // contain F32 data inside them.  The assumption is that in most cases
      // equivalent behavior can be obtained by adding ReducePrecision
      // instructions after the instructions that pull the F32 data out of the
      // tuples.
      if (instruction->shape().element_type() == PrimitiveType::F32 &&
          should_reduce_output_precision_(instruction->opcode())) {
        instructions_to_suffix.push_back(instruction.get());
      }
    }

    for (auto& instruction : instructions_to_suffix) {
      HloInstruction* reduced =
          computation->AddInstruction(HloInstruction::CreateReducePrecision(
              instruction->shape(), instruction, exponent_bits_,
              mantissa_bits_));
      TF_RETURN_IF_ERROR(
          computation->ReplaceUsesOfInstruction(instruction, reduced));
      VLOG(2) << "Inserted new op after instruction: "
              << instruction->ToString();
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
