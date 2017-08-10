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

#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"

#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace cpu {

bool CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return false;
  }

  // Condition for consumer: must be elementwise or a fusion op
  // (which necessarily only contains elementwise operations)
  if (!(consumer->opcode() == HloOpcode::kFusion ||
        consumer->IsElementwise())) {
    return false;
  }

  // Producer or consumer cannot be Map. Maps are technically elementwise but
  // of a slightly different form (call instead of a computation). These are not
  // yet supported in the CPU backend.
  return producer->IsElementwise() && producer->operand_count() > 0 &&
         producer->opcode() != HloOpcode::kMap &&
         consumer->opcode() != HloOpcode::kMap &&
         InstructionFusion::ShouldFuse(consumer, operand_index);
}

}  // namespace cpu
}  // namespace xla
