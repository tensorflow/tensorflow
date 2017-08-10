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

namespace {

bool CanBeLoweredIntoElementalLoop(const HloInstruction& hlo) {
  if (hlo.IsElementwise()) {
    return hlo.operand_count() > 0;
  }

  // These non-elementwise ops have a lowering that generates the output for a
  // specified element at a time.
  return (hlo.opcode() == HloOpcode::kConcatenate ||
          hlo.opcode() == HloOpcode::kReverse ||
          hlo.opcode() == HloOpcode::kBroadcast ||
          hlo.opcode() == HloOpcode::kSlice ||
          hlo.opcode() == HloOpcode::kDynamicSlice ||
          hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
          hlo.opcode() == HloOpcode::kReshape ||
          hlo.opcode() == HloOpcode::kTranspose ||
          hlo.opcode() == HloOpcode::kPad);
}

}  // namespace

bool CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return false;
  }

  // Condition for consumer: must act elementwise on the operand. This permits
  // only elementwise ops or (potentially) fusion ops to act as consumers.
  if (!consumer->IsElementwiseOnOperand(operand_index)) {
    return false;
  }

  // Producer or consumer cannot be Map. Maps are technically elementwise but of
  // a slightly different form (call instead of a computation). These are not
  // yet supported in the CPU backend.
  if (producer->opcode() == HloOpcode::kMap ||
      consumer->opcode() == HloOpcode::kMap) {
    return false;
  }

  // Avoid dragging something that could otherwise be implemented as a
  // bitcast into the loop.
  if (producer->CouldBeBitcast()) {
    return false;
  }

  // Check to make sure that the producer can generate output a specified
  // element at a time.
  if (!CanBeLoweredIntoElementalLoop(*producer)) {
    return false;
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
}

}  // namespace cpu
}  // namespace xla
