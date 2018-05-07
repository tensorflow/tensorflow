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

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

bool IsFusile(const HloInstruction& hlo) {
  // Don't fuse get-tuple-element on GPU: We can, but it's slower than not
  // fusing.  We never generate kernels for unfused GTEs.  Instead, if an
  // unfused GTE is an input to a kernel (including a fusion kernel), we
  // compute the address of the GTE at the top of the kernel.  Often we know the
  // address of the GTE result statically, so we can do this without chasing any
  // pointers.
  return (hlo.IsElementwise() && hlo.operand_count() > 0) ||
         hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kFusion ||
         hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReduceWindow ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}

}  // namespace

/*static*/ bool GpuInstructionFusion::IsExpensive(
    const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    // We say that floating-point division is cheap on the GPU.
    case HloOpcode::kDivide:
      return !ShapeUtil::ElementIsFloating(instruction.shape()) &&
             InstructionFusion::IsExpensive(instruction);

    default:
      return InstructionFusion::IsExpensive(instruction);
  }
}

bool GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Check if we can use output fusion for (A @ B) * alpha
  if (producer->opcode() == HloOpcode::kDot) {
    if (consumer->opcode() == HloOpcode::kMultiply) {
      CHECK_EQ(consumer->operand_count(), 2);
      int64 other_operand_index = 1 - operand_index;
      const HloInstruction* alpha = consumer->operand(other_operand_index);
      if (alpha->opcode() == HloOpcode::kConstant &&
          ShapeUtil::IsScalar(alpha->shape())) {
        return true;
      }
    }
  }

  // Only allow to fuse transpose into an output fusion.
  if (consumer->opcode() == HloOpcode::kFusion &&
      consumer->fusion_kind() == HloInstruction::FusionKind::kOutput) {
    if (producer->opcode() != HloOpcode::kTranspose) {
      return false;
    }
    // Check that the transpose is the operand of a dot.
    auto producer_operand_index = consumer->operand_index(producer);
    auto fused_parameter = consumer->fused_parameter(producer_operand_index);
    const std::vector<HloInstruction*>& fused_parameter_users =
        fused_parameter->users();
    return (fused_parameter_users.size() == 1 &&
            fused_parameter_users[0]->opcode() == HloOpcode::kDot);
  }

  // Output fusion is not currently supported on GPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return false;
  }

  // RNG operations are not currently parallel-friendly on GPU.
  if (producer->opcode() == HloOpcode::kRng) {
    return false;
  }

  // Do not fuse to-vector reduction into other consumers. They should be
  // unfused or the root of a kInput fusion.
  if (IsReductionToVector(*producer)) {
    return false;
  }

  // We can't fuse library calls, so if a user of such an op could become a
  // bitcast, leave it unfused. See `xla::InstructionFusion::ShouldFuse` for
  // further rationale.
  if (producer->CouldBeBitcast() &&
      ImplementedAsLibraryCall(*producer->operand(0))) {
    return false;
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion &&
      consumer->ReusesOperandElements(operand_index) &&
      is_expensive(*producer)) {
    return false;
  }

  return IsFusile(*producer) && IsFusile(*consumer) &&
         InstructionFusion::ShouldFuse(consumer, operand_index);
}

HloInstruction::FusionKind GpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  if (IsReductionToVector(*consumer)) {
    return HloInstruction::FusionKind::kInput;
  }
  if (producer->opcode() == HloOpcode::kDot) {
    return HloInstruction::FusionKind::kOutput;
  }
  if (HloOpcode::kFusion == consumer->opcode()) {
    return consumer->fusion_kind();
  }
  return InstructionFusion::ChooseKind(producer, consumer);
}

}  // namespace gpu
}  // namespace xla
