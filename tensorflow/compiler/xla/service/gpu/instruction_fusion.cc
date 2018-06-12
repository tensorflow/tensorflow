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
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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

bool IsIEEEFloatingPointScalarConstant(const HloInstruction* constant) {
  if (constant->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(constant->shape())) {
    return false;
  }
  auto type = constant->shape().element_type();
  return type == F16 || type == F32 || type == F64;
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
  if (producer->opcode() == HloOpcode::kDot ||
      (producer->opcode() == HloOpcode::kFusion &&
       producer->fused_expression_root()->opcode() == HloOpcode::kDot)) {
    int64 other_operand_index = 1 - operand_index;
    HloInstruction* op1 = nullptr;
    HloInstruction* op2 = nullptr;
    if (consumer->operand_count() == 1 &&
        consumer->opcode() == HloOpcode::kFusion &&
        consumer->fusion_kind() == HloInstruction::FusionKind::kLoop &&
        Match(consumer->fused_expression_root(),
              match::Op()
                  .WithOpcode(HloOpcode::kMultiply)
                  .WithOperand(0, match::Op(&op1))
                  .WithOperand(1, match::Op(&op2)))) {
      CHECK(op1 != nullptr && op2 != nullptr);
      // If 'consumer' is a fusion node, it should consist of a broadcast of a
      // scalar constant fused into a multiply, but nothing more. So one operand
      // should be a parameter, and the other should be a broadcast.
      if (op1->opcode() != HloOpcode::kParameter) {
        std::swap(op1, op2);
      }
      if (op1->opcode() != HloOpcode::kParameter ||
          op2->opcode() != HloOpcode::kBroadcast) {
        return false;
      }
      if (IsIEEEFloatingPointScalarConstant(op2->operand(0))) {
        return true;
      }
    } else if (consumer->operand_count() == 2 &&
               consumer->opcode() == HloOpcode::kMultiply) {
      const HloInstruction* alpha = consumer->operand(other_operand_index);
      // Fuse if 'alpha' is a broadcast of a scalar constant.
      if (alpha->opcode() == HloOpcode::kBroadcast &&
          alpha->dimensions().empty() &&
          IsIEEEFloatingPointScalarConstant(alpha->operand(0))) {
        return true;
      }
    }
  }

  // Only allow fusing transpose or broadcast into an output fusion that is
  // implemented as a Gemm call.
  if (consumer->opcode() == HloOpcode::kFusion &&
      consumer->fusion_kind() == HloInstruction::FusionKind::kOutput &&
      ImplementedAsGemm(*consumer)) {
    auto producer_operand_index = consumer->operand_index(producer);
    auto fused_parameter = consumer->fused_parameter(producer_operand_index);
    const std::vector<HloInstruction*>& fused_parameter_users =
        fused_parameter->users();
    if (fused_parameter_users.size() != 1) {
      return false;
    }
    if (producer->opcode() == HloOpcode::kTranspose) {
      // Check that the transpose is an operand of a dot.
      return fused_parameter_users[0]->opcode() == HloOpcode::kDot;
    }
    if (producer->opcode() == HloOpcode::kBroadcast) {
      // Check that the broadcast is a broadcast of a scalar constant into a
      // multiply.
      return producer->dimensions().empty() &&
             IsIEEEFloatingPointScalarConstant(producer->operand(0)) &&
             fused_parameter_users[0]->opcode() == HloOpcode::kMultiply;
    }
  }

  // Other output fusions are not currently supported on GPUs.
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

  // Fuse scalar constants into loop fusion nodes, this reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  if (ShapeUtil::IsEffectiveScalar(producer->shape()) &&
      consumer->opcode() == HloOpcode::kFusion &&
      producer->opcode() == HloOpcode::kConstant) {
    return true;
  }

  return IsFusile(*producer) && IsFusile(*consumer) &&
         InstructionFusion::ShouldFuse(consumer, operand_index);
}

bool GpuInstructionFusion::ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                     int64 operand_index) {
  const HloInstruction* producer = consumer->operand(operand_index);
  // The IR emitter has limited support for non-loop fusions with multi output
  // at present.
  // TODO(tjoerg): Relax this constraint to allow for arbitraty kinds of fusion.
  if (consumer->opcode() == HloOpcode::kFusion &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return false;
  }
  // Multi-output fusion requires instructions with compatible shapes.
  if (!ShapeUtil::Compatible(producer->shape(), consumer->shape())) {
    return false;
  }
  // TODO(tjoerg): Stop calling `ShouldFuse` to relax the criteria for
  // multi-output fusion. In particular, do not check whether an instruction is
  // expensive to duplicate, since this doesn't matter here.
  return GpuInstructionFusion::ShouldFuse(consumer, operand_index);
}

HloInstruction::FusionKind GpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  if (IsReductionToVector(*consumer)) {
    return HloInstruction::FusionKind::kInput;
  }
  if (producer->opcode() == HloOpcode::kDot ||
      (producer->opcode() == HloOpcode::kFusion &&
       producer->fused_expression_root()->opcode() == HloOpcode::kDot)) {
    return HloInstruction::FusionKind::kOutput;
  }
  if (HloOpcode::kFusion == consumer->opcode()) {
    return consumer->fusion_kind();
  }
  return InstructionFusion::ChooseKind(producer, consumer);
}

}  // namespace gpu
}  // namespace xla
