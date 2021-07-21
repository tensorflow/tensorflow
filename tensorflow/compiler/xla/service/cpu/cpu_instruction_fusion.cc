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

#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

namespace xla {
namespace cpu {

namespace {

bool CanBeLoopFused(const HloInstruction& hlo) {
  // These are the only ones we fuse since we rely on effective elemental IR
  // generation.
  return hlo.IsElementwise() ||  //
         hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kGather ||
         hlo.opcode() == HloOpcode::kIota || hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kReverse ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}

bool IsNonComplexNonBatchedMatrixVectorDot(const HloInstruction* hlo) {
  const Shape& hlo_shape = hlo->shape();
  return !ShapeUtil::ElementIsComplex(hlo_shape) &&
         hlo->opcode() == HloOpcode::kDot && hlo_shape.dimensions_size() <= 1 &&
         hlo->dot_dimension_numbers().lhs_batch_dimensions_size() == 0;
}

bool HasExactlyOneUse(const HloInstruction& hlo_instr) {
  return hlo_instr.user_count() == 1 &&
         absl::c_count(hlo_instr.users().front()->operands(), &hlo_instr) == 1;
}

bool CanBeOutputFused(const HloInstruction* producer,
                      const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         IsNonComplexNonBatchedMatrixVectorDot(producer) &&
         HasExactlyOneUse(*producer) == 1;
}

bool CanBeOutputFusedIntoSomeOperand(const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         (CanBeOutputFused(consumer->operand(0), consumer) ||
          CanBeOutputFused(consumer->operand(1), consumer));
}
}  // namespace

bool CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  VLOG(2) << "Considering for fusion: operand " << operand_index << " of "
          << consumer->ToString();

  constexpr int kFusionThresholdBytes = 16 * 1024;

  if (CanBeOutputFused(producer, consumer)) {
    VLOG(2) << "Fusion OK: Can create output fusion.";
    return true;
  }

  if (CanBeOutputFusedIntoSomeOperand(producer)) {
    VLOG(2)
        << "Bailing because producer can be output-fused into some operand.";
    return false;
  }

  if (!CanBeLoopFused(*producer)) {
    VLOG(2) << "Producer is not fusible.";
    return false;
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    VLOG(2) << "Fusion is not profitable.";
    return false;
  }

  if (!InstructionFusion::ShouldFuse(consumer, operand_index)) {
    VLOG(2) << "Not fusing: !ShouldFuse(consumer).";
    return false;
  }

  // Fuse constants in general but avoid creating 2-instruction fusions with
  // just a constant and another node.
  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() != HloOpcode::kFusion) {
    VLOG(2) << "Not fusing: insufficient non-constant nodes.";
    return false;
  }

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(2) << "Not fusing: producer is itself a fusion node.";
    return false;
  }

  // Don't fuse if fusing would cause too much code duplication because of
  // inefficiencies in the fusion emitter.
  // TODO(b/119692968): Remove this once the fusion emitter can handle
  // arbitrary fusion nodes.
  if (consumer->opcode() == HloOpcode::kFusion) {
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      // We have no cached results for this fusion node yet. This can happen
      // when we run the InstructionFusion pass more than once. We can only
      // cache the results within one run.
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(
            producer)) {
      return false;
    }
  }

  if (consumer->opcode() == HloOpcode::kDot) {
    // In the general case we call out to optimized "black box" GEMM routines
    // for Dot, which precludes fusion.  However, in very specific cases, we try
    // to fuse Dot operations by generating an elemental dot implementation.
    //
    // We need to be careful and conservative here since any benefit we get from
    // fusion can easily be overshadowed by the overhead of a naive GEMM
    // algorithm in the IR.
    const Shape& output_shape = consumer->shape();
    if (output_shape.dimensions_size() <= 1) {
      // We fuse in cases where we have a matrix*vector or vector*matrix dot and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (consumer->operand(0)->shape().rank() == 1 && operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return true;
      } else if (consumer->operand(1)->shape().rank() == 1 &&
                 operand_index == 0 &&
                 ShapeUtil::ByteSizeOfElements(consumer->operand(1)->shape()) <
                     kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return true;
      }
    }
  }

  // Don't fuse reductions over the major dimensions. These have an efficient
  // lowering that's only implemented for the unfused case.
  if (consumer->opcode() == HloOpcode::kReduce) {
    return absl::c_linear_search(
        consumer->dimensions(),
        LayoutUtil::Minor(consumer->operand(0)->shape().layout(), 0));
  }
  if (producer->opcode() == HloOpcode::kReduce) {
    return absl::c_linear_search(
        producer->dimensions(),
        LayoutUtil::Minor(producer->operand(0)->shape().layout(), 0));
  }

  if (consumer->IsLoopFusion()) {
    VLOG(2) << "Fusing: consumer is a fusion node.";
    return true;
  }

  if (CanBeLoopFused(*consumer)) {
    VLOG(2) << "Fusing: consumer is elementwise or fusible.";
    return true;
  }

  VLOG(2) << "Not fusing.";
  return false;
}

HloInstruction::FusionKind CpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return CanBeOutputFused(producer, consumer)
             ? HloInstruction::FusionKind::kOutput
             : HloInstruction::FusionKind::kLoop;
}

HloInstruction* CpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
                     .emplace(fusion_instruction,
                              FusionNodeIndexingEvaluation(fusion_instruction))
                     .first;
  }
  auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
  HloInstruction* new_producer =
      InstructionFusion::FuseInstruction(fusion_instruction, producer);
  evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
  return new_producer;
}
}  // namespace cpu
}  // namespace xla
