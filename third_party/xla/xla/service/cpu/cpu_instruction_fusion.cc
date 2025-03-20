/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_instruction_fusion.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"

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
         hlo->opcode() == HloOpcode::kDot && hlo_shape.rank() <= 1 &&
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

void CpuInstructionFusion::ComputeInstructionsToSkip(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto computations_list =
      module->MakeComputationPostOrder(execution_threads);
  instructions_to_skip_.clear();
  const bool is_fusion_emitters =
      module->config().debug_options().xla_cpu_use_thunk_runtime() &&
      module->config().debug_options().xla_cpu_use_fusion_emitters();
  for (auto* computation : computations_list) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomFusion() ||
          instruction->opcode() == HloOpcode::kCustomCall) {
        HloCallableInstruction* callable =
            Cast<HloCallableInstruction>(instruction);
        if (callable->called_computations().empty()) {
          continue;
        }
        for (HloInstruction* instr :
             callable->called_computation()->instructions())
          instructions_to_skip_.insert(instr);
      } else if (is_fusion_emitters &&
                 instruction->opcode() == HloOpcode::kScatter) {
        // Disallow fusions in the called computation (e.g. reduction)
        // of a scatter "fusion"; the fusion emitter can't handle them.
        auto* scatter = Cast<HloScatterInstruction>(instruction);
        for (const auto* computation : scatter->called_computations()) {
          for (const auto* instr : computation->instructions()) {
            instructions_to_skip_.insert(instr);
          }
        }
      }
    }
  }
}

bool CpuInstructionFusion::ShouldSkip(const HloInstruction* inst) const {
  return instructions_to_skip_.contains(inst);
}

FusionDecision CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  if (ShouldSkip(consumer)) {
    return FusionDecision::Forbid(
        "Don't fuse instructions from custom fusions/calls");
  }

  HloInstruction* producer = consumer->mutable_operand(operand_index);
  VLOG(2) << "Considering for fusion: operand " << operand_index << " of "
          << consumer->ToString();

  static constexpr int64_t kFusionThresholdBytes = 16 * 1024;

  // When we fuse a concatenate we don't take the fast path of simple memcpy /
  // for-loop; instead we currently emit a tree mapping the input to output idx
  // with a depth of log2(#args), this can have a large overhead for large
  // number of arguments.
  static constexpr int64_t kMaxConcatenateArguments = 8;

  if (IsLargeConstant(producer)) {
    return FusionDecision::Forbid("Don't fuse large constants.");
  }

  if (CanBeOutputFused(producer, consumer)) {
    VLOG(2) << "Fusion OK: Can create output fusion.";
    return FusionDecision::Allow();
  }

  if (CanBeOutputFusedIntoSomeOperand(producer)) {
    return FusionDecision::Forbid(
        "Bailing because producer can be output-fused into some operand.");
  }

  if (!CanBeLoopFused(*producer)) {
    return FusionDecision::Forbid("Producer is not loop-fusible.");
  }

  // Concatenation on the minor dimension leads to inefficient code with a lot
  // of branches in the innermost loop. We prefer to materialize concatenated
  // buffers and run concat as a separate operation, as LLVM tends to do a
  // better job with pure data movement loops.
  auto is_minor_dim_concatenate = [](const HloInstruction* hlo) {
    // For vectors it's always beneficial to fuse concatenations.
    if (hlo->shape().rank() <= 1) return false;

    // For small concatenated dimensions we don't loose any performance by
    // fusing the concatenation as we don't have opportunities for vectorization
    // anyway.
    int64_t concat_dim = hlo->concatenate_dimension();
    return concat_dim == LayoutUtil::Minor(hlo->shape().layout(), 0) &&
           hlo->shape().dimensions(concat_dim) >= 128;
  };

  if ((producer->opcode() == HloOpcode::kConcatenate &&
       (producer->operand_count() > kMaxConcatenateArguments ||
        is_minor_dim_concatenate(producer))) ||
      (consumer->opcode() == HloOpcode::kConcatenate &&
       (consumer->operand_count() > kMaxConcatenateArguments ||
        is_minor_dim_concatenate(consumer)))) {
    return FusionDecision::Forbid("Concatenate fusion is inefficient.");
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return FusionDecision::Forbid("Fusion is not profitable.");
  }

  RETURN_IF_NOT_FUSIBLE(InstructionFusion::ShouldFuse(consumer, operand_index));

  // Fuse constants in general but avoid creating 2-instruction fusions with
  // just a constant and another node.
  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() != HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: insufficient non-constant nodes.");
  }

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: producer is itself a fusion node.");
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
      return FusionDecision::Forbid("Code duplication too high");
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
    if (output_shape.rank() <= 1) {
      // We fuse in cases where we have a matrix*vector or vector*matrix dot and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (consumer->operand(0)->shape().rank() == 1 && operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      } else if (consumer->operand(1)->shape().rank() == 1 &&
                 operand_index == 0 &&
                 ShapeUtil::ByteSizeOfElements(consumer->operand(1)->shape()) <
                     kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      }
    }
  }

  if (consumer->IsLoopFusion()) {
    VLOG(2) << "Fusing: consumer is a fusion node.";
    return FusionDecision::Allow();
  }

  if (CanBeLoopFused(*consumer)) {
    VLOG(2) << "Fusing: consumer is elementwise or fusible.";
    return FusionDecision::Allow();
  }

  return FusionDecision::Forbid("Not fusing: not found a fusible case");
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

bool CpuInstructionFusion::IsLargeConstant(
    const HloInstruction* constant) const {
  return constant->IsConstant() &&
         Cast<HloConstantInstruction>(constant)->literal().size_bytes() >
             GetLargeConstantThresholdBytes();
}
}  // namespace cpu
}  // namespace xla
