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
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

namespace {

bool CanBeLoopFused(const HloInstruction& hlo) {
  // These are the only ones we fuse since we rely on effective elemental IR
  // generation.
  return hlo.IsElementwise() || hlo.opcode() == HloOpcode::kBitcast ||
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
         hlo->opcode() == HloOpcode::kDot &&
         hlo_shape.dimensions().size() <= 1 &&
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

// Should we block the fusion of the subcomputation of the passed instruction?
bool BlockSubcomputationFusion(const HloInstruction* instruction,
                               const HloModuleConfig& config) {
  HloOpcode opcode = instruction->opcode();
  const bool is_fusion_emitters =
      config.debug_options().xla_cpu_use_fusion_emitters();

  if (is_fusion_emitters && opcode == HloOpcode::kScatter) {
    return true;
  }

  const bool use_experemental_fusion_emitters =
      options::UseExperimentalLoopFusion(config);

  // If the instruction itself can be fused then the subcomputation should be
  // blocked as the fusion emitter can't emit fusion ops inside another
  // fusion.
  if (is_fusion_emitters && use_experemental_fusion_emitters &&
      emitters::IsSupportedElementalOp(opcode)) {
    return true;
  }

  return false;
}

}  // namespace

bool CpuInstructionFusion::IsExpensive(const HloInstruction& instruction) {
  namespace m = match;

  switch (instruction.opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFloor:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOr:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSlice:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kXor:
      return false;

    // Cheap instructions for reals, but expensive for complex.
    case HloOpcode::kAbs:
    case HloOpcode::kSign:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    case HloOpcode::kConvert:
      // Converting from f32 to bf16 is expensive as we have to do multiple
      // checks for NaN, converting from bf16 to f32 is cheap as it is a simple
      // shift.
      return instruction.shape().element_type() == PrimitiveType::BF16 &&
             instruction.operand(0)->shape().element_type() ==
                 PrimitiveType::F32;

    // We say that integer div/mod by a constant is cheap because it gets
    // compiled down to multiplies and shifts, and we consider those to be
    // cheap.
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
      return !ShapeUtil::ElementIsIntegral(instruction.shape()) ||
             !Match(instruction.operand(0),
                    m::AnyOf<const HloInstruction>(
                        m::ConstantEffectiveScalar(),
                        m::Broadcast(m::ConstantEffectiveScalar())));

    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    case HloOpcode::kAcos:
    case HloOpcode::kAcosh:
    case HloOpcode::kSinh:
    case HloOpcode::kAsin:
    case HloOpcode::kAsinh:
    case HloOpcode::kAtan2:
    case HloOpcode::kAtanh:
    case HloOpcode::kCosh:
    case HloOpcode::kTanh:
      return true;

    case HloOpcode::kCbrt:
    case HloOpcode::kPower:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
      return true;

    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
      return true;

      // Expensive instructions or unusual instructions for which fusion is
      // nonsensical.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCall:
    case HloOpcode::kCholesky:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kLogistic:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kScaledDot:
    case HloOpcode::kScan:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kWhile:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return true;
  }

  return false;
}

void CpuInstructionFusion::ComputeInstructionsToSkip(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto computations_list =
      module->MakeComputationPostOrder(execution_threads);
  instructions_to_skip_.clear();

  for (auto* computation : computations_list) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomFusion()) {
        instructions_to_skip_.insert(instruction);
      } else if (instruction->opcode() == HloOpcode::kCustomCall) {
        HloCallableInstruction* callable =
            Cast<HloCallableInstruction>(instruction);
        if (callable->called_computations().empty()) {
          continue;
        }
        for (HloInstruction* instr :
             callable->called_computation()->instructions())
          instructions_to_skip_.insert(instr);
      } else if (BlockSubcomputationFusion(instruction, module->config())) {
        for (const auto* computation : instruction->called_computations()) {
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

  if (HloPredicateIsOp<HloOpcode::kConstant>(producer) &&
      !ShapeUtil::IsEffectiveScalar(producer->shape())) {
    return FusionDecision::Forbid("Don't fuse non-scalar constants.");
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
    if (hlo->shape().dimensions().size() <= 1) return false;

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

  // Fusing too many reductions together can lead to a giant LLVM modules after
  // loop unrolling. We prefer to split such fusions into multiple kernels to
  // avoid excessive compilation times. X86TargetLowering::PerformDAGCombine
  // spends tens of minutes trying to combine load operations.
  //
  // TODO(b/419635451): Remove this once we have a better way to control the
  // size of the generated LLVM IR.
  static constexpr int64_t kMaxReductionsInFusion = 5;
  if (consumer->opcode() == HloOpcode::kFusion &&
      producer->opcode() == HloOpcode::kReduce) {
    int64_t num_fused_reductions = absl::c_count_if(
        consumer->fused_instructions(), [](const HloInstruction* instr) {
          return instr->opcode() == HloOpcode::kReduce;
        });
    if (num_fused_reductions > kMaxReductionsInFusion) {
      return FusionDecision::Forbid(
          "Too many reductions inside single fusion.");
    }
  }

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
  if (may_duplicate() && consumer->opcode() == HloOpcode::kFusion) {
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
    if (output_shape.dimensions().size() <= 1) {
      // We fuse in cases where we have a matrix*vector or vector*matrix dot and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (consumer->operand(0)->shape().dimensions().size() == 1 &&
          operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      } else if (consumer->operand(1)->shape().dimensions().size() == 1 &&
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
  if (!may_duplicate()) {
    return InstructionFusion::FuseInstruction(fusion_instruction, producer);
  }

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
