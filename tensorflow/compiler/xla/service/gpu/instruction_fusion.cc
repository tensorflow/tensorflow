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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

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

// This function limits the maximum number of operands to a fusion.
//
// There's a cap on how many parameters we can pass to a CUDA kernel, but
// exactly what that limit is hazy, as it depends on (among other things) how
// much GPU constant memory is in use for other purposes.
//
// Moreover, we don't even know at the point that we're running fusion how many
// arguments the CUDA kernel for a fusion node will have: It depends on buffer
// assignment, where we will decide which of the fusion's operands live in XLA's
// big temp buffer versus in other allocations.
//
// As a heuristic, we simply cap the number of fusion operands plus outputs at
// kMaxOperandsAndOutputsPerFusion.  This puts an upper bound on the number of
// parameters to the kernel, working around the correctness problem.
//
// This limit is also often good for performance.  In a fusion with many
// operands, each GPU thread likely has to do a lot of work, and so possibly
// uses a lot of registers, thus limiting occupancy.
/*static*/ bool GpuInstructionFusion::FusionWouldBeTooLarge(
    const HloInstruction* a, const HloInstruction* b) {
  // Compute the number of outputs of the (possibly multi-output) fusion node
  // we're considering creating.
  //
  // This isn't precise; we may be off by one if
  //  - We're creating a multi-output fusion out of two non-MOFs.  Creating a
  //    MOF adds a new buffer, namely, the tuple buffer.
  //  - We're merging two MOFs.  In this case, we should count the tuple buffer
  //    only once.
  //  - WLOG there's an edge from `a` to `b` and `b` is the only consumer of
  //    `a`.  In this case the result of `a` is not part of the output of the
  //    fusion.
  //
  // But because this is a heuristic and our limit
  // kMaxOperandsAndOutputsPerFusion is a large value (so +/- 1 doesn't make a
  // big difference), we ignore this small inaccuracy in favor of simplicity.
  int64 num_output_buffers = ShapeUtil::SubshapeCount(a->shape()) +
                             ShapeUtil::SubshapeCount(b->shape());

  // The new fusion will have no more operands and outputs than
  //   producer_operands + consumer_operands - 1 + num_output_buffers
  // (minus one because we may be fusing a producer->consumer edge between `a`
  // and `b`).
  //
  // This fact may be enough to let us avoid having to compute the true total
  // number of operands, which can be expensive.
  if (a->operand_count() + b->operand_count() - 1 + num_output_buffers <=
      kMaxOperandsAndOutputsPerFusion) {
    return false;
  }

  // Compute the precise number of operands to the new fusion.
  absl::flat_hash_set<const HloInstruction*> operands(a->operands().begin(),
                                                      a->operands().end());
  operands.insert(b->operands().begin(), b->operands().end());
  // If there's an edge between `a` and `b`, don't count it: We're fusing that
  // producer -> consumer relationship.
  operands.erase(a);
  operands.erase(b);
  return operands.size() + num_output_buffers > kMaxOperandsAndOutputsPerFusion;
}

bool GpuInstructionFusion::ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                                       int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Check if we can use output fusion for (A @ B) * alpha
  if (producer->opcode() == HloOpcode::kDot && ImplementedAsGemm(*producer)) {
    int64 other_operand_index = 1 - operand_index;
    HloInstruction* op1 = nullptr;
    HloInstruction* op2 = nullptr;
    if (consumer->operand_count() == 1 && consumer->IsLoopFusion() &&
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
    } else if (consumer->operand_count() == 2 &&
               consumer->opcode() == HloOpcode::kAdd &&
               consumer->operand(other_operand_index) != producer) {
      // Fuse a bias add into the output of the dot.
      return true;
    }
  }

  // Only allow fusing transpose or broadcast into an output fusion that is
  // implemented as a Gemm call.
  if (consumer->IsOutputFusion() && ImplementedAsGemm(*consumer)) {
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
    return false;
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
  if (IsReductionFromOrToContiguousDimensions(*producer)) {
    return false;
  }

  // Scatter is only supported at the root of a kInput fusion.
  if (producer->opcode() == HloOpcode::kScatter) {
    return false;
  }

  // Do not fuse into reduce input fusions if the resulting kernel would suffer
  // from poor data locality (due to unfriendly input layouts).
  if (IsInputFusibleReduction(*consumer) &&
      !LayoutsAreReduceInputFusionFriendly(*producer, *consumer)) {
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

  // Fuse scalar constants into loop fusion nodes. This reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  //
  // Don't fuse other constants: Unfused constants in GPU land can be
  // represented as an external constant (i.e. not emitted in LLVM IR / PTX),
  // but fused constants are handled by shrared CPU/GPU code and always emitted
  // in the IR/PTX.  The external constant representation makes for faster
  // compiles and significantly smaller assembly code.
  if (producer->opcode() == HloOpcode::kConstant) {
    return ShapeUtil::IsEffectiveScalar(producer->shape()) &&
           consumer->opcode() == HloOpcode::kFusion;
  }

  if (!IsFusible(*producer) || !IsFusible(*consumer) ||
      !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    return false;
  }
  return true;
}

bool GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  if (!ShouldFuseInexpensiveChecks(consumer, operand_index)) {
    return false;
  }
  auto producer = consumer->operand(operand_index);

  // Don't fuse variadic reduce.
  if (consumer->opcode() == HloOpcode::kReduce && consumer->shape().IsTuple()) {
    return false;
  }
  // The following checks are potentially expensive.
  if (FusionWouldBeTooLarge(consumer, producer)) {
    return false;
  }
  // Also check that our emitter can handle the fusion node. We currently can
  // have exponential time/memory requirements for emitting certain fusion
  // kernels, in which case we don't want to fuse.
  // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
  return !FusedIrEmitter::IsFusedIrEmitterInefficient(consumer, producer);
}

bool GpuInstructionFusion::ShouldFuseIntoMultiOutput(HloInstruction* consumer,
                                                     int64 operand_index) {
  return false;
}

HloInstruction::FusionKind GpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  if (IsReductionFromOrToContiguousDimensions(*consumer) ||
      consumer->opcode() == HloOpcode::kScatter) {
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
