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

int64 BytesInDimension(const Shape& shape, int64 dimension) {
  return ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) *
         shape.dimensions(dimension);
}

bool IsFusile(const HloInstruction& hlo) {
  // These are the only ones we fuse since we rely on effective elemental IR
  // generation.
  return (hlo.opcode() == HloOpcode::kBroadcast ||
          hlo.opcode() == HloOpcode::kReshape ||
          hlo.opcode() == HloOpcode::kBitcast ||
          hlo.opcode() == HloOpcode::kReverse ||
          hlo.opcode() == HloOpcode::kSlice ||
          hlo.opcode() == HloOpcode::kDynamicSlice ||
          hlo.opcode() == HloOpcode::kTranspose || hlo.IsElementwise());
}

}  // namespace

bool CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  constexpr int kFusionThresholdBytes = 16 * 1024;

  if (!IsFusile(*producer)) {
    return false;
  }

  // Producer or consumer cannot be Map. Maps are technically elementwise but
  // of a slightly different form (call instead of a computation). These are not
  // yet supported in the CPU backend.
  if (producer->opcode() == HloOpcode::kMap ||
      consumer->opcode() == HloOpcode::kMap) {
    return false;
  }

  // TODO(b/28644064): see if the "producer->operand_count() == 0" check is
  // necessary.
  if (producer->operand_count() == 0 ||
      !InstructionFusion::ShouldFuse(consumer, operand_index)) {
    return false;
  }

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return false;
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
    if (output_shape.dimensions_size() == 2) {
      // We fuse in cases where we have dot([A,B],[B,1]) or dot([1,A],[A,B]) and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (output_shape.dimensions(0) == 1 && operand_index == 1 &&
          BytesInDimension(output_shape, 1) < kFusionThresholdBytes) {
        return true;
      } else if (output_shape.dimensions(1) == 1 && operand_index == 0 &&
                 BytesInDimension(output_shape, 0) < kFusionThresholdBytes) {
        return true;
      }
    }
  }

  // InstructionFusion::ShouldFuse above only allows kLoop and kInput fusions.
  // The CPU backend does not create kInput fusions, so we only expect to see
  // kLoop here.
  CHECK(consumer->opcode() != HloOpcode::kFusion ||
        consumer->fusion_kind() == HloInstruction::FusionKind::kLoop);
  return consumer->opcode() == HloOpcode::kFusion || consumer->IsElementwise();
}

}  // namespace cpu
}  // namespace xla
