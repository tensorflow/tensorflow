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
  return (hlo.IsElementwise() && hlo.operand_count() > 0) ||
         hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kFusion ||
         hlo.opcode() == HloOpcode::kGetTupleElement ||
         hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReduceWindow ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}

}  // namespace

bool GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                      int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

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

  // We may need to know original operand layout to emit input fusion, and so
  // far, we merely use the layout of an operand of the fusion node, which means
  // we must fuse only elementwise operations. This restriction should be lifted
  // later if we need to fuse other operations, e.g. transpose, for performance.
  if ((IsReductionToVector(*consumer) ||
       (HloOpcode::kFusion == consumer->opcode() &&
        HloInstruction::FusionKind::kInput == consumer->fusion_kind())) &&
      !producer->IsElementwise()) {
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
  if (HloOpcode::kFusion == consumer->opcode()) {
    return consumer->fusion_kind();
  }
  return InstructionFusion::ChooseKind(producer, consumer);
}

}  // namespace gpu
}  // namespace xla
