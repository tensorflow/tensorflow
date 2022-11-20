/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/bfloat16_support.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {

bool BFloat16Support::SupportsBF16Operand(const HloInstruction& hlo,
                                          int64_t operand_index) const {
  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kConvert:
      CHECK_EQ(operand_index, 0);
      return hlo.operand(0)->shape().element_type() == BF16;
    default:
      break;
  }
  return false;
}

bool BFloat16Support::SupportsBF16Output(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kConvert:
      return hlo.shape().element_type() == BF16;
    default:
      break;
  }
  return false;
}

bool BFloat16Support::SupportsMixedPrecisions(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConvert:
    case HloOpcode::kCustomCall:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    default:
      break;
  }
  return false;
}

/* static */
bool BFloat16Support::EffectiveOperandPrecisionIsOutputPrecision(
    const HloInstruction& hlo, int64_t operand_index) {
  switch (hlo.opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kBroadcast:
    case HloOpcode::kClamp:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kBitcast:
      return hlo.shape().element_type() ==
             hlo.operand(0)->shape().element_type();
    case HloOpcode::kDynamicSlice:
      return operand_index == 0;
    case HloOpcode::kDynamicUpdateSlice:
      return operand_index == 0 || operand_index == 1;
    case HloOpcode::kGather:
      return operand_index == 0;
    case HloOpcode::kSelect:
      return operand_index == 1 || operand_index == 2;
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow: {
      HloComputation* reduce_comp = hlo.called_computations()[0];
      for (HloInstruction* inst : reduce_comp->instructions()) {
        if (inst->opcode() == HloOpcode::kParameter) {
          continue;
        }
        for (int64_t i = 0; i < inst->operand_count(); ++i) {
          if (!EffectiveOperandPrecisionIsOutputPrecision(*inst, i)) {
            return false;
          }
        }
      }
      return true;
    }
    default:
      break;
  }
  return false;
}

bool BFloat16Support::EffectiveOperandPrecisionIsBF16(
    const HloInstruction& hlo, int64_t operand_index) const {
  return false;
}

}  // namespace xla
