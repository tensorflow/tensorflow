/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/transforms/fusion_wrapper.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"

namespace xla {
namespace gpu {

bool FusionWrapper::MustWrapInstruction(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCbrt:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvolution:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kGather:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPad:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReshape:
    case HloOpcode::kReduce:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kTranspose:
    case HloOpcode::kXor:
      return true;
    default:
      return false;
  }
}

HloInstruction::FusionKind FusionWrapper::ChooseFusionKind(
    const HloInstruction& producer, const HloInstruction& consumer) {
  return gpu::ChooseFusionKind(producer, consumer, device_description_);
}

}  // namespace gpu
}  // namespace xla
