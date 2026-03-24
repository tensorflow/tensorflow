/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/fusion_wrapper.h"

#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

bool FusionWrapper::MustWrapInstruction(const HloInstruction& instruction) {
  const HloOpcode opcode = instruction.opcode();
  switch (opcode) {
    case HloOpcode::kScatter:
      return true;
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicSlice:
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
    case HloOpcode::kReduce:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kXor:
      return using_new_fusion_emitter_;
    case HloOpcode::kCopy:
      // If it is a simple copy with no change in layout then it is more
      // efficient to use the default copy thunk which will just be a simple
      // large memcpy.
      if (instruction.shape() == instruction.operand(0)->shape()) {
        return false;
      }

      if (use_tiled_emitter_) {
        PrimitiveType type = instruction.shape().element_type();
        return IsSupportedTilingType(type);
      }
      return false;
    // The following ops are supported but the performance is not as good as the
    // non-fusion path.
    // TODO(willfroom): Remove this once the performance is improved.
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kDot:
      return false;
    default:
      return false;
  }
}

}  // namespace cpu
}  // namespace xla
