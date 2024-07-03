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

#include "xla/service/gpu/gpu_float_support.h"

#include <variant>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/float_support.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool GpuFloatSupport::SupportsMixedPrecisions(const HloInstruction& hlo) const {
  if (FloatSupport::SupportsMixedPrecisions(hlo)) return true;

  switch (hlo.opcode()) {
    // Handled by Triton GEMM or cuBLAS.
    case HloOpcode::kDot: {
      CHECK_GE(hlo.operand_count(), HloDotInstruction::kOperands);
      const PrimitiveType lhs_type = hlo.operand(0)->shape().element_type();
      const PrimitiveType rhs_type = hlo.operand(1)->shape().element_type();
      const PrimitiveType result_type = hlo.shape().element_type();
      return (lhs_type == F16 && rhs_type == F16 && result_type == F32) ||
             (lhs_type == BF16 && rhs_type == BF16 && result_type == F32);
    }
    default:
      return false;
  }
}

bool GpuFloatSupport::IsSupported(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    // Collective ops.
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kReduceScatter:
    // Handled by Triton GEMM.
    case HloOpcode::kDot:
      return LowPrecisionType() == BF16;
    // Data movement only ops.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kConcatenate:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    // Other special ops.
    case HloOpcode::kBitcast:
      return true;
    // Elementwise ops.
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply: {
      if (LowPrecisionType() == BF16) {
        auto* cuda_compute_capability =
            std::get_if<se::CudaComputeCapability>(&compute_capability_);
        return cuda_compute_capability != nullptr &&
               cuda_compute_capability->IsAtLeastHopper();
      }
      return false;
    }
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
