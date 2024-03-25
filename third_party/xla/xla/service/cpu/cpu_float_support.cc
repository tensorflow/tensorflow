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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/cpu_float_support.h"

#include "xla/service/cpu/onednn_matmul_rewriter.h"

namespace xla {
namespace cpu {

bool CpuFloatSupport::IsSupported(const HloInstruction& hlo) const {
  switch (hlo.opcode()) {
    // oneDNN rewritable ops
    case HloOpcode::kDot:
      return LowPrecisionType() == BF16 &&
             OneDnnMatMulRewriter::ShouldRewrite(&hlo) && DotSupported(hlo);
    // Collective ops.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kReduceScatter:
    // Data movement only ops.
    case HloOpcode::kBroadcast:
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
    default:
      return false;
  }
}

bool CpuFloatSupport::DotSupported(const HloInstruction& hlo) const {
  bool supported = true;
  const Shape& lhs_shape = hlo.operand(0)->shape();
  const Shape& rhs_shape = hlo.operand(1)->shape();
  if (lhs_shape.rank() == rhs_shape.rank() && lhs_shape.rank() == 2) {
    // If first dim size is 1, it may be removed by a later pass which makes it
    // unsupported case.
    supported &= lhs_shape.dimensions(0) != 1;
  }
  return supported;
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
