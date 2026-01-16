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

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/float_support.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool GpuFloatSupport::SupportsMixedPrecisions(const HloInstruction& hlo) const {
  if (FloatSupport::SupportsMixedPrecisions(hlo)) return true;

  switch (hlo.opcode()) {
    // Handled by Triton GEMM or cuBLAS.
    case HloOpcode::kScaledDot:
      // We accept any scaled dot, because there is a rewrite pass that will
      // lower it to a dot + multiply for unsupported types.
      return true;
    // Handled by Triton GEMM or cuBLAS.
    case HloOpcode::kDot: {
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

bool IsAnySubByteNonPredType(const Shape& shape) {
  bool result = false;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        result |= primitive_util::IsSubByteNonPredType(subshape.element_type());
      });
  return result;
}

bool GpuFloatSupport::IsSupported(const HloInstruction& hlo) const {
  if (IsCollective(&hlo) && IsAnySubByteNonPredType(hlo.shape())) {
    return false;
  }
  switch (hlo.opcode()) {
    // Collective ops.
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kScaledDot:
      // We accept any scaled dot, because there is a rewrite pass that will
      // lower it to a dot + multiply for unsupported types.
      return true;
    case HloOpcode::kDot:  // Handled by Triton GEMM.
      using TypeAndCC =
          std::pair<PrimitiveType, stream_executor::CudaComputeCapability>;
      for (auto [type, cc] :
           {TypeAndCC(F8E4M3FN, se::CudaComputeCapability::Ampere()),
            TypeAndCC(F8E5M2, se::CudaComputeCapability::Hopper())}) {
        if (LowPrecisionType() == type) {
          auto* cuda_compute_capability =
              compute_capability_.cuda_compute_capability();
          // Do not normalize supported types inside Triton fused computations.
          return cuda_compute_capability &&
                 cuda_compute_capability->SupportsAllFeaturesOf(cc) &&
                 IsTritonFusedComputation(*hlo.parent());
        }
      }
      return LowPrecisionType() == BF16;
    case HloOpcode::kConvolution:
      // Only affect conv fusion codepath, conv is rewritten to custom call
      // before this if conv fusion not enabled
      return true;
    // Data movement only ops.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kConcatenate:
    case HloOpcode::kCopy:
    case HloOpcode::kConstant:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    // Other special ops.
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kConvert:
    case HloOpcode::kCompare:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kXor:
      return true;
    // Elementwise ops.
    case HloOpcode::kExp:
    case HloOpcode::kLog:
      if (LowPrecisionType() == BF16) {
        return compute_capability_.IsCuda();
      }
      return false;
    case HloOpcode::kAbs:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kNegate:
      if (LowPrecisionType() == BF16) {
        auto* cuda_compute_capability =
            compute_capability_.cuda_compute_capability();
        return cuda_compute_capability != nullptr &&
               cuda_compute_capability->IsAtLeastAmpere();
      }
      return false;
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract: {
      if (LowPrecisionType() == BF16) {
        auto* cuda_compute_capability =
            compute_capability_.cuda_compute_capability();
        return cuda_compute_capability != nullptr &&
               cuda_compute_capability->IsAtLeastHopper();
      }
      return false;
    }
    // Reduction.
    case HloOpcode::kReduce:
      return absl::c_all_of(hlo.called_computations().front()->instructions(),
                            [this](const HloInstruction* hlo) {
                              return hlo->opcode() == HloOpcode::kParameter ||
                                     this->IsSupported(*hlo);
                            });
    // Sort
    case HloOpcode::kSort:
      VLOG(10) << "Sort: " << hlo.ToString();
      VLOG(10) << "Comparator: " << hlo.to_apply()->ToString();
      return absl::c_all_of(hlo.to_apply()->instructions(),
                            [this](const HloInstruction* hlo) {
                              return hlo->opcode() == HloOpcode::kParameter ||
                                     this->IsSupported(*hlo);
                            });
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
