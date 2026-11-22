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

#include "xla/backends/gpu/codegen/triton/support_legacy.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {
namespace legacy_triton {

// Types that are supported by Triton as dot output.
//
// BF16 is supported in a sense that all operations on it are implemented
// through F32 and converts have to be inserted into the HLO graph, but
// they can be missing during fusion.
bool IsTritonSupportedDotOutputType(
    const PrimitiveType t, const se::GpuComputeCapability& gpu_version) {
  switch (t) {
    case F16:
    case F32:
      return true;
    case F8E5M2:
      if (auto* cuda_cc = gpu_version.cuda_compute_capability()) {
        return cuda_cc->IsAtLeastAmpere();
      }
      if (auto* rocm_cc = gpu_version.rocm_compute_capability()) {
        return rocm_cc->has_ocp_fp8_support();
      }
      return false;
    case F8E4M3FN:
      if (auto* cuda_cc = gpu_version.cuda_compute_capability()) {
        return cuda_cc->IsAtLeastHopper();
      }
      if (auto* rocm_cc = gpu_version.rocm_compute_capability()) {
        return rocm_cc->has_ocp_fp8_support();
      }
      return false;
    case F8E4M3FNUZ:
    case F8E5M2FNUZ:
      if (auto* rocm_cc = gpu_version.rocm_compute_capability()) {
        return rocm_cc->has_nanoo_fp8_support();
      }
      return false;
    case BF16:
      if (auto ptr = gpu_version.rocm_compute_capability()) {
        return ptr->has_bf16_dtype_support();
      }
      return true;
    case S32:
      if (auto ptr = gpu_version.cuda_compute_capability()) {
        return ptr->IsAtLeastAmpere();
      }
      return false;
    default:
      return false;
  }
};

// Data types that are supported by the Triton emitters.
bool IsTritonSupportedDataType(PrimitiveType type,
                               const se::GpuComputeCapability& gpu_version) {
  if (IsTritonSupportedDotOutputType(type, gpu_version)) {
    return true;
  }
  switch (type) {
    case PRED:
    case S8:
    case S16:
    case S32:
      return true;
    default:
      return false;
  }
}

std::vector<HloOpcode> TritonSupportedUnaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  std::vector<HloOpcode> ret = {HloOpcode::kConvert};
  if (element_type == PrimitiveType::PRED) {
    ret.push_back(HloOpcode::kNot);
    return ret;
  }
  ret.push_back(HloOpcode::kAbs);
  ret.push_back(HloOpcode::kNegate);
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::F64) {
    ret.insert(ret.end(), {
                              HloOpcode::kAcos,  HloOpcode::kAcosh,
                              HloOpcode::kAsin,  HloOpcode::kAsinh,
                              HloOpcode::kAtanh, HloOpcode::kCbrt,
                              HloOpcode::kCeil,  HloOpcode::kCos,
                              HloOpcode::kCosh,  HloOpcode::kErf,
                              HloOpcode::kExp,   HloOpcode::kExpm1,
                              HloOpcode::kFloor, HloOpcode::kLog,
                              HloOpcode::kLog1p, HloOpcode::kRoundNearestEven,
                              HloOpcode::kRsqrt, HloOpcode::kSin,
                              HloOpcode::kSinh,  HloOpcode::kSqrt,
                              HloOpcode::kTan,   HloOpcode::kTanh,
                          });
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedBinaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor,
            HloOpcode::kCompare};
  }
  std::vector<HloOpcode> ret = {HloOpcode::kAdd,      HloOpcode::kCompare,
                                HloOpcode::kMaximum,  HloOpcode::kMinimum,
                                HloOpcode::kMultiply, HloOpcode::kSubtract};
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kPower);
    if (element_type != PrimitiveType::F16) {
      ret.push_back(HloOpcode::kDivide);
    }
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedTernaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  std::vector<HloOpcode> ret{HloOpcode::kSelect, HloOpcode::kClamp};
  if (element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kFma);
  }
  return ret;
}

bool IsTritonSupportedElementwiseUpToFloatNormalization(
    HloOpcode opcode, PrimitiveType element_type) {
  return absl::c_linear_search(
             TritonSupportedUnaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode) ||
         absl::c_linear_search(
             TritonSupportedBinaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode) ||
         absl::c_linear_search(
             TritonSupportedTernaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode);
}

CodegenDecision IsTritonSupportedDynamicSlice(
    const HloDynamicSliceInstruction& instr) {
  for (const HloInstruction* index_operand : instr.index_operands()) {
    switch (index_operand->shape().element_type()) {
      case S8:
      case S16:
      case S32:
        break;  // supported
      default:
        return CodegenDecision::Forbid(
            "Dynamic slice is only supported with S8, S16, or S32 indices.");
    }
  }

  // Similar to normal slice, we cannot slice a non-major-most dimension as
  // that would introduce non-contiguous strides under tiling. The existing
  // check against this in GetRequirementsIfSupportedOrder is not suitable for
  // dynamic slices, so we instead check for this here.
  const HloInstruction* input = instr.operand(0);
  Layout in_layout = input->shape().layout();
  int64_t majormost_dim_id =
      in_layout.minor_to_major(in_layout.minor_to_major().size() - 1);

  for (int i = 0; i < input->shape().dimensions().size(); ++i) {
    if (i == majormost_dim_id) {
      continue;
    }
    if (input->shape().dimensions(i) != instr.slice_sizes(i)) {
      return CodegenDecision::Forbid(
          "Unsupported dynamic slice on non-major-most dimension.");
    }
  }

  // TODO(b/343143854): Check the subtleties of which dynamic slices are
  // supported, for example that a fragmented dimension cannot be sliced.
  return CodegenDecision::Allow();
}

}  // namespace legacy_triton
}  // namespace gpu
}  // namespace xla
