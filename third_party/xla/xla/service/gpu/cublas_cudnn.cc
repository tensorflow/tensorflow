/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/cublas_cudnn.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

bool IsCublasGemm(const HloInstruction& hlo) {
  return IsLegacyCublasMatmul(hlo) || IsCublasLtMatmul(hlo) ||
         IsCublasLtMatmulF8(hlo);
}

bool IsLegacyCublasMatmul(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kGemmCallTarget;
}

bool IsCublasLtMatmul(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kCublasLtMatmulCallTarget;
}

bool IsCublasLtMatmulF8(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kCublasLtMatmulF8CallTarget;
}

bool IsTriangularSolve(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kTriangularSolveCallTarget;
}

const absl::string_view kGemmCallTarget = "__cublas$gemm";
const absl::string_view kCublasLtMatmulCallTarget = "__cublas$lt$matmul";
const absl::string_view kCublasLtMatmulF8CallTarget = "__cublas$lt$matmul$f8";
const absl::string_view kTriangularSolveCallTarget = "__cublas$triangularSolve";

const absl::string_view kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const absl::string_view kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const absl::string_view kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";
const absl::string_view kCudnnConvForwardCallTarget = "__cudnn$convForward";
const absl::string_view kCudnnConvForwardGraphCallTarget =
    "__cudnn$convForwardGraph";
const absl::string_view kCudnnConvReorderFilterCallTarget =
    "__cudnn$convReorderFilter";
const absl::string_view kCudnnConvReorderFilterAndBiasCallTarget =
    "__cudnn$convReorderFilterAndBias";

const absl::string_view kCudnnNormCallTarget = "__cudnn$norm";

// fMHA forward call targets.
const absl::string_view kCudnnfMHASoftmaxCallTarget = "__cudnn$fmhaSoftmax";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget =
    "__cudnn$fmhaScaleBiasSoftmaxDropout";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxCallTarget =
    "__cudnn$fmhaScaleBiasSoftmax";
const absl::string_view kCudnnfMHASoftmaxDropoutCallTarget =
    "__cudnn$fmhaSoftmaxDropout";

// fMHA backward call targets.
const absl::string_view kCudnnfMHASoftmaxBackwardCallTarget =
    "__cudnn$fmhaSoftmaxBackward";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget =
    "__cudnn$fmhaScaleBiasSoftmaxDropoutBackward";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget =
    "__cudnn$fmhaScaleBiasSoftmaxBackward";
const absl::string_view kCudnnfMHASoftmaxDropoutBackwardCallTarget =
    "__cudnn$fmhaSoftmaxDropoutBackward";

const absl::string_view kCubDeviceRadixSortTarget = "__cub$DeviceRadixSort";

bool IsCustomCallToDnnConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvForwardGraphCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget ||
         target == kCudnnConvBiasActivationForwardCallTarget;
}

bool IsCudnnConvolutionReorder(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvReorderFilterCallTarget ||
         target == kCudnnConvReorderFilterAndBiasCallTarget;
}

bool IsCustomCallToDnnNorm(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnNormCallTarget;
}

bool IsFwdCustomCallTofMHA(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnfMHASoftmaxCallTarget ||
         target == kCudnnfMHASoftmaxDropoutCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget;
}

bool IsBwdCustomCallTofMHA(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnfMHASoftmaxBackwardCallTarget ||
         target == kCudnnfMHASoftmaxDropoutBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
}

bool MHACallHasDropout(const absl::string_view fmha_call_name) {
  return fmha_call_name == kCudnnfMHASoftmaxDropoutCallTarget ||
         fmha_call_name == kCudnnfMHASoftmaxDropoutBackwardCallTarget ||
         fmha_call_name == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget ||
         fmha_call_name == kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
}

bool IsCustomCallTofMHA(const HloInstruction& hlo) {
  return (IsFwdCustomCallTofMHA(hlo) || IsBwdCustomCallTofMHA(hlo));
}

bool IsCubDeviceRadixSort(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kCubDeviceRadixSortTarget;
}

absl::StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    return CudnnConvKind::kForward;
  }
  if (target == kCudnnConvForwardGraphCallTarget) {
    return CudnnConvKind::kForwardGraph;
  }
  if (target == kCudnnConvBackwardInputCallTarget) {
    return CudnnConvKind::kBackwardInput;
  }
  if (target == kCudnnConvBackwardFilterCallTarget) {
    return CudnnConvKind::kBackwardFilter;
  }
  if (target == kCudnnConvBiasActivationForwardCallTarget) {
    return CudnnConvKind::kForwardActivation;
  }
  return Internal("Unexpected call target: %s", target);
}

std::string CudnnConvKindToString(CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kForward:
      return "forward";
    case CudnnConvKind::kBackwardFilter:
      return "backward_filter";
    case CudnnConvKind::kBackwardInput:
      return "backward_input";
    case CudnnConvKind::kForwardActivation:
      return "forward with activation";
    case CudnnConvKind::kForwardGraph:
      return "forward with pointwise operations";
  }
}

absl::StatusOr<CudnnfMHAKind> GetCudnnfMHAKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnfMHASoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxCallTarget) return CudnnfMHAKind::kSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxCallTarget)
    return CudnnfMHAKind::kScaleBiasSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kScaleBiasSoftmaxDropout;
  // backward
  if (target == kCudnnfMHASoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout;
  return Internal("Unexpected call target: %s", target);
}

std::string CudnnfMHAKindToString(CudnnfMHAKind kind) {
  switch (kind) {
    case CudnnfMHAKind::kSoftmax:
      return "fmha_softmax";
    case CudnnfMHAKind::kSoftmaxDropout:
      return "fmha_softmax_with_dropout";
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      return "fmha_bias_softmax_with_dropout";
    case CudnnfMHAKind::kScaleBiasSoftmax:
      return "fmha_bias_softmax";
    // backward
    case CudnnfMHAKind::kBackwardSoftmax:
      return "fmha_softmax_backward";
    case CudnnfMHAKind::kBackwardSoftmaxDropout:
      return "fmha_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout:
      return "fmha_bias_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleBiasSoftmax:
      return "fmha_bias_softmax_backward";
  }
}

absl::StatusOr<std::string> GetFMHAInstructionPrefix(
    const std::string& custom_call_target) {
  if (custom_call_target == kCudnnfMHASoftmaxDropoutCallTarget) {
    return "fmha-bmm-softmax-dropout-bmm";
  }
  if (custom_call_target == kCudnnfMHASoftmaxCallTarget) {
    return "fmha-bmm-softmax-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasSoftmaxCallTarget) {
    return "fmha-bmm-scale-bias-softmax-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget) {
    return "fmha-bmm-scale-bias-softmax-dropout-bmm";
  }

  // Backward calls
  if (custom_call_target == kCudnnfMHASoftmaxDropoutBackwardCallTarget) {
    return "fmha-bmm-softmax-dropout-bmm-backward";
  }
  if (custom_call_target == kCudnnfMHASoftmaxBackwardCallTarget) {
    return "fmha-bmm-softmax-bmm-backward";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget) {
    return "fmha-bmm-scale-bias-softmax-bmm-backward";
  }
  if (custom_call_target ==
      kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget) {
    return "fmha-bmm-scale-bias-softmax-dropout-bmm-backward";
  }
  return Internal("Unexpected call target: %s", custom_call_target);
}

// Give fmha instruction a more useful name than "custom-call.42".
absl::Status SetFMHAInstructionName(HloModule* module, HloInstruction* fmha) {
  TF_ASSIGN_OR_RETURN(std::string fmha_prefix,
                      GetFMHAInstructionPrefix(fmha->custom_call_target()));
  module->SetAndUniquifyInstrName(fmha, fmha_prefix);
  return absl::OkStatus();
}
}  // namespace gpu
}  // namespace xla
