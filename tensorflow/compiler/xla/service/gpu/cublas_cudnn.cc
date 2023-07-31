/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"

#include <string>

#include "absl/strings/string_view.h"

namespace xla {
namespace gpu {

bool IsCublasGemm(const HloInstruction& hlo) {
  return IsLegacyCublasMatmul(hlo) || IsCublasLtMatmul(hlo);
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

// fMHA forward call targets.
const absl::string_view kCudnnfMHABmmBmmCallTarget = "__cudnn$fhmaBmmBmm";
const absl::string_view kCudnnfMHASoftmaxCallTarget = "__cudnn$fhmaSoftmax";
const absl::string_view kCudnnfMHAScaleBiasMaskSoftmaxCallTarget =
    "__cudnn$fhmaScaleBiasMaskSoftmax";
const absl::string_view kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget =
    "__cudnn$fhmaScaleBiasMaskSoftmaxDropout";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget =
    "__cudnn$fhmaScaleBiasSoftmaxDropout";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxCallTarget =
    "__cudnn$fhmaScaleBiasSoftmax";
const absl::string_view kCudnnfMHAScaleMaskSoftmaxCallTarget =
    "__cudnn$fhmaScaleMaskSoftmax";
const absl::string_view kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget =
    "__cudnn$fhmaScaleMaskSoftmaxDropout";
const absl::string_view kCudnnfMHASoftmaxDropoutCallTarget =
    "__cudnn$fhmaSoftmaxDropout";

// fMHA backward call targets.
const absl::string_view kCudnnfMHABmmBmmBackwardCallTarget =
    "__cudnn$fhmaBmmBmmBackward";
const absl::string_view kCudnnfMHASoftmaxBackwardCallTarget =
    "__cudnn$fhmaSoftmaxBackward";
const absl::string_view kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget =
    "__cudnn$fhmaScaleBiasMaskSoftmaxBackward";
const absl::string_view
    kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget =
        "__cudnn$fhmaScaleBiasMaskSoftmaxDropoutBackward";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget =
    "__cudnn$fhmaScaleBiasSoftmaxDropoutBackward";
const absl::string_view kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget =
    "__cudnn$fhmaScaleBiasSoftmaxBackward";
const absl::string_view kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget =
    "__cudnn$fhmaScaleMaskSoftmaxBackward";
const absl::string_view kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget =
    "__cudnn$fhmaScaleMaskSoftmaxDropoutBackward";
const absl::string_view kCudnnfMHASoftmaxDropoutBackwardCallTarget =
    "__cudnn$fhmaSoftmaxDropoutBackward";

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

bool IsFwdCustomCallTofMHA(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnfMHABmmBmmCallTarget ||
         target == kCudnnfMHASoftmaxCallTarget ||
         target == kCudnnfMHAScaleBiasMaskSoftmaxCallTarget ||
         target == kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget ||
         target == kCudnnfMHAScaleMaskSoftmaxCallTarget ||
         target == kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget ||
         target == kCudnnfMHASoftmaxDropoutCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget;
}

bool IsBwdCustomCallTofMHA(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnfMHABmmBmmBackwardCallTarget ||
         target == kCudnnfMHASoftmaxBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget ||
         target == kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget ||
         target == kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget ||
         target == kCudnnfMHASoftmaxDropoutBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget ||
         target == kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
}

bool MHACallHasDropout(const absl::string_view fmha_call_name) {
  return fmha_call_name == kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget ||
         fmha_call_name == kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget ||
         fmha_call_name == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget ||
         fmha_call_name ==
             kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget ||
         fmha_call_name ==
             kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget ||
         fmha_call_name == kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget;
}

bool IsCustomCallTofMHA(const HloInstruction& hlo) {
  return (IsFwdCustomCallTofMHA(hlo) || IsBwdCustomCallTofMHA(hlo));
}

StatusOr<CudnnConvKind> GetCudnnConvKind(
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
  return InternalError("Unexpected call target: %s", target);
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

StatusOr<CudnnfMHAKind> GetCudnnfMHAKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnfMHABmmBmmCallTarget) return CudnnfMHAKind::kBmmBmm;
  if (target == kCudnnfMHAScaleBiasMaskSoftmaxCallTarget)
    return CudnnfMHAKind::kScaleBiasMaskSoftmax;
  if (target == kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout;
  if (target == kCudnnfMHAScaleMaskSoftmaxCallTarget)
    return CudnnfMHAKind::kScaleMaskSoftmax;
  if (target == kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kScaleMaskSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxCallTarget) return CudnnfMHAKind::kSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxCallTarget)
    return CudnnfMHAKind::kScaleBiasSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget)
    return CudnnfMHAKind::kScaleBiasSoftmaxDropout;
  // backward
  if (target == kCudnnfMHABmmBmmBackwardCallTarget)
    return CudnnfMHAKind::kBackwardBmmBmm;
  if (target == kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasMaskSoftmax;
  if (target == kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasMaskSoftmaxDropout;
  if (target == kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleMaskSoftmax;
  if (target == kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleMaskSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardSoftmaxDropout;
  if (target == kCudnnfMHASoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasSoftmax;
  if (target == kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget)
    return CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout;
  return InternalError("Unexpected call target: %s", target);
}

std::string CudnnfMHAKindToString(CudnnfMHAKind kind) {
  switch (kind) {
    case CudnnfMHAKind::kBmmBmm:
      return "fused_batched_matmuls";
    case CudnnfMHAKind::kSoftmax:
      return "fmha_softmax";
    case CudnnfMHAKind::kSoftmaxDropout:
      return "fmha_softmax_with_dropout";
    case CudnnfMHAKind::kScaleMaskSoftmax:
      return "fmha_scaled_masked_softmax";
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      return "fmha_scaled_masked_softmax_with_dropout";
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
      return "fmha_scaled_bias_masked_softmax";
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      return "fmha_scaled_bias_masked_softmax_with_dropout";
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      return "fmha_bias_softmax_with_dropout";
    case CudnnfMHAKind::kScaleBiasSoftmax:
      return "fmha_bias_softmax";
    // backward
    case CudnnfMHAKind::kBackwardBmmBmm:
      return "fused_batched_matmuls_backward";
    case CudnnfMHAKind::kBackwardSoftmax:
      return "fmha_softmax_backward";
    case CudnnfMHAKind::kBackwardSoftmaxDropout:
      return "fmha_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleMaskSoftmax:
      return "fmha_scaled_masked_softmax_backward";
    case CudnnfMHAKind::kBackwardScaleMaskSoftmaxDropout:
      return "fmha_scaled_masked_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleBiasMaskSoftmax:
      return "fmha_scaled_bias_masked_softmax_backward";
    case CudnnfMHAKind::kBackwardScaleBiasMaskSoftmaxDropout:
      return "fmha_scaled_bias_masked_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout:
      return "fmha_bias_softmax_with_dropout_backward";
    case CudnnfMHAKind::kBackwardScaleBiasSoftmax:
      return "fmha_bias_softmax_backward";
  }
}

StatusOr<std::string> GetFMHAInstructionPrefix(
    const std::string& custom_call_target) {
  if (custom_call_target == kCudnnfMHABmmBmmCallTarget) {
    return "fmha-bmm-bmm";
  }
  if (custom_call_target == kCudnnfMHASoftmaxDropoutCallTarget) {
    return "fmha-bmm-softmax-dropout-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleMaskSoftmaxCallTarget) {
    return "fmha-bmm-scale-mask-softmax-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget) {
    return "fmha-bmm-scale-mask-softmax-dropout-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasMaskSoftmaxCallTarget) {
    return "fmha-bmm-scale-bias-mask-softmax-bmm";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget) {
    return "fmha-bmm-scale-bias-mask-softmax-dropout-bmm";
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
  if (custom_call_target == kCudnnfMHABmmBmmBackwardCallTarget) {
    return "fmha-bmm-bmm-backward";
  }
  if (custom_call_target == kCudnnfMHASoftmaxDropoutBackwardCallTarget) {
    return "fmha-bmm-softmax-dropout-bmm-backward";
  }
  if (custom_call_target == kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget) {
    return "fmha-bmm-scale-mask-softmax-bmm-backward";
  }
  if (custom_call_target ==
      kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget) {
    return "fmha-bmm-scale-mask-softmax-dropout-bmm-backward";
  }
  if (custom_call_target == kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget) {
    return "fmha-bmm-scale-bias-mask-softmax-bmm-backward";
  }
  if (custom_call_target ==
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget) {
    return "fmha-bmm-scale-bias-mask-softmax-dropout-bmm-backward";
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
  return InternalError("Unexpected call target: %s", custom_call_target);
}

// Give fmha instruction a more useful name than "custom-call.42".
Status SetFMHAInstructionName(HloModule* module, HloInstruction* fmha) {
  TF_ASSIGN_OR_RETURN(std::string fmha_prefix,
                      GetFMHAInstructionPrefix(fmha->custom_call_target()));
  module->SetAndUniquifyInstrName(fmha, fmha_prefix);
  return OkStatus();
}
}  // namespace gpu
}  // namespace xla
