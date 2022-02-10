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

namespace xla {
namespace gpu {

bool IsCublasGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kGemmCallTarget;
}

const char* const kGemmCallTarget = "__cublas$gemm";
const char* const kTriangularSolveCallTarget = "__cublas$triangularSolve";
const char* const kCudnnConvForwardCallTarget = "__cudnn$convForward";
const char* const kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const char* const kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const char* const kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";

bool IsCustomCallToDnnConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget ||
         target == kCudnnConvBiasActivationForwardCallTarget;
}

StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    return CudnnConvKind::kForward;
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
  }
}

}  // namespace gpu
}  // namespace xla
