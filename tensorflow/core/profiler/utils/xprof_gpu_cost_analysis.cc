/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/xprof_gpu_cost_analysis.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/errors.h"

namespace tensorflow {
namespace profiler {

namespace {

std::vector<uint32_t> GetInputBitwidths(const xla::HloInstruction& hlo) {
  std::vector<uint32_t> input_bitwidths;
  for (const auto& operand : hlo.operands()) {
    switch (operand->shape().element_type()) {
      case xla::PRIMITIVE_TYPE_INVALID:
      case xla::TUPLE:
      case xla::OPAQUE_TYPE:
      case xla::TOKEN:
        break;
      default:
        input_bitwidths.push_back(
            xla::primitive_util::BitWidth(operand->shape().element_type()));
    }
  }
  return input_bitwidths;
}

}  // namespace

absl::Status XProfGpuCostAnalysis::HandleCustomCall(
    const xla::HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(xla::gpu::GpuHloCostAnalysis::HandleCustomCall(hlo));

  if (xla::gpu::IsCublasGemm(*hlo)) {
    // The naming conventions and meanings of gemm parameters are documented at:
    // https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api
    // as inherited from GpuHloCostAnalysis, we only normalize the flops based
    // on the datatype of A and B, which are supposed of same bitwidth.
    int dot_operands_bitwidth =
        xla::primitive_util::BitWidth(hlo->operand(0)->shape().element_type());
    uint32_t flop_rate_adjustment = 1;
    switch (dot_operands_bitwidth) {
      case 8:
        flop_rate_adjustment = 2;
        break;
      case 4:
        flop_rate_adjustment = 4;
        break;
      default:
        break;
    }
    float model_flops = current_properties_[kFlopsKey];
    current_properties_[kDeviceFlopsAdjustment] =
        model_flops - model_flops / flop_rate_adjustment;
  }
  return absl::OkStatus();
}

absl::Status XProfGpuCostAnalysis::DefaultPostprocess(
    const xla::HloInstruction* hlo) {
  uint32_t flop_rate_adjustment = 1;
  float model_flops = current_properties_[kFlopsKey];

  // Calculate adjustment of device flops based on input bit widths.
  // This provide most general adjustment for all ops, and for all gpus.
  std::vector<uint32_t> input_bitwidths = GetInputBitwidths(*hlo);
  if (!input_bitwidths.empty()) {
    int max_input_bitwidth =
        *std::max_element(input_bitwidths.begin(), input_bitwidths.end());
    if (model_flops) {
      // for int8/fp8, 2x flops assumed comparing with fp16 flops(most of
      // recent GPU models); for int4, 4x of model flops assumed comparing
      // with fp16 flops. (like Nvidia T4, 3090). It will be more precise
      // after adjustment based on specific GPUs mentioned above.
      switch (max_input_bitwidth) {
        case 8:
          flop_rate_adjustment = 2;
          break;
        case 4:
          flop_rate_adjustment = 4;
          break;
        default:
          break;
      }
    }
  }
  current_properties_[kDeviceFlopsAdjustment] =
      model_flops - model_flops / flop_rate_adjustment;
  return absl::OkStatus();
}

absl::Status XProfGpuCostAnalysis::Postprocess(const xla::HloInstruction* hlo) {
  if (hlo == nullptr) {
    return absl::OkStatus();
  }

  switch (hlo->opcode()) {
    case xla::HloOpcode::kCustomCall:
      // Already handled specially in HandleCustomCall(), skip here.
      // Add more OpCode here if it is handled specially in future.
      break;
    default:
      DefaultPostprocess(hlo).IgnoreError();
      break;
  }

  return xla::gpu::GpuHloCostAnalysis::Postprocess(hlo);
}

std::unique_ptr<xla::HloCostAnalysis>
XProfGpuCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<XProfGpuCostAnalysis>(options_);
}

int64_t XProfGpuCostAnalysis::GetDeviceFlopsAdjustment(
    const xla::HloInstruction& hlo) {
  return GetPropertyForHlo(hlo, kDeviceFlopsAdjustment, hlo_properties_);
}

}  // namespace profiler
}  // namespace tensorflow
