/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/conv_kind_assignment.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::Status CheckTypes(HloInstruction* conv, const se::GpuComputeCapability cc,
                        const se::dnn::VersionInfo dnn_version) {
  auto valid_shape = [conv, &cc,
                      &dnn_version](const Shape& shape) -> absl::Status {
    PrimitiveType type = shape.element_type();
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsIntegralType(type)) {
      return Unimplemented(
          "Convolutions must have floating-point or integral operands/outputs, "
          "but got convolution with type %s: %s",
          primitive_util::LowercasePrimitiveTypeName(type), conv->ToString());
    }
    if (primitive_util::IsF8Type(type)) {
      if (type != F8E4M3FN && type != F8E5M2) {
        return Unimplemented(
            "The only FP8 types supported in convolutions are f8e5m2 and "
            "f8e4m3, "
            "but got convolution with FP8 type %s: %s",
            primitive_util::LowercasePrimitiveTypeName(type), conv->ToString());
      }
      if (!cc.IsCuda()) {
        return Unimplemented(
            "FP8 convolutions are only supported on CUDA GPUs, but got "
            "FP8 convolution on ROCm GPU: %s",
            conv->ToString());
      }
      if (dnn_version >= se::dnn::VersionInfo{9, 8, 0}) {
        if (!cc.cuda_compute_capability()->IsAtLeastAda()) {
          return Unimplemented(
              "FP8 convolutions are only supported on CUDA GPUs with compute "
              "capability at least 8.9, but got "
              "FP8 convolution on GPU with compute capability %s: %s",
              cc.ToString(), conv->ToString());
        }
      } else if (!cc.cuda_compute_capability()->IsAtLeastHopper()) {
        return Unimplemented(
            "FP8 convolutions are only supported on CUDA GPUs with compute "
            "capability at least 9.0, but got "
            "FP8 convolution on GPU with compute capability %s: %s",
            cc.ToString(), conv->ToString());
      }
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(valid_shape(conv->shape()));
  ABSL_RETURN_IF_ERROR(valid_shape(conv->operand(0)->shape()));
  ABSL_RETURN_IF_ERROR(valid_shape(conv->operand(1)->shape()));
  return absl::OkStatus();
}

// Try to match the backward filter pattern of a transposed (i.e. base-dilated)
// convolution:
//
//   dw = conv(x, dy), window={size=Y pad=lo_hi lhs_dilate=d}
//
// The convolution is equivalent to WGRAD backward convolution:
//
//   reverse(dw) = conv(dy, x), window={size=X pad=(F-1-lo)_(F-1-hi)
//   rhs_dilate=d}
//
// where F is the filter size (output of `conv`) and X the spatial size of `x`.
HloInstruction* MatchTransposedConvBackwardFilter(HloInstruction* conv) {
  const Window& window = conv->window();
  if (!window_util::HasBaseDilation(window) || conv->operand_count() != 2 ||
      conv->feature_group_count() != 1 || conv->batch_group_count() != 1) {
    return nullptr;
  }
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();

  Window swapped_window;
  for (int i = 0; i < window.dimensions_size(); ++i) {
    const WindowDimension& dim = window.dimensions(i);
    const int64_t filter_size =
        conv->shape().dimensions(dnums.output_spatial_dimensions(i));
    WindowDimension* swapped_dim = swapped_window.add_dimensions();
    swapped_dim->set_size(conv->operand(0)->shape().dimensions(
        dnums.input_spatial_dimensions(i)));
    swapped_dim->set_stride(1);
    swapped_dim->set_padding_low(filter_size - 1 - dim.padding_low());
    swapped_dim->set_padding_high(filter_size - 1 - dim.padding_high());
    swapped_dim->set_base_dilation(1);
    swapped_dim->set_window_dilation(dim.base_dilation());
    if (dim.stride() != 1 || dim.window_dilation() != 1 ||
        dim.window_reversal() || swapped_dim->padding_low() < 0 ||
        swapped_dim->padding_high() < 0) {
      return nullptr;
    }
  }

  // Swap the roles of the operands; the output keeps its shape, but its batch
  // and feature dimensions swap roles.
  ConvolutionDimensionNumbers swapped_dnums = dnums;
  swapped_dnums.set_input_batch_dimension(
      dnums.kernel_output_feature_dimension());
  swapped_dnums.set_input_feature_dimension(
      dnums.kernel_input_feature_dimension());
  *swapped_dnums.mutable_input_spatial_dimensions() =
      dnums.kernel_spatial_dimensions();
  swapped_dnums.set_kernel_input_feature_dimension(
      dnums.input_feature_dimension());
  swapped_dnums.set_kernel_output_feature_dimension(
      dnums.input_batch_dimension());
  *swapped_dnums.mutable_kernel_spatial_dimensions() =
      dnums.input_spatial_dimensions();
  swapped_dnums.set_output_batch_dimension(dnums.output_feature_dimension());
  swapped_dnums.set_output_feature_dimension(dnums.output_batch_dimension());

  PrecisionConfig swapped_precision_config = conv->precision_config();
  if (swapped_precision_config.operand_precision_size() == 2) {
    swapped_precision_config.mutable_operand_precision()->SwapElements(0, 1);
  }

  std::unique_ptr<HloInstruction> swapped_conv = HloInstruction::CreateConvolve(
      conv->shape(), {conv->mutable_operand(1), conv->mutable_operand(0)},
      /*feature_group_count=*/1, /*batch_group_count=*/1, swapped_window,
      swapped_dnums, swapped_precision_config);
  // Reuse the regular backward filter heuristics on the swapped convolution.
  if (!MatchBackwardFilter(swapped_conv.get())) {
    return nullptr;
  }
  Cast<HloConvolutionInstruction>(swapped_conv.get())
      ->set_convolution_kind(CONVOLUTION_KIND_WGRAD);

  HloComputation* computation = conv->parent();
  return computation->AddInstruction(
      HloInstruction::CreateReverse(
          conv->shape(),
          computation->AddInstruction(std::move(swapped_conv),
                                      &conv->metadata()),
          dnums.output_spatial_dimensions()),
      &conv->metadata());
}

HloInstruction* CreateGpuConv(ConvolutionKind convolution_kind,
                              HloInstruction* conv, HloInstruction* lhs,
                              HloInstruction* rhs) {
  HloInstruction* cloned_conv = conv->parent()->AddInstruction(
      conv->CloneWithNewOperands(conv->shape(), {lhs, rhs}));
  DynCast<HloConvolutionInstruction>(cloned_conv)
      ->set_convolution_kind(convolution_kind);
  return cloned_conv;
}

absl::StatusOr<HloInstruction*> AssignConvKind(
    HloInstruction* conv, const se::GpuComputeCapability& cc,
    const se::dnn::VersionInfo& dnn_version) {
  ABSL_RETURN_IF_ERROR(CheckTypes(conv, cc, dnn_version));

  PrimitiveType element_type = conv->shape().element_type();
  // cuDNN graph API does not support f64 convolutions.
  if (element_type == F64) {
    return nullptr;
  }
  // cuDNN graph API does not support disabling TF32 for f32 convolutions.
  if (element_type == F32) {
    bool is_highest_precision = !absl::c_all_of(
        conv->precision_config().operand_precision(),
        [](int precision) { return precision <= PrecisionConfig::HIGH; });
    if (is_highest_precision) {
      return nullptr;
    }
  }

  // cuDNN graph API does not have engine execution plans for INT8 convolutions
  // on pre-Ampere GPUs.
  if ((primitive_util::Is8BitIntegralType(element_type) ||
       primitive_util::Is8BitIntegralType(
           conv->operand(0)->shape().element_type())) &&
      cc.cuda_compute_capability() != nullptr &&
      !cc.cuda_compute_capability()->IsAtLeastAmpere()) {
    return nullptr;
  }

  if (ConvolutionMatch m = MatchBackwardInput(conv)) {
    conv = CreateGpuConv(CONVOLUTION_KIND_DGRAD, conv, conv->mutable_operand(0),
                         std::get<2>(*m));
  } else if (ConvolutionMatch m = MatchBackwardFilter(conv)) {
    conv = CreateGpuConv(CONVOLUTION_KIND_WGRAD, conv, std::get<2>(*m),
                         conv->mutable_operand(1));
  } else if (HloInstruction* reversed_wgrad =
                 MatchTransposedConvBackwardFilter(conv)) {
    conv = reversed_wgrad;
  } else if (CanImplementAsGpuForwardConv(conv)) {
    // If all else fails, try a forward convolution.
    if (conv->batch_group_count() > 1) {
      conv = ConvertBatchGroupedToFeatureGroupedConvolution(conv);
      Cast<HloConvolutionInstruction>(conv)->set_convolution_kind(
          CONVOLUTION_KIND_FPROP);
    } else {
      conv = CreateGpuConv(CONVOLUTION_KIND_FPROP, conv,
                           conv->mutable_operand(0), conv->mutable_operand(1));
    }
  } else {
    return nullptr;
  }
  return conv;
}

// Tries to assign conv kind(fprop/dgrad/wgrad) to conv instruction.
absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv,
                                      const se::GpuComputeCapability& cc,
                                      const se::dnn::VersionInfo& dnn_version) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  ABSL_ASSIGN_OR_RETURN(HloInstruction * conv_with_kind,
                   AssignConvKind(conv, cc, dnn_version));
  if (conv_with_kind == nullptr || conv_with_kind == conv) {
    return false;
  }

  if (auto* new_conv = DynCast<HloConvolutionInstruction>(conv_with_kind);
      new_conv != nullptr &&
      new_conv->convolution_kind() == CONVOLUTION_KIND_UNSET) {
    return false;
  }

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << conv_with_kind->ToString();
  ABSL_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(conv, conv_with_kind));
  return true;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      const se::GpuComputeCapability& cc,
                                      const se::dnn::VersionInfo dnn_version) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kConvolution>(hlo)) {
      ConvolutionKind convolution_kind =
          DynCast<HloConvolutionInstruction>(hlo)->convolution_kind();
      if (convolution_kind == CONVOLUTION_KIND_UNSET) {
        convs.push_back(hlo);
      }
    }
  }

  bool changed = false;
  for (HloInstruction* conv : convs) {
    ABSL_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv, cc, dnn_version));
    changed |= result;
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> ConvKindAssignment::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2,
                 "ConvKindAssignment::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(
        bool result,
        RunOnComputation(computation, compute_capability_, dnn_version_));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvKindAssignment::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
