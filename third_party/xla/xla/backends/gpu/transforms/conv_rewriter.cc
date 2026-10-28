/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/conv_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<bool> CheckTypes(HloInstruction* conv,
                                const se::GpuComputeCapability cc,
                                const se::dnn::VersionInfo dnn_version) {
  auto valid_shape =
      [conv, &cc, &dnn_version](const Shape& shape) -> absl::StatusOr<bool> {
    PrimitiveType type = shape.element_type();
    // oneDNN integer convolutions are not supported yet.
    if (cc.IsOneAPI() && primitive_util::IsIntegralType(type)) {
      return false;
    }
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsIntegralType(type)) {
      // Among integral types, only S8 is supported. But CudnnFusedConvRewriter
      // may rewrite convolutions of wider types into S8 convolutions, so allow
      // all integral convolutions here.
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
            "FP8 convolution on ROCm/Intel GPU: %s",
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
    return true;
  };

  for (const Shape* shape : {&conv->shape(), &conv->operand(0)->shape(),
                             &conv->operand(1)->shape()}) {
    ABSL_ASSIGN_OR_RETURN(bool is_supported, valid_shape(*shape));
    if (!is_supported) {
      return false;
    }
  }
  return true;
}

HloInstruction* CreateGpuConv(absl::string_view call_target, const Shape& shape,
                              HloInstruction* lhs, HloInstruction* rhs,
                              const Window& window,
                              const ConvolutionDimensionNumbers& dnums,
                              int64_t feature_group_count,
                              const PrecisionConfig& precision_config,
                              const OpMetadata& metadata) {
  HloComputation* computation = lhs->parent();

  // This call returns a tuple of (conv_result, scratch_memory), where
  // conv_result is the actual result of the convolution, and scratch_memory is
  // temporary memory used by cudnn.
  //
  // At the moment, we don't know how much scratch memory this conv is going to
  // use, so we put u8[0] in this place.  Later on another pass will choose
  // which conv algorithm to use, and at that point we'll modify the shape of
  // this second tuple element.
  Shape call_shape =
      ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U8, {0})});

  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(call_shape, {lhs, rhs}, call_target));
  custom_call->set_window(window);
  custom_call->set_convolution_dimension_numbers(dnums);
  custom_call->set_feature_group_count(feature_group_count);
  *custom_call->mutable_precision_config() = precision_config;
  custom_call->set_metadata(metadata);

  // Give the customcall a user-friendly name.
  std::optional<std::string> name;
  if (call_target == kCudnnConvForwardCallTarget) {
    name = "cudnn-conv";
  } else if (call_target == kCudnnConvBackwardInputCallTarget) {
    name = "cudnn-conv-bw-input";
  } else if (call_target == kCudnnConvBackwardFilterCallTarget) {
    name = "cudnn-conv-bw-filter";
  } else if (call_target == kCudnnConvBiasActivationForwardCallTarget) {
    name = "cudnn-conv-bias-activation";
  }
  if (name.has_value()) {
    computation->parent()->SetAndUniquifyInstrName(custom_call, *name);
  }

  return custom_call;
}

CudnnConvBackendConfig GetDefaultBackendConfig() {
  CudnnConvBackendConfig config;
  config.set_conv_result_scale(1);
  return config;
}

// Helper function to create a custom_call instruction to replace the given
// conv instruction
static absl::StatusOr<HloInstruction*> CreateCustomCallHelper(
    HloInstruction* conv, const se::GpuComputeCapability& cc,
    const se::dnn::VersionInfo& dnn_version) {
  ABSL_ASSIGN_OR_RETURN(bool types_supported, CheckTypes(conv, cc, dnn_version));
  if (!types_supported) {
    return nullptr;
  }
  if (ConvolutionMatch m = MatchBackwardInput(conv)) {
    auto& [window, dnums, rhs] = *m;
    return CreateGpuConv(kCudnnConvBackwardInputCallTarget, conv->shape(),
                         conv->mutable_operand(0), rhs, window, dnums,
                         conv->feature_group_count(), conv->precision_config(),
                         conv->metadata());
  }

  if (ConvolutionMatch m = MatchBackwardFilter(conv)) {
    auto& [window, dnums, lhs] = *m;
    return CreateGpuConv(kCudnnConvBackwardFilterCallTarget, conv->shape(), lhs,
                         conv->mutable_operand(1), window, dnums,
                         conv->batch_group_count(), conv->precision_config(),
                         conv->metadata());
  }

  // If all else fails, try a forward convolution.
  if (CanImplementAsGpuForwardConv(conv, cc)) {
    if (conv->batch_group_count() > 1) {
      conv = ConvertBatchGroupedToFeatureGroupedConvolution(conv);
    }

    return CreateGpuConv(kCudnnConvForwardCallTarget, conv->shape(),
                         conv->mutable_operand(0), conv->mutable_operand(1),
                         conv->window(), conv->convolution_dimension_numbers(),
                         conv->feature_group_count(), conv->precision_config(),
                         conv->metadata());
  }

  return nullptr;
}

// Tries to rewrite a single convolution into a call to cudnn/miopen.
absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv,
                                      const se::GpuComputeCapability& cc,
                                      const se::dnn::VersionInfo& dnn_version) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);

  // If a convolution kind is already assigned (by ConvKindAssignment for cuDNN
  // graph API / ConvFusionRewriter), do not rewrite it to legacy custom calls.
  if (Cast<HloConvolutionInstruction>(conv)->convolution_kind() !=
      CONVOLUTION_KIND_UNSET) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(HloInstruction * custom_call,
                   CreateCustomCallHelper(conv, cc, dnn_version));
  if (custom_call == nullptr) {
    return false;
  }

  GpuBackendConfig gpu_backend_config;
  *gpu_backend_config.mutable_cudnn_conv_backend_config() =
      GetDefaultBackendConfig();
  ABSL_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << custom_call->ToString();

  // The CustomCall returns a tuple (conv_result, scratch_memory).  Extract
  // out the conv result and replace `conv` with it.
  ABSL_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv,
      HloInstruction::CreateGetTupleElement(conv->shape(), custom_call, 0)));
  return true;
}

// Rewrites the convolutions in the given computation into calls to
// cudnn/miopen.
// Returns true if it made any changes.
absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      const se::GpuComputeCapability& cc,
                                      const se::dnn::VersionInfo dnn_version) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kConvolution>(hlo)) {
      convs.push_back(hlo);
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

absl::StatusOr<bool> ConvRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "ConvRewriter::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(
        bool result,
        RunOnComputation(computation, compute_capability_, dnn_version_));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvRewriter::RunImpl(), after:\n" + module->ToString());
  return changed;
}

/*static*/ bool ConvRewriter::ConvIsLowerable(HloInstruction* conv) {
  return CanImplementAsGpuForwardConv(conv) || MatchBackwardFilter(conv) ||
         MatchBackwardInput(conv);
}

}  // namespace gpu
}  // namespace xla
