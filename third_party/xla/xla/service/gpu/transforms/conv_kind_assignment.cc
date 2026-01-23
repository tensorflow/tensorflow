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

#include "xla/service/gpu/transforms/conv_kind_assignment.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
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

  TF_RETURN_IF_ERROR(valid_shape(conv->shape()));
  TF_RETURN_IF_ERROR(valid_shape(conv->operand(0)->shape()));
  TF_RETURN_IF_ERROR(valid_shape(conv->operand(1)->shape()));
  return absl::OkStatus();
}

using ConvolutionMatch = std::optional<HloInstruction*>;
using ConvKind = HloConvolutionInstruction::ConvKind;

// Determine whether conv2d is equal to conv1d.
bool MaybeConv1dToConv2d(HloInstruction* conv) {
  if (conv->window().dimensions().size() != 2) {
    return false;
  }
  if (conv->operand(1)->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto filter = conv->operand(1);
  std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
      filter->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
  if (reshape_degenerate.has_value() &&
      reshape_degenerate->deleted_dimensions.empty() &&
      reshape_degenerate->inserted_dimensions.size() == 1) {
    const auto& dnums = conv->convolution_dimension_numbers();
    for (auto dim : dnums.kernel_spatial_dimensions()) {
      if (dim == reshape_degenerate->inserted_dimensions[0]) {
        return true;
      }
    }
  }
  return false;
}

bool LooksLikeForwardConvolution(const HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  const Shape& lhs_shape = conv->operand(0)->shape();
  const Shape& rhs_shape = conv->operand(1)->shape();
  const Shape& result_shape = conv->shape();

  // Compare batch and output feature counts. Backward-filter convolutions swap
  // these, so matching values are a strong signal that this is a forward
  // convolution, even if it has dilation.
  int64_t lhs_batches = lhs_shape.dimensions(dnums.input_batch_dimension());
  int64_t result_batches =
      result_shape.dimensions(dnums.output_batch_dimension());
  if (lhs_batches != result_batches) {
    return false;
  }

  int64_t rhs_output_features =
      rhs_shape.dimensions(dnums.kernel_output_feature_dimension());
  int64_t result_output_features =
      result_shape.dimensions(dnums.output_feature_dimension());
  if (rhs_output_features != result_output_features) {
    return false;
  }

  for (int i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    int64_t kdim = rhs_shape.dimensions(dnums.kernel_spatial_dimensions(i));
    int64_t odim = result_shape.dimensions(dnums.output_spatial_dimensions(i));
    if (kdim > odim) {
      return false;
    }
  }

  return true;
}

bool CanImplementAsGpuForwardConv(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  if (dnums.input_spatial_dimensions_size() > 3) {
    return false;
  }

  // CuDNN does not accept zero-element arguments
  if (ShapeUtil::IsZeroElementArray(conv->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(conv->operand(1)->shape())) {
    return false;
  }

  // CuDNN can perform either cross correlation (no reversal),
  // or convolution (all dimensions reversed).
  if (dnums.input_spatial_dimensions_size() == 2
          ? !window_util::AllOrNoneReversed(conv->window())
          : window_util::HasWindowReversal(conv->window())) {
    return false;
  }
  return true;
}

// Try to match a backward filter pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
ConvolutionMatch MatchBackwardFilter(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward filter.";

  if (conv->feature_group_count() > 1) {
    VLOG(1) << conv->ToString()
            << " is a forward convolution. All grouped backward filters are "
               "mapped to batch grouped convolutions in tf2xla bridge. Hence "
               "backward filter "
               "convolutions cannot have feature groups greater than 1 at this "
               "point. No need to fold to backward filter.";
    return std::nullopt;
  }

  // Step 1: match the instruction pattern without considering the paddings and
  // dimension numbers just yet. We may need some generic pattern matcher
  // similar to third_party/llvm/llvm/include/llvm/IR/PatternMatch.h
  //
  // Backward filter convolution is implemented in XLA as the forward
  // convolution of padded activations and dilated gradients. Padding on
  // activations and dilation on gradients are specified in the "window" field
  // of the forward convolution.
  //
  //        activations  gradients
  //              \         /
  //               v       v
  //              Convolution
  //                 conv
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  if (LooksLikeForwardConvolution(conv)) {
    VLOG(1) << "Convolution " << conv->ToString()
            << " looks like a forward convolution; skipping backward filter "
               "rewrite.";
    return std::nullopt;
  }

  // Step 2: match paddings and dimension numbers of the forward convolution.
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return std::nullopt;
    }
    if (window_dim.base_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no base (LHS) dilation.";
      return std::nullopt;
    }
    if (window_dim.padding_low() < 0) {
      VLOG(1) << "Padding low should be non-negative.";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return std::nullopt;
    }
    // Padding high will be checked in Step 3.
  }
  // Mathematically, there is no difference between convolution forward vs
  // backward filter. A backward filter:
  //   [N, O, H+h-1, W+w-1] x [N, C, H, W] -> [O, C, h, w]
  // Can be treated as a forward convolution with `N` treated as the new
  // contracting (feature) dimension, `O` treated as the new batch dimension,
  // and `C` treated as the new output feature dimension. The only difference is
  // layouts and performance.
  //
  // Since there is no way to precisely tell whether we want a foward conv or
  // backward filter conv, we have to rely on heuristics. Empirically forward
  // convolutions have very small kernel dimensions, while in the backward pass
  // "kernel dimensions" are large. If kernel dimensions are smaller than the
  // output dimensions, return foward conv; otherwise proceed with backward
  // filter conv. But for conv1d, it is not same. Due to conv1d always reshape
  // 1D-filter to 2D-filter, even backward or forward will exist one small
  // kernel dimension. We should handle this special case.
  int small_kernel_dimension_num = 0;
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    if (conv->operand(1)->shape().dimensions(kernel_spatial_dims[i]) <=
        conv->shape().dimensions(output_spatial_dims[i])) {
      small_kernel_dimension_num += 1;
    }
  }
  if ((kernel_spatial_dims.empty() || small_kernel_dimension_num > 1 ||
       (!MaybeConv1dToConv2d(conv) && small_kernel_dimension_num == 1)) &&
      !window_util::HasWindowDilation(conv->window())) {
    VLOG(1) << conv->ToString()
            << " is a regular forward convolution. No need "
               "to fold it to a backward filter convolution....";
    return std::nullopt;
  }

  // trying computing the window of the backward convolution.
  auto backward_conv_window =
      RestoreWindowFromBackwardFilter(DynCast<HloConvolutionInstruction>(conv));
  if (!backward_conv_window) {
    return std::nullopt;
  }

  return conv->mutable_operand(0);
}

// Try to match a backward input pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
ConvolutionMatch MatchBackwardInput(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward input.";

  // TODO(timshen) Theoretically cuDNN supports grouped convolutions also
  // for the backward input convolution, but based on the cudnn's current state
  // there is not much performance improvement when using the
  // cudnn backward input API for grouped conv.
  // This needs to be re-evaluated for future cuDNN versions.
  // Note that we already have the necessary code down below, the only thing to
  // enable it is to remove the following early return.
  if (conv->feature_group_count() > 1) {
    return std::nullopt;
  }

  // Match instruction pattern.
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  HloInstruction* reverse_filter = conv->mutable_operand(1);
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();

  // Match BackwardInput for a depthwise convolution and thunk it to forward
  // convolution Output feature dimension and input feature dimension has been
  // swapped in the bridge. Hence to get the actual input features we need to
  // query the output feature dimension
  auto kernel_out_feature_dim = dnums.kernel_output_feature_dimension();
  auto kernel_out_features =
      reverse_filter->shape().dimensions(kernel_out_feature_dim);

  // For a depthwise convolution, the input features must be equal to the
  // feature_group_count. We can leverage this property to match a depthwise
  // convolution and thunk it to forward conv
  if (conv->feature_group_count() > 1 &&
      kernel_out_features == conv->feature_group_count()) {
    return std::nullopt;
  }

  // We pattern-match to a backwards input conv if:
  //
  //  - all spatial dims of the filter are reversed
  //
  // OR
  //
  //  - filter is 1x1 or a constant AND
  //  - conv has base dilation (otherwise this is just a regular forward conv).
  //
  // The final criterion above is just for canonicalization; cudnn seems to run
  // just as fast if we canonicalize 1x1/constant filters without base dilation
  // to forward or backward convs.  We canonicalize to forward conv because (a)
  // it's more natural (constant filters usually show up when doing inference,
  // and having backwards convolutions in inference graphs would be weird), and
  // (b) cudnn has special fusions for forward conv plus bias and activation.
  bool is_reversed_filter =
      HloPredicateIsOp<HloOpcode::kReverse>(reverse_filter) &&
      absl::c_is_permutation(dnums.kernel_spatial_dimensions(),
                             reverse_filter->dimensions());
  // For conv1d which reshape to conv2d, filter reverse pattern is
  // reshape(reverse(filter)). It seems we can reuse conv2d backward input
  // pattern matcher, but after algsimp pass, this pattern will change to
  // reverse(reshape(filter)) and fail to match. So matching conv1d backward
  // input need different processing logic.
  bool is_reversed_conv1d_filter =
      MaybeConv1dToConv2d(conv) &&
      reverse_filter->operand(0)->opcode() == HloOpcode::kReverse;
  bool is_1x1_filter =
      absl::c_all_of(conv->window().dimensions(),
                     [](const WindowDimension& d) { return d.size() == 1; });
  if (!is_reversed_filter && !is_reversed_conv1d_filter &&
      !(window_util::HasBaseDilation(conv->window()) &&
        (reverse_filter->IsConstant() || is_1x1_filter))) {
    VLOG(1) << "Can't match to backwards convolution. Either filter is not "
               "kReverse, or it's not a base-dilated conv with a 1x1 or "
               "constant filter.";
    return std::nullopt;
  }

  // Match padding and dilation of the forward convolution.
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return std::nullopt;
    }
    if (window_dim.window_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no window dilation.";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return std::nullopt;
    }
  }

  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();
  CHECK_EQ(conv->window().dimensions().size(), input_spatial_dims.size());
  CHECK_EQ(output_spatial_dims.size(), input_spatial_dims.size());

  auto backward_conv_window =
      RestoreWindowFromBackwardInput(DynCast<HloConvolutionInstruction>(conv));
  if (!backward_conv_window) {
    return std::nullopt;
  }

  // If we matched against a constant, we need to add a reverse op that can be
  // subsumed by the cuDNN call. algebraic-simplifier will later remove any
  // unnecessary reverses.
  if (HloPredicateIsNotOp<HloOpcode::kReverse>(reverse_filter) &&
      reverse_filter->IsConstant()) {
    // Create a double-reverse, which is a nop.
    HloComputation* c = conv->parent();
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    TF_CHECK_OK(conv->ReplaceOperandWith(/*operand_num=*/1, reverse_filter));
  }

  // Calculate the 'rhs' that goes into the backward input convolution.
  HloInstruction* rhs = reverse_filter;
  // One reverse is subsumed by the cuDNN call.
  if (HloPredicateIsOp<HloOpcode::kReverse>(rhs)) {
    rhs = rhs->mutable_operand(0);
  } else if (is_reversed_conv1d_filter) {
    auto src = rhs->mutable_operand(0)->mutable_operand(0);
    rhs = conv->parent()->AddInstruction(
        HloInstruction::CreateReshape(rhs->shape(), src));
  }
  return rhs;
}

HloInstruction* CreateGpuConv(ConvKind conv_kind, HloInstruction* conv,
                              HloInstruction* lhs, HloInstruction* rhs) {
  HloInstruction* cloned_conv = conv->parent()->AddInstruction(
      conv->CloneWithNewOperands(conv->shape(), {lhs, rhs}));
  DynCast<HloConvolutionInstruction>(cloned_conv)->set_conv_kind(conv_kind);
  return cloned_conv;
}

HloInstruction* ConvertBatchGroupedToFeatureGroupedConvolution(
    HloInstruction* conv) {
  CHECK_EQ(conv->feature_group_count(), 1);
  int64_t num_groups = conv->batch_group_count();
  auto dim_numbers = conv->convolution_dimension_numbers();
  auto lhs = conv->mutable_operand(0);
  auto rhs = conv->mutable_operand(1);

  int64_t input_batch_dimension = dim_numbers.input_batch_dimension();

  Shape output_shape = conv->shape();
  int64_t input_feature_dimension = dim_numbers.input_feature_dimension();
  int64_t input_feature = lhs->shape().dimensions(input_feature_dimension);

  HloComputation* computation = lhs->parent();
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation->AddInstruction(std::move(inst));
  };
  // Reshape batch_dim N -> [G, N/G]
  std::vector<int64_t> reshape_dims = SpanToVector(lhs->shape().dimensions());
  reshape_dims[input_batch_dimension] =
      reshape_dims[input_batch_dimension] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + input_batch_dimension, num_groups);
  lhs = add(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(lhs->shape().element_type(), reshape_dims), lhs));

  // Transpose G to the axis before C, For eg: [G, N/G, H, W, C ] -> [N/G, H,
  // W, G, C]
  std::vector<int64_t> transpose_dims(lhs->shape().dimensions().size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  transpose_dims.erase(transpose_dims.begin() + input_batch_dimension);
  transpose_dims.insert(transpose_dims.begin() + input_feature_dimension,
                        input_batch_dimension);
  std::vector<int64_t> transpose_reshape_dims =
      ComposePermutations(lhs->shape().dimensions(), transpose_dims);
  lhs = add(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(lhs->shape().element_type(), transpose_reshape_dims),
      lhs, transpose_dims));

  // Merge [G,C] -> [C*G]
  Shape new_shape = lhs->shape();
  new_shape.DeleteDimension(input_feature_dimension);
  new_shape.set_dimensions(input_feature_dimension, input_feature * num_groups);
  lhs = add(HloInstruction::CreateReshape(new_shape, lhs));

  std::vector<HloInstruction*> new_operands = {lhs, rhs};
  auto new_conv = conv->CloneWithNewOperands(output_shape, new_operands);
  new_conv->set_feature_group_count(num_groups);
  new_conv->set_batch_group_count(1);
  new_conv->set_convolution_dimension_numbers(dim_numbers);
  return computation->AddInstruction(std::move(new_conv));
}

static absl::StatusOr<HloInstruction*> AssignConvKind(
    HloInstruction* conv, const se::GpuComputeCapability& cc,
    const se::dnn::VersionInfo& dnn_version,
    std::vector<HloInstruction*>& fusion_outputs) {
  TF_RETURN_IF_ERROR(CheckTypes(conv, cc, dnn_version));
  if (ConvolutionMatch m = MatchBackwardInput(conv)) {
    conv = CreateGpuConv(ConvKind::DGRAD, conv, conv->mutable_operand(0), *m);
  } else if (ConvolutionMatch m = MatchBackwardFilter(conv)) {
    conv = CreateGpuConv(ConvKind::WGRAD, conv, *m, conv->mutable_operand(1));
  } else if (CanImplementAsGpuForwardConv(conv)) {
    // If all else fails, try a forward convolution.
    if (conv->batch_group_count() > 1) {
      conv = ConvertBatchGroupedToFeatureGroupedConvolution(conv);
    }
    conv = CreateGpuConv(ConvKind::FPROP, conv, conv->mutable_operand(0),
                         conv->mutable_operand(1));
  }
  return conv;
}

// Tries to rewrite convolution and fusible instructions into cudnn fusion.
absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv,
                                      const se::GpuComputeCapability& cc,
                                      const se::dnn::VersionInfo& dnn_version) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  std::vector<HloInstruction*> fusion_outputs;
  TF_ASSIGN_OR_RETURN(HloInstruction * conv_with_kind,
                      AssignConvKind(conv, cc, dnn_version, fusion_outputs));
  if (conv == nullptr) {
    return false;
  }

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << conv_with_kind->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(conv, conv_with_kind));
  return true;
}

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
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv, cc, dnn_version));
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
    TF_ASSIGN_OR_RETURN(
        bool result,
        RunOnComputation(computation, compute_capability_, dnn_version_));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvKindAssignment::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
