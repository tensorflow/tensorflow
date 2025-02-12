/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/dot_as_convolution_util.h"

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/shape_inference.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {
namespace dot_as_convolution_util {

SpatialBatchRepresentation SpatialIsBatch(int64_t lhs_spatial_size,
                                          const WindowDimension& spatial_wd) {
  if (lhs_spatial_size == spatial_wd.size() &&
      lhs_spatial_size == spatial_wd.base_dilation() &&
      ((std::max<int64_t>(1, lhs_spatial_size - 1) == spatial_wd.stride() &&
        spatial_wd.window_dilation() == 1) ||
       (std::max<int64_t>(1, lhs_spatial_size - 1) ==
            spatial_wd.window_dilation() &&
        spatial_wd.stride() == 1)) &&
      spatial_wd.padding_high() == 0 && spatial_wd.padding_low() == 0 &&
      !spatial_wd.window_reversal()) {
    return SpatialBatchRepresentation::kUnpaddedVersion;
  } else if (lhs_spatial_size == spatial_wd.size() &&
             spatial_wd.padding_high() == lhs_spatial_size - 1 &&
             spatial_wd.padding_low() == lhs_spatial_size - 1 &&
             spatial_wd.window_reversal() &&
             spatial_wd.window_dilation() == 1 &&
             spatial_wd.stride() == lhs_spatial_size &&
             spatial_wd.base_dilation() == lhs_spatial_size - 1) {
    return SpatialBatchRepresentation::kPaddedVersion;
  }
  return SpatialBatchRepresentation::kNone;
}

bool SpatialIsLhsNonContracting(int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd) {
  return spatial_wd.stride() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.base_dilation() == 1 && rhs_spatial_size == 1 &&
         spatial_wd.size() == 1 && spatial_wd.padding_high() == 0 &&
         spatial_wd.padding_low() == 0 && !spatial_wd.window_reversal();
}

bool SpatialIsRhsNonContracting(int64_t lhs_spatial_size,
                                int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd) {
  return spatial_wd.stride() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.base_dilation() == 1 && lhs_spatial_size == 1 &&
         spatial_wd.size() == rhs_spatial_size &&
         spatial_wd.padding_high() == rhs_spatial_size - 1 &&
         spatial_wd.padding_low() == rhs_spatial_size - 1 &&
         spatial_wd.window_reversal();
}

bool SpatialIsContracting(int64_t lhs_spatial_size, int64_t rhs_spatial_size,
                          const WindowDimension& spatial_wd) {
  return lhs_spatial_size == spatial_wd.size() &&
         spatial_wd.base_dilation() == 1 && spatial_wd.window_dilation() == 1 &&
         spatial_wd.padding_high() == 0 && spatial_wd.padding_low() == 0 &&
         !spatial_wd.window_reversal();
}

/* static */ DotConvolutionDimsInfo ParseConvolutionDimsInfo(
    const HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  const auto& conv_dims = conv->convolution_dimension_numbers();
  DotConvolutionDimsInfo dims;
  dims.lhs_non_contracting_dims.push_back(
      {conv_dims.input_batch_dimension(), -1,
       conv_dims.output_batch_dimension(), -1});
  dims.rhs_non_contracting_dims.push_back(
      {-1, conv_dims.kernel_output_feature_dimension(),
       conv_dims.output_feature_dimension(), -1});
  dims.contracting_dims.push_back({conv_dims.input_feature_dimension(),
                                   conv_dims.kernel_input_feature_dimension(),
                                   -1, -1});

  for (int64_t i = 0; i < conv_dims.input_spatial_dimensions_size(); ++i) {
    int64_t lhs = conv_dims.input_spatial_dimensions(i);
    int64_t lhs_size = conv->operand(0)->shape().dimensions(lhs);
    int64_t rhs = conv_dims.kernel_spatial_dimensions(i);
    int64_t rhs_size = conv->operand(1)->shape().dimensions(rhs);
    int64_t output = conv_dims.output_spatial_dimensions(i);
    const auto& wd = conv->window().dimensions(i);
    if (SpatialIsBatch(lhs_size, wd) != SpatialBatchRepresentation::kNone) {
      dims.batch_dims.push_back({lhs, rhs, output, i});
    } else if (lhs_size == wd.size() && wd.base_dilation() == 1 &&
               wd.window_dilation() == 1 && wd.padding_high() == 0 &&
               wd.padding_low() == 0 && !wd.window_reversal()) {
      // A contracting dimension be represented as a spatial dimension with
      // window size C (contracting dimension size). Stride can be any size
      // since there is only one window.
      dims.contracting_dims.push_back({lhs, rhs, output, i});
    } else if (wd.stride() == 1 && wd.window_dilation() == 1 &&
               wd.base_dilation() == 1) {
      if (rhs_size == 1 && wd.size() == 1 && wd.padding_high() == 0 &&
          wd.padding_low() == 0 && !wd.window_reversal()) {
        // A LHS non-contracting dimension can be represented as a spatial
        // dimension with window size 1.
        dims.lhs_non_contracting_dims.push_back({lhs, rhs, output, i});
      } else if (lhs_size == 1 && wd.size() == rhs_size &&
                 wd.padding_high() == rhs_size - 1 &&
                 wd.padding_low() == rhs_size - 1 && wd.window_reversal()) {
        // A RHS non-contracting dimension can be represented as a spatial
        // dimension with window size N (non-contracting dimension size), low
        // padding N - 1,  high padding N - 1 and window reversal.
        dims.rhs_non_contracting_dims.push_back({lhs, rhs, output, i});
      } else {
        dims.conv_spatial_dims.push_back({lhs, rhs, output, i});
      }
    } else {
      dims.conv_spatial_dims.push_back({lhs, rhs, output, i});
    }
  }

  dims.lhs_shape_rank = conv->operand(0)->shape().rank();
  dims.rhs_shape_rank = conv->operand(1)->shape().rank();
  dims.output_shape_rank = conv->shape().rank();
  return dims;
}

absl::StatusOr<std::unique_ptr<HloInstruction>>
CreateShardedConvForDotGeneralConvolution(
    const HloInstruction& conv, const DotConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo) {
  CHECK_EQ(conv.opcode(), HloOpcode::kConvolution);
  const auto& conv_dnums = conv.convolution_dimension_numbers();
  auto window = conv.window();
  for (const auto& dim : dot_dnums.batch_dims) {
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
    wd->set_stride(std::max<int64_t>(1, wd->size() - 1));
    wd->set_base_dilation(wd->size());
  }
  for (const auto& dim : dot_dnums.contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
  }
  for (const auto& dim : dot_dnums.rhs_non_contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_rhs_hlo->shape().dimensions(
        conv_dnums.kernel_spatial_dimensions(dim.spatial_dim)));
    wd->set_padding_high(wd->size() - 1);
    wd->set_padding_low(wd->size() - 1);
  }
  TF_ASSIGN_OR_RETURN(
      Shape sharded_conv_shape,
      ShapeInference::InferConvolveShape(
          sharded_lhs_hlo->shape(), sharded_rhs_hlo->shape(),
          /*feature_group_count=*/conv.feature_group_count(),
          /*batch_group_count=*/conv.batch_group_count(), window, conv_dnums,
          /*preferred_element_type=*/conv.shape().element_type()));
  *sharded_conv_shape.mutable_layout() = conv.shape().layout();
  return HloInstruction::CreateConvolve(
      sharded_conv_shape, sharded_lhs_hlo, sharded_rhs_hlo,
      /*feature_group_count=*/conv.feature_group_count(),
      /*batch_group_count=*/conv.batch_group_count(), window, conv_dnums,
      conv.precision_config());
}

DotConvolutionDimsInfo ParseDotGeneralFromDot(const HloInstruction* dot) {
  const auto& dot_dim_numbs = dot->dot_dimension_numbers();
  dot_as_convolution_util::DotConvolutionDimsInfo dnums;
  for (int64_t i = 0; i < dot_dim_numbs.lhs_batch_dimensions().size(); ++i) {
    dnums.batch_dims.emplace_back();
    dnums.batch_dims.back().lhs = dot_dim_numbs.lhs_batch_dimensions(i);
    dnums.batch_dims.back().rhs = dot_dim_numbs.rhs_batch_dimensions(i);
    dnums.batch_dims.back().output = i;
    dnums.batch_dims.back().spatial_dim = -1;
  }
  for (int64_t i = 0; i < dot_dim_numbs.lhs_contracting_dimensions().size();
       ++i) {
    dnums.contracting_dims.emplace_back();
    dnums.contracting_dims.back().lhs =
        dot_dim_numbs.lhs_contracting_dimensions(i);
    dnums.contracting_dims.back().rhs =
        dot_dim_numbs.rhs_contracting_dimensions(i);
    dnums.contracting_dims.back().output = -1;
    dnums.contracting_dims.back().spatial_dim = -1;
  }
  for (auto i :
       GetNonContractingDims(dot->operand(0)->shape().rank(),
                             dot_dim_numbs.lhs_contracting_dimensions(),
                             dot_dim_numbs.lhs_batch_dimensions())) {
    dnums.lhs_non_contracting_dims.emplace_back();
    dnums.lhs_non_contracting_dims.back().lhs = i;
    dnums.lhs_non_contracting_dims.back().rhs = -1;
    dnums.lhs_non_contracting_dims.back().output =
        dot_dim_numbs.lhs_batch_dimensions_size() +
        dnums.lhs_non_contracting_dims.size() - 1;
    dnums.lhs_non_contracting_dims.back().spatial_dim = -1;
  }
  for (auto i :
       GetNonContractingDims(dot->operand(1)->shape().rank(),
                             dot_dim_numbs.rhs_contracting_dimensions(),
                             dot_dim_numbs.rhs_batch_dimensions())) {
    dnums.rhs_non_contracting_dims.emplace_back();
    dnums.rhs_non_contracting_dims.back().lhs = -1;
    dnums.rhs_non_contracting_dims.back().rhs = i;
    dnums.rhs_non_contracting_dims.back().output =
        dot_dim_numbs.lhs_batch_dimensions_size() +
        dnums.lhs_non_contracting_dims.size() +
        dnums.rhs_non_contracting_dims.size() - 1;
    dnums.rhs_non_contracting_dims.back().spatial_dim = -1;
  }

  dnums.lhs_shape_rank = dot->operand(0)->shape().rank();
  dnums.rhs_shape_rank = dot->operand(1)->shape().rank();
  dnums.output_shape_rank = dot->shape().rank();
  return dnums;
}

}  // namespace dot_as_convolution_util
}  // namespace xla
