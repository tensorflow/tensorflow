/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_as_convolution_util.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace dot_as_convolution_util {

/* static */ absl::optional<DotGeneralAsConvolutionDimsInfo>
ParseDotGeneralFromConvolution(const HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  if (conv->feature_group_count() != 1 || conv->batch_group_count() != 1) {
    return absl::nullopt;
  }
  const auto& conv_dims = conv->convolution_dimension_numbers();
  DotGeneralAsConvolutionDimsInfo dims;
  dims.lhs_non_contracting_dims.push_back(
      {conv_dims.input_batch_dimension(), -1,
       conv_dims.output_batch_dimension(), -1});
  dims.rhs_non_contracting_dims.push_back(
      {-1, conv_dims.kernel_output_feature_dimension(),
       conv_dims.output_feature_dimension(), -1});
  dims.contracting_dims.push_back({conv_dims.input_feature_dimension(),
                                   conv_dims.kernel_input_feature_dimension(),
                                   -1, -1});

  for (int64 i = 0; i < conv_dims.input_spatial_dimensions_size(); ++i) {
    int64 lhs = conv_dims.input_spatial_dimensions(i);
    int64 lhs_size = conv->operand(0)->shape().dimensions(lhs);
    int64 rhs = conv_dims.kernel_spatial_dimensions(i);
    int64 rhs_size = conv->operand(1)->shape().dimensions(rhs);
    int64 output = conv_dims.output_spatial_dimensions(i);
    const auto& wd = conv->window().dimensions(i);
    if (lhs_size == wd.size() &&
        std::max<int64>(1, lhs_size - 1) == wd.stride() &&
        lhs_size == wd.base_dilation() && wd.window_dilation() == 1 &&
        wd.padding_high() == 0 && wd.padding_low() == 0 &&
        !wd.window_reversal()) {
      // A batch dimension in DotGeneral is represented as a spatial dimension
      // with window size B (batch dimension size), stride B - 1, and base
      // dilation B.
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
        return absl::nullopt;
      }
    } else {
      return absl::nullopt;
    }
  }

  return dims;
}

StatusOr<std::unique_ptr<HloInstruction>>
CreateShardedConvForDotGeneralConvolution(
    const HloInstruction& conv,
    const DotGeneralAsConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo) {
  CHECK_EQ(conv.opcode(), HloOpcode::kConvolution);
  const auto& conv_dnums = conv.convolution_dimension_numbers();
  auto window = conv.window();
  for (const auto& dim : dot_dnums.batch_dims) {
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
    wd->set_stride(std::max<int64>(1, wd->size() - 1));
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
  TF_ASSIGN_OR_RETURN(Shape sharded_conv_shape,
                      ShapeInference::InferConvolveShape(
                          sharded_lhs_hlo->shape(), sharded_rhs_hlo->shape(),
                          /*feature_group_count=*/1,
                          /*batch_group_count=*/1, window, conv_dnums));
  *sharded_conv_shape.mutable_layout() = conv.shape().layout();
  return HloInstruction::CreateConvolve(
      sharded_conv_shape, sharded_lhs_hlo, sharded_rhs_hlo,
      /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, conv_dnums, conv.precision_config());
}

}  // namespace dot_as_convolution_util
}  // namespace xla
