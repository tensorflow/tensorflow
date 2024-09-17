/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_params.h"

namespace tensorflow {
namespace {

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unknown;

// If the rank and all dim sizes are known, return corresponding TensorShape.
// Otherwise return Unknown error.
absl::StatusOr<TensorShape> ToTensorShape(ShapeHandle shape_handle,
                                          int64_t rank) {
  TensorShape shape;
  for (int i = 0; i < rank; ++i) {
    int64_t dim_size = shape_inference::InferenceContext::Value(
        shape_inference::InferenceContext::DimKnownRank(shape_handle, i));
    if (dim_size == shape_inference::InferenceContext::kUnknownDim) {
      return Unknown("Dim size unknown.");
    }
    shape.AddDim(dim_size);
  }
  return shape;
}

Status ScalesZeroPointsShapeValid(shape_inference::InferenceContext* context,
                                  DimensionHandle match_dimension_handle,
                                  ShapeHandle scales, ShapeHandle zero_points) {
  const int32_t scales_rank = shape_inference::InferenceContext::Rank(scales);
  const int32_t zero_points_rank =
      shape_inference::InferenceContext::Rank(zero_points);
  // Skip validation when rank is unknown.
  if (scales_rank == shape_inference::InferenceContext::kUnknownRank ||
      zero_points_rank == shape_inference::InferenceContext::kUnknownRank) {
    return absl::OkStatus();
  }

  if (scales_rank != zero_points_rank) {
    return InvalidArgument("scales and zero_points must have same rank.");
  }
  if (scales_rank == 0) {
    return absl::OkStatus();
  }
  DimensionHandle scales_size = context->Dim(scales, 0);
  DimensionHandle zero_points_size = context->Dim(zero_points, 0);
  DimensionHandle merged_scales;
  TF_RETURN_IF_ERROR(
      context->Merge(scales_size, match_dimension_handle, &merged_scales));
  DimensionHandle merged_zero_points;
  TF_RETURN_IF_ERROR(context->Merge(zero_points_size, match_dimension_handle,
                                    &merged_zero_points));
  return absl::OkStatus();
}

Status DotShape(shape_inference::InferenceContext* context) {
  ShapeHandle lhs;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &lhs));
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &rhs));
  // lhs scales and zero_points must be scalar tensors.
  ShapeHandle lhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(2), 0, &lhs_scales));
  ShapeHandle lhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(3), 0, &lhs_zero_points));
  ShapeHandle rhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(4), 1, &rhs_scales));
  ShapeHandle rhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(5), 1, &rhs_zero_points));
  ShapeHandle output_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(6), 1, &output_scales));
  ShapeHandle output_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(7), 1, &output_zero_points));

  // Validate that the inner shapes are compatible.
  DimensionHandle inner_lhs = context->Dim(lhs, 1);
  DimensionHandle inner_rhs = context->Dim(rhs, 0);
  DimensionHandle merged;
  TF_RETURN_IF_ERROR(context->Merge(inner_lhs, inner_rhs, &merged));

  DimensionHandle output_rows = context->Dim(lhs, 0);
  DimensionHandle output_cols = context->Dim(rhs, 1);

  TF_RETURN_IF_ERROR(ScalesZeroPointsShapeValid(context, output_cols,
                                                rhs_scales, rhs_zero_points));
  TF_RETURN_IF_ERROR(ScalesZeroPointsShapeValid(
      context, output_cols, output_scales, output_zero_points));

  context->set_output(0, context->Matrix(output_rows, output_cols));
  return absl::OkStatus();
}

Status DotHybridShape(shape_inference::InferenceContext* context) {
  ShapeHandle lhs;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &lhs));
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 2, &rhs));
  ShapeHandle rhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(2), 1, &rhs_scales));
  ShapeHandle rhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(3), 1, &rhs_zero_points));

  // Validate that the inner shapes are compatible.
  DimensionHandle inner_lhs = context->Dim(lhs, 1);
  DimensionHandle inner_rhs = context->Dim(rhs, 0);
  DimensionHandle merged;
  TF_RETURN_IF_ERROR(context->Merge(inner_lhs, inner_rhs, &merged));

  DimensionHandle output_rows = context->Dim(lhs, 0);
  DimensionHandle output_cols = context->Dim(rhs, 1);

  TF_RETURN_IF_ERROR(ScalesZeroPointsShapeValid(context, output_cols,
                                                rhs_scales, rhs_zero_points));

  context->set_output(0, context->Matrix(output_rows, output_cols));
  return absl::OkStatus();
}

struct ShapeCommonParams {
  ShapeHandle lhs;
  ShapeHandle rhs;
  ShapeHandle lhs_scales;
  ShapeHandle lhs_zero_points;
  ShapeHandle rhs_scales;
  ShapeHandle rhs_zero_points;
  ShapeHandle output_scales;
  ShapeHandle output_zero_points;
  bool is_output_scales_zero_points_set;

  ShapeCommonParams(ShapeHandle lhs, ShapeHandle rhs, ShapeHandle lhs_scales,
                    ShapeHandle lhs_zero_points, ShapeHandle rhs_scales,
                    ShapeHandle rhs_zero_points, ShapeHandle output_scales,
                    ShapeHandle output_zero_points)
      : lhs(lhs),
        rhs(rhs),
        lhs_scales(lhs_scales),
        lhs_zero_points(lhs_zero_points),
        rhs_scales(rhs_scales),
        rhs_zero_points(rhs_zero_points),
        output_scales(output_scales),
        output_zero_points(output_zero_points),
        is_output_scales_zero_points_set(true) {}

  ShapeCommonParams(ShapeHandle lhs, ShapeHandle rhs, ShapeHandle rhs_scales,
                    ShapeHandle rhs_zero_points)
      : lhs(lhs),
        rhs(rhs),
        rhs_scales(rhs_scales),
        rhs_zero_points(rhs_zero_points),
        is_output_scales_zero_points_set(false) {}
};

Status ConvolutionShapeCommon(shape_inference::InferenceContext* context,
                              const ShapeCommonParams& params) {
  const int32_t lhs_rank = shape_inference::InferenceContext::Rank(params.lhs);
  const int32_t rhs_rank = shape_inference::InferenceContext::Rank(params.rhs);

  if (lhs_rank == shape_inference::InferenceContext::kUnknownRank &&
      rhs_rank == shape_inference::InferenceContext::kUnknownRank) {
    context->set_output(0, context->UnknownShape());
    return absl::OkStatus();
  } else if (lhs_rank == shape_inference::InferenceContext::kUnknownRank ||
             rhs_rank == shape_inference::InferenceContext::kUnknownRank) {
    context->set_output(
        0, context->UnknownShapeOfRank(
               lhs_rank == shape_inference::InferenceContext::kUnknownRank
                   ? rhs_rank
                   : lhs_rank));
    return absl::OkStatus();
  } else if (lhs_rank != rhs_rank) {
    return InvalidArgument("lhs and rhs must have same rank.");
  }

  auto lhs_shape = ToTensorShape(params.lhs, lhs_rank);
  auto rhs_shape = ToTensorShape(params.rhs, rhs_rank);
  if (!lhs_shape.ok() || !rhs_shape.ok()) {
    context->set_output(0, context->UnknownShapeOfRank(lhs_rank));
    return absl::OkStatus();
  }

  UniformQuantizedConvolutionParams convolution_params;
  TF_RETURN_IF_ERROR(convolution_params.LoadFromAttrs(*context));
  TF_RETURN_IF_ERROR(convolution_params.ValidateOrFillParamsAndValidateShape(
      lhs_shape.value(), rhs_shape.value()));

  DimensionHandle output_feature = context->Dim(
      params.rhs,
      convolution_params.dimension_numbers().kernel_output_feature_dimension());
  TF_RETURN_IF_ERROR(ScalesZeroPointsShapeValid(
      context, output_feature, params.rhs_scales, params.rhs_zero_points));
  if (params.is_output_scales_zero_points_set) {
    TF_RETURN_IF_ERROR(ScalesZeroPointsShapeValid(context, output_feature,
                                                  params.output_scales,
                                                  params.output_zero_points));
    if (shape_inference::InferenceContext::Rank(params.output_scales) > 0) {
      DimensionHandle scales_merged;
      TF_RETURN_IF_ERROR(context->Merge(context->Dim(params.rhs_scales, 0),
                                        context->Dim(params.output_scales, 0),
                                        &scales_merged));
    }
  }

  TF_ASSIGN_OR_RETURN(const auto& out_shape,
                      convolution_params.CalculateOutputShape(
                          lhs_shape.value(), rhs_shape.value()));
  ShapeHandle out_shape_handle;
  TF_RETURN_IF_ERROR(
      context->MakeShapeFromTensorShape(out_shape, &out_shape_handle));
  context->set_output(0, out_shape_handle);
  return absl::OkStatus();
}

Status ConvolutionShape(shape_inference::InferenceContext* context) {
  ShapeHandle lhs;
  TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(0), 2, &lhs));
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(1), 2, &rhs));
  // lhs scales and zero_points must be scalar tensors.
  ShapeHandle lhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(2), 0, &lhs_scales));
  ShapeHandle lhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(3), 0, &lhs_zero_points));
  ShapeHandle rhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(4), 1, &rhs_scales));
  ShapeHandle rhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(5), 1, &rhs_zero_points));
  ShapeHandle output_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(6), 1, &output_scales));
  ShapeHandle output_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(7), 1, &output_zero_points));

  return ConvolutionShapeCommon(
      context,
      ShapeCommonParams(lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales,
                        rhs_zero_points, output_scales, output_zero_points));
}

Status ConvolutionHybridShape(shape_inference::InferenceContext* context) {
  ShapeHandle lhs;
  TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(0), 2, &lhs));
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(1), 2, &rhs));
  ShapeHandle rhs_scales;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(2), 1, &rhs_scales));
  ShapeHandle rhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->WithRankAtMost(context->input(3), 1, &rhs_zero_points));
  return ConvolutionShapeCommon(
      context, ShapeCommonParams(lhs, rhs, rhs_scales, rhs_zero_points));
}

}  // namespace

REGISTER_OP("UniformQuantize")
    .Input("input: Tin")
    .Input("scales: float")
    .Input("zero_points: int32")
    .Output("output: Tout")
    .Attr("Tin: {float}")
    .Attr("Tout: {qint8, qint32}")
    .Attr("quantization_axis: int = -1")
    .Attr("quantization_min_val: int")
    .Attr("quantization_max_val: int")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("UniformRequantize")
    .Input("input: Tin")
    .Input("input_scales: float")
    .Input("input_zero_points: int32")
    .Input("output_scales: float")
    .Input("output_zero_points: int32")
    .Output("output: Tout")
    .Attr("Tin: {qint8, qint32}")
    .Attr("Tout: {qint8, qint32}")
    .Attr("input_quantization_axis: int = -1")
    .Attr("input_quantization_min_val: int")
    .Attr("input_quantization_max_val: int")
    .Attr("output_quantization_axis: int = -1")
    .Attr("output_quantization_min_val: int")
    .Attr("output_quantization_max_val: int")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("UniformDequantize")
    .Input("input: Tin")
    .Input("scales: float")
    .Input("zero_points: int32")
    .Output("output: Tout")
    .Attr("Tin: {qint8, qint32}")
    .Attr("Tout: {float}")
    .Attr("quantization_axis: int = -1")
    .Attr("quantization_min_val: int")
    .Attr("quantization_max_val: int")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("UniformQuantizedDot")
    .Input("lhs: Tin")
    .Input("rhs: Tin")
    .Input("lhs_scales: float")
    .Input("lhs_zero_points: int32")
    .Input("rhs_scales: float")
    .Input("rhs_zero_points: int32")
    .Input("output_scales: float")
    .Input("output_zero_points: int32")
    .Output("output: Tout")
    .Attr("Tin: {qint8}")
    .Attr("Tout: {qint32}")
    .Attr("lhs_quantization_axis: int = -1")
    .Attr("lhs_quantization_min_val: int")
    .Attr("lhs_quantization_max_val: int")
    .Attr("rhs_quantization_axis: int = -1")
    .Attr("rhs_quantization_min_val: int")
    .Attr("rhs_quantization_max_val: int")
    .Attr("output_quantization_axis: int = -1")
    .Attr("output_quantization_min_val: int")
    .Attr("output_quantization_max_val: int")
    .SetShapeFn(DotShape);

REGISTER_OP("UniformQuantizedDotHybrid")
    .Input("lhs: Tlhs")
    .Input("rhs: Trhs")
    .Input("rhs_scales: float")
    .Input("rhs_zero_points: int32")
    .Output("output: Tout")
    .Attr("Tlhs: {float}")
    .Attr("Trhs: {qint8}")
    .Attr("Tout: {float}")
    .Attr("rhs_quantization_axis: int = -1")
    .Attr("rhs_quantization_min_val: int")
    .Attr("rhs_quantization_max_val: int")
    .SetShapeFn(DotHybridShape);

REGISTER_OP("UniformQuantizedConvolution")
    .Input("lhs: Tin")
    .Input("rhs: Tin")
    .Input("lhs_scales: float")
    .Input("lhs_zero_points: int32")
    .Input("rhs_scales: float")
    .Input("rhs_zero_points: int32")
    .Input("output_scales: float")
    .Input("output_zero_points: int32")
    .Output("output: Tout")
    .Attr("Tin: {qint8}")
    .Attr("Tout: {qint32}")
    .Attr("window_strides: list(int) = []")
    .Attr("padding: string")
    .Attr("explicit_padding: list(int) = []")
    .Attr("lhs_dilation: list(int) = []")
    .Attr("rhs_dilation: list(int) = []")
    .Attr("batch_group_count: int = 1")
    .Attr("feature_group_count: int = 1")
    .Attr("dimension_numbers: string = ''")
    .Attr("lhs_quantization_axis: int = -1")
    .Attr("lhs_quantization_min_val: int")
    .Attr("lhs_quantization_max_val: int")
    .Attr("rhs_quantization_axis: int = -1")
    .Attr("rhs_quantization_min_val: int")
    .Attr("rhs_quantization_max_val: int")
    .Attr("output_quantization_axis: int = -1")
    .Attr("output_quantization_min_val: int")
    .Attr("output_quantization_max_val: int")
    .SetShapeFn(ConvolutionShape);

REGISTER_OP("UniformQuantizedConvolutionHybrid")
    .Input("lhs: Tlhs")
    .Input("rhs: Trhs")
    .Input("rhs_scales: float")
    .Input("rhs_zero_points: int32")
    .Output("output: Tout")
    .Attr("Tlhs: {float}")
    .Attr("Trhs: {qint8}")
    .Attr("Tout: {float}")
    .Attr("window_strides: list(int) = []")
    .Attr("padding: string")
    .Attr("explicit_padding: list(int) = []")
    .Attr("lhs_dilation: list(int) = []")
    .Attr("rhs_dilation: list(int) = []")
    .Attr("batch_group_count: int = 1")
    .Attr("feature_group_count: int = 1")
    .Attr("dimension_numbers: string = ''")
    .Attr("rhs_quantization_axis: int = -1")
    .Attr("rhs_quantization_min_val: int")
    .Attr("rhs_quantization_max_val: int")
    .SetShapeFn(ConvolutionHybridShape);

REGISTER_OP("UniformQuantizedAdd")
    .Input("lhs: T")
    .Input("rhs: T")
    .Input("lhs_scales: float")
    .Input("lhs_zero_points: int32")
    .Input("rhs_scales: float")
    .Input("rhs_zero_points: int32")
    .Input("output_scales: float")
    .Input("output_zero_points: int32")
    .Output("output: T")
    .Attr("lhs_quantization_axis: int = -1")
    .Attr("lhs_quantization_min_val: int")
    .Attr("lhs_quantization_max_val: int")
    .Attr("rhs_quantization_axis: int = -1")
    .Attr("rhs_quantization_min_val: int")
    .Attr("rhs_quantization_max_val: int")
    .Attr("output_quantization_axis: int = -1")
    .Attr("output_quantization_min_val: int")
    .Attr("output_quantization_max_val: int")
    .Attr("T: {qint32}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("UniformQuantizedClipByValue")
    .Input("operand: T")
    .Input("min: T")
    .Input("max: T")
    .Input("scales: float")
    .Input("zero_points: int32")
    .Output("output: T")
    .Attr("T: {qint32}")
    .Attr("quantization_axis: int = -1")
    .Attr("quantization_min_val: int")
    .Attr("quantization_max_val: int")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
