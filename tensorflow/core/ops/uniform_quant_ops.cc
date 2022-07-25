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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;
using tensorflow::errors::InvalidArgument;

Status ScalesZeroPointsShapeValid(shape_inference::InferenceContext* context,
                                  DimensionHandle match_dimension_handle,
                                  ShapeHandle scales, ShapeHandle zero_points) {
  const int32_t scales_rank = shape_inference::InferenceContext::Rank(scales);
  const int32_t zero_points_rank =
      shape_inference::InferenceContext::Rank(zero_points);
  // Skip validation when rank is unknown.
  if (scales_rank == shape_inference::InferenceContext::kUnknownRank ||
      zero_points_rank == shape_inference::InferenceContext::kUnknownRank) {
    return Status::OK();
  }

  if (scales_rank != zero_points_rank) {
    return InvalidArgument("scales and zero_points must have same rank.");
  }
  if (scales_rank == 0) {
    return Status::OK();
  }
  DimensionHandle scales_size = context->Dim(scales, 0);
  DimensionHandle zero_points_size = context->Dim(zero_points, 0);
  DimensionHandle merged_scales;
  TF_RETURN_IF_ERROR(
      context->Merge(scales_size, match_dimension_handle, &merged_scales));
  DimensionHandle merged_zero_points;
  TF_RETURN_IF_ERROR(context->Merge(zero_points_size, match_dimension_handle,
                                    &merged_zero_points));
  return Status::OK();
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
  return OkStatus();
}

}  // namespace

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

}  // namespace tensorflow
