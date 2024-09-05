/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SetSize")
    .Input("set_indices: int64")
    .Input("set_values: T")
    .Input("set_shape: int64")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("DenseToDenseSetOperation")
    .Input("set1: T")
    .Input("set2: T")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() != 2) {
        return errors::InvalidArgument("len(inputs) != 2.");
      }
      // The following should stay in sync with `ComputeDenseToDense` shape
      // assertions in kernels/set_kernels.cc.
      // Dimension n contains the set values to be compared, so ranks must be
      // >= 2, and the first n-1 dimensions of inputs and output must be
      // compatible.
      DimensionHandle output_rank;
      ShapeHandle input0_shape = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(input0_shape, 2, &input0_shape));
      if (c->RankKnown(input0_shape)) {
        const int32_t input0_rank = c->Rank(input0_shape);
        ShapeHandle input1_shape = c->input(1);
        TF_RETURN_IF_ERROR(
            c->WithRank(input1_shape, input0_rank, &input1_shape));
        if (c->RankKnown(input1_shape)) {
          // If both ranks are specified, the first n-1 dims must be compatible.
          const int32_t rank = c->Rank(input1_shape);
          ShapeHandle group0_shape;
          TF_RETURN_IF_ERROR(
              c->Subshape(input0_shape, 0, rank - 1, &group0_shape));
          ShapeHandle group1_shape;
          TF_RETURN_IF_ERROR(
              c->Subshape(input1_shape, 0, rank - 1, &group1_shape));
          ShapeHandle unused_shape;
          TF_RETURN_IF_ERROR(
              c->Merge(group0_shape, group1_shape, &unused_shape));
        }
        output_rank = c->MakeDim(input0_rank);
      } else {
        ShapeHandle input1_shape = c->input(1);
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(input1_shape, 2, &input1_shape));
        if (c->RankKnown(input1_shape)) {
          output_rank = c->MakeDim(c->Rank(input1_shape));
        } else {
          output_rank = c->UnknownDim();
        }
      }

      c->set_output(0, c->Matrix(c->UnknownDim(), output_rank));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(output_rank));
      return absl::OkStatus();
    });

REGISTER_OP("DenseToSparseSetOperation")
    .Input("set1: T")
    .Input("set2_indices: int64")
    .Input("set2_values: T")
    .Input("set2_shape: int64")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() != 4) {
        return errors::InvalidArgument("len(inputs) != 4.");
      }
      // The following should stay in sync with `ComputeDenseToSparse` shape
      // assertions in kernels/set_kernels.cc.
      // Ranks must be compatible, and be >= 2.
      ShapeHandle input1_shape_shape = c->input(3);
      TF_RETURN_IF_ERROR(shape_inference::ValidateSparseTensor(
          c, c->input(1), c->input(2), input1_shape_shape));

      DimensionHandle input1_rank_dim = c->Dim(input1_shape_shape, 0);

      DimensionHandle output_rank_dim;
      ShapeHandle input0_shape = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(input0_shape, 2, &input0_shape));
      if (c->RankKnown(input0_shape)) {
        const int32_t input0_rank = c->Rank(input0_shape);
        TF_RETURN_IF_ERROR(
            c->WithValue(input1_rank_dim, input0_rank, &input1_rank_dim));
        output_rank_dim = c->MakeDim(input0_rank);
      } else if (c->ValueKnown(input1_rank_dim)) {
        output_rank_dim = input1_rank_dim;
      } else {
        output_rank_dim = c->UnknownDim();
      }

      c->set_output(0, c->Matrix(c->UnknownDim(), output_rank_dim));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(output_rank_dim));
      return absl::OkStatus();
    });

REGISTER_OP("SparseToSparseSetOperation")
    .Input("set1_indices: int64")
    .Input("set1_values: T")
    .Input("set1_shape: int64")
    .Input("set2_indices: int64")
    .Input("set2_values: T")
    .Input("set2_shape: int64")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() != 6) {
        return errors::InvalidArgument("len(inputs) != 6.");
      }
      // The following should stay in sync with `ComputeSparseToSparse` shape
      // assertions in kernels/set_kernels.cc.
      // Ranks must be compatible, and be >= 2.
      ShapeHandle input0_shape_shape = c->input(2);
      ShapeHandle input1_shape_shape = c->input(5);
      TF_RETURN_IF_ERROR(shape_inference::ValidateSparseTensor(
          c, c->input(0), c->input(1), input0_shape_shape));
      TF_RETURN_IF_ERROR(shape_inference::ValidateSparseTensor(
          c, c->input(3), c->input(4), input1_shape_shape));

      DimensionHandle input0_rank_dim = c->Dim(input0_shape_shape, 0);
      DimensionHandle input1_rank_dim = c->Dim(input1_shape_shape, 0);
      DimensionHandle output_rank_dim;
      if (c->ValueKnown(input0_rank_dim)) {
        const int64_t input0_rank = c->Value(input0_rank_dim);
        if (input0_rank < 2) {
          return errors::InvalidArgument("Input 0, expected rank >= 2, got ",
                                         input0_rank, ".");
        }
        TF_RETURN_IF_ERROR(
            c->WithValue(input1_rank_dim, input0_rank, &input1_rank_dim));
        output_rank_dim = input0_rank_dim;
      } else if (c->ValueKnown(input1_rank_dim)) {
        const int64_t input1_rank = c->Value(input1_rank_dim);
        if (input1_rank < 2) {
          return errors::InvalidArgument("Input 1, expected rank >= 2, got ",
                                         input1_rank, ".");
        }
        output_rank_dim = input1_rank_dim;
      } else {
        output_rank_dim = c->UnknownDim();
      }

      c->set_output(0, c->Matrix(c->UnknownDim(), output_rank_dim));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(output_rank_dim));
      return absl::OkStatus();
    });

}  // namespace tensorflow
