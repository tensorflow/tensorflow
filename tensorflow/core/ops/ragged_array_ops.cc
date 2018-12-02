/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

Status RaggedGatherShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedGather")
    .Input("params_nested_splits: PARAMS_RAGGED_RANK * int64")
    .Input("params_dense_values: Tvalues")
    .Input("indices: Tindices")
    .Output("output_nested_splits: OUTPUT_RAGGED_RANK * int64")
    .Output("output_dense_values: Tvalues")
    .Attr("Tvalues: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("PARAMS_RAGGED_RANK: int >= 1")
    .Attr("OUTPUT_RAGGED_RANK: int >= 0")
    .SetShapeFn(RaggedGatherShapeFn);

//==============================================================================
// Shape Functions
//==============================================================================

Status RaggedGatherShapeFn(InferenceContext* c) {
  int num_splits;
  int64 PARAMS_RAGGED_RANK;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64>("PARAMS_RAGGED_RANK", &PARAMS_RAGGED_RANK));
  TF_RETURN_IF_ERROR(c->GetAttr<int>("OUTPUT_RAGGED_RANK", &num_splits));

  // Check rank of `indices`.
  ShapeHandle indices = c->input(PARAMS_RAGGED_RANK + 1);
  TF_RETURN_IF_ERROR(
      c->WithRank(indices, num_splits - PARAMS_RAGGED_RANK + 1, &indices));

  // Check that all params_nested_splits have rank 1.
  for (int64 i = 0; i < PARAMS_RAGGED_RANK; ++i) {
    ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }

  // Check that `params_dense_values` has rank>=1.
  ShapeHandle params_dense_values = c->input(PARAMS_RAGGED_RANK);
  TF_RETURN_IF_ERROR(
      c->WithRankAtLeast(params_dense_values, 1, &params_dense_values));

  // Set the rank for the `splits` outputs.
  for (int i = 0; i < num_splits; ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }

  // Calculate the `values` shape.
  ShapeHandle value = c->UnknownShape();
  ShapeHandle values = c->UnknownShape();
  TF_RETURN_IF_ERROR(c->Subshape(params_dense_values, 1, &value));
  TF_RETURN_IF_ERROR(c->Concatenate(c->UnknownShapeOfRank(1), value, &values));
  c->set_output(num_splits, values);

  return Status::OK();
}

}  // namespace tensorflow
