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

Status RaggedRangeShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedRange")
    .Input("starts: T")
    .Input("limits: T")
    .Input("deltas: T")
    .Output("rt_nested_splits: Tsplits")
    .Output("rt_dense_values: T")
    .Attr("T: {bfloat16, float, double, int32, int64} = DT_INT32")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedRangeShapeFn);

//==============================================================================
// Shape Functions
//==============================================================================

Status RaggedRangeShapeFn(InferenceContext* c) {
  // Check that all inputs (starts, limits, and deltas) have rank 0 or 1.
  ShapeHandle starts = c->input(0);
  ShapeHandle limits = c->input(1);
  ShapeHandle deltas = c->input(2);
  TF_RETURN_IF_ERROR(c->WithRankAtMost(starts, 1, &starts));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(limits, 1, &limits));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(deltas, 1, &deltas));

  // For the inputs with rank 1, make sure shapes match.
  DimensionHandle dim = c->UnknownDim();
  if (c->Rank(starts) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(starts, 0), dim, &dim));
  }
  if (c->Rank(limits) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(limits, 0), dim, &dim));
  }
  if (c->Rank(deltas) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(deltas, 0), dim, &dim));
  }

  // If any input shape is known, then calculate `rt_nested_splits` shape.
  int64 rt_nested_splits_dim = InferenceContext::kUnknownDim;
  if (c->ValueKnown(dim)) {
    rt_nested_splits_dim = c->Value(dim) + 1;
  } else if (c->Rank(starts) == 0 && c->Rank(limits) == 0 &&
             c->Rank(deltas) == 0) {
    rt_nested_splits_dim = 2;
  }
  c->set_output(0, c->Vector(rt_nested_splits_dim));

  // `rt_dense_values` is rank 1, but size can't be calculated statically.
  c->set_output(1, c->UnknownShapeOfRank(1));
  return Status::OK();
}

}  // namespace tensorflow
