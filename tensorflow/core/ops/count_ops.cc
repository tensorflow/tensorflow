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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status DenseCountSparseOutputShapeFn(InferenceContext *c) {
  auto values = c->input(0);
  auto weights = c->input(1);
  ShapeHandle output;
  auto num_weights = c->NumElements(weights);
  if (c->ValueKnown(num_weights) && c->Value(num_weights) == 0) {
    output = values;
  } else {
    TF_RETURN_IF_ERROR(c->Merge(weights, values, &output));
  }
  auto rank = c->Rank(output);
  auto nvals = c->UnknownDim();
  c->set_output(0, c->Matrix(nvals, rank));  // out.indices
  c->set_output(1, c->Vector(nvals));        // out.values
  c->set_output(2, c->Vector(rank));         // out.dense_shape
  return Status::OK();
}

Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
  auto rank = c->Dim(c->input(0), 1);
  auto nvals = c->UnknownDim();
  c->set_output(0, c->Matrix(nvals, rank));  // out.indices
  c->set_output(1, c->Vector(nvals));        // out.values
  c->set_output(2, c->Vector(rank));         // out.dense_shape
  return Status::OK();
}

Status RaggedCountSparseOutputShapeFn(InferenceContext *c) {
  int32 rank = c->Rank(c->input(1));
  if (rank != c->kUnknownRank) {
    ++rank;  // Add the ragged dimension
  }
  auto nvals = c->UnknownDim();
  c->set_output(0, c->Matrix(nvals, rank));  // out.indices
  c->set_output(1, c->Vector(nvals));        // out.values
  c->set_output(2, c->Vector(rank));         // out.dense_shape
  return Status::OK();
}

REGISTER_OP("DenseCountSparseOutput")
    .Input("values: T")
    .Input("weights: output_type")
    .Attr("T: {int32, int64}")
    .Attr("minlength: int >= -1 = -1")
    .Attr("maxlength: int >= -1 = -1")
    .Attr("binary_output: bool")
    .Attr("output_type: {int32, int64, float, double}")
    .SetShapeFn(DenseCountSparseOutputShapeFn)
    .Output("output_indices: int64")
    .Output("output_values: output_type")
    .Output("output_dense_shape: int64");

REGISTER_OP("SparseCountSparseOutput")
    .Input("indices: int64")
    .Input("values: T")
    .Input("dense_shape: int64")
    .Input("weights: output_type")
    .Attr("T: {int32, int64}")
    .Attr("minlength: int >= -1 = -1")
    .Attr("maxlength: int >= -1 = -1")
    .Attr("binary_output: bool")
    .Attr("output_type: {int32, int64, float, double}")
    .SetShapeFn(SparseCountSparseOutputShapeFn)
    .Output("output_indices: int64")
    .Output("output_values: output_type")
    .Output("output_dense_shape: int64");

REGISTER_OP("RaggedCountSparseOutput")
    .Input("splits: int64")
    .Input("values: T")
    .Input("weights: output_type")
    .Attr("T: {int32, int64}")
    .Attr("minlength: int >= -1 = -1")
    .Attr("maxlength: int >= -1 = -1")
    .Attr("binary_output: bool")
    .Attr("output_type: {int32, int64, float, double}")
    .SetShapeFn(RaggedCountSparseOutputShapeFn)
    .Output("output_indices: int64")
    .Output("output_values: output_type")
    .Output("output_dense_shape: int64");

}  // namespace tensorflow
