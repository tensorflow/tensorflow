/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/tensorflow/community/blob/master/rfcs/20190726-custom-ops.md

REGISTER_OP("Examples>MultiplexSparse")
    .Input("cond_indices: int64")
    .Input("cond_values: bool")
    .Input("cond_shape: int64")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // cond_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // cond_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));  // cond_shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));  // a_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused));  // a_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &unused));  // a_shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &unused));  // b_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &unused));  // b_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &unused));  // b_shape
      const auto num_rows = c->UnknownDim();
      const auto dense_rank = c->UnknownDim();
      c->set_output(0, c->Matrix(num_rows, dense_rank));
      c->set_output(1, c->Vector(num_rows));
      c->set_output(2, c->Vector(dense_rank));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Return elements chosen from `a` or `b` depending on `cond`.

This is similar to `np.where` and `tf.where`, but simplified to only handle
the case of sparse tensors that are vectors, no optional parameters,
no broadcasting, etc.. Elements for `a` are chosen if there is a `true` `cond`
value at the same position. Elements for `b` are chosen if there is not a `true`
`cond` value at the same position, i.e., if either there is a `false` `cond`
value or the `cond` value is not specified.

Indices must be ordered as described by tf.sparse_reorder.

cond_indices: a rank-2 tensor of sparse indices.
cond_values: a rank-1 tensor of sparse values.
cond_shape: a rank-1 tensor representing the dense shape.
a_indices: a rank-2 tensor of sparse indices.
a_values: a rank-1 tensor of sparse values.
a_shape: a rank-1 tensor representing the dense shape.
b_indices: a rank-2 tensor of sparse indices.
b_values: a rank-1 tensor of sparse values.
b_shape: a rank-1 tensor representing the dense shape.
output_indices: a rank-2 tensor of sparse indices.
output_values: a rank-1 tensor of sparse values.
output_shape: a rank-1 tensor representing the dense shape.
)doc");
