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

#include <algorithm>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

static Status StatelessRandomPermuteShape(InferenceContext* c) {
  ShapeHandle index_shape, seed_shape, max_index_shape, rounds_shape;

  // Basic constraints but unknown ranks will not raise errors here.
  // index, seed and max_index can be scalars or vectors (when batching).
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &index_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &seed_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 2, &seed_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &max_index_shape));

  // Figure out if the output is a scalar or tensor.
  const int32 index_rank = c->Rank(index_shape);
  const int32 seed_rank = c->Rank(seed_shape);
  const int32 max_index_rank = c->Rank(max_index_shape);

  // Check that last dimension of seed is 3.
  if (seed_rank == 1 && c->Value(c->Dim(seed_shape, 0)) != 3) {
    return errors::InvalidArgument(
        "Seed must have shape [3] or [n, 3] but got [",
        c->Value(c->Dim(seed_shape, 0)), "].");
  }
  if (seed_rank == 2 && c->Value(c->Dim(seed_shape, 1)) != 3) {
    return errors::InvalidArgument(
        "Seed must have shape [3] or [n, 3] but got [",
        c->Value(c->Dim(seed_shape, 0)), ", ", c->Value(c->Dim(seed_shape, 1)),
        "].");
  }

  // Below we handle 3 cases:
  // 1. If all inputs are scalars the output is a scalar.
  // 2. If we cannot decide if the output is a scalar or a vector we output
  //    unknown shape.
  // 3. The output must be a vector and try to compute it's size.

  // Case 1.
  // If all inputs are scalars the output is a scalar.
  const bool output_is_scalar =
      (index_rank == 0 && seed_rank == 1 && max_index_rank == 0);
  if (output_is_scalar) {
    c->set_output(0, c->Scalar());
    return OkStatus();
  }

  // Case 2.
  // If we know for certain that the output is a vector we should proceed to
  // calculate the size below. Otherwise the output could be a scalar or a
  // vector.
  const bool output_must_be_vector =
      (index_rank == 1 || seed_rank == 2 || max_index_rank == 1);
  if (!output_must_be_vector) {
    c->set_output(0, c->UnknownShape());
    return OkStatus();
  }

  // Case 3.
  // Output is a vector and we try to compute the size `num_outputs`. The result
  // can be kUknownDim.
  int64_t num_outputs = InferenceContext::kUnknownDim;

  // Check index.
  if (index_rank == 1) num_outputs = c->Value(c->Dim(index_shape, 0));

  // Check seed.
  if (seed_rank == 2) {
    const int64_t num_seeds = c->Value(c->Dim(seed_shape, 0));
    if (num_outputs == InferenceContext::kUnknownDim) {
      num_outputs = num_seeds;
    } else if (num_outputs > 1 && num_seeds != InferenceContext::kUnknownDim &&
               num_seeds > 1 && num_seeds != num_outputs) {
      return errors::InvalidArgument(
          "Seed has shape [", num_seeds, ", 3] but must have shape [",
          num_outputs, ", 3]. since index had shape [", num_outputs, "].");
    }
  }

  // Check max index.
  if (max_index_rank == 1) {
    int64_t num_max_indices = c->Value(c->Dim(max_index_shape, 0));
    if (num_outputs == InferenceContext::kUnknownDim) {
      num_outputs = num_max_indices;
    } else if (num_outputs > 1 &&
               num_max_indices != InferenceContext::kUnknownDim &&
               num_max_indices > 1 && num_max_indices != num_outputs) {
      return errors::InvalidArgument("Max index has shape [", num_max_indices,
                                     "] but must have shape [", num_outputs,
                                     "].");
    }
  }

  c->set_output(0, c->Vector(num_outputs));
  return OkStatus();
}

REGISTER_OP("RandomIndexShuffle")
    .Input("index: dtype")
    .Input("seed: Tseed")
    .Input("max_index: dtype")
    .Output("output: dtype")
    .Attr("rounds: int = 4")
    .Attr("dtype: {int32, uint32, int64, uint64}")
    .Attr("Tseed: {int32, uint32, int64, uint64}")
    .SetShapeFn(StatelessRandomPermuteShape);

}  // namespace
}  // namespace tensorflow
