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
  ShapeHandle index_shape, seed_shape, max_index_shape;

  // Basic constraints but unknown ranks will not raise errors here.
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
    return errors::InvalidArgument("Seed must have shape [3] but got [",
                                   c->Value(c->Dim(seed_shape, 0)), "].");
  }
  if (seed_rank == 2 && c->Value(c->Dim(seed_shape, 1)) != 3) {
    return errors::InvalidArgument("Seed must have shape [n, 3] but got [",
                                   c->Value(c->Dim(seed_shape, 0)), ", ",
                                   c->Value(c->Dim(seed_shape, 1)), "].");
  }

  // If all inputs are scalars the output is a scalar.
  const bool output_is_scalar =
      (index_rank == 0 && seed_rank == 1 && max_index_rank == 0);
  if (output_is_scalar) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  }

  if (!c->FullyDefined(index_shape) || !c->FullyDefined(seed_shape) ||
      !c->FullyDefined(max_index_shape)) {
    const bool output_is_vector =
        (index_rank == 1 || seed_rank == 2 || max_index_rank == 1);
    if (output_is_vector) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
    }
    return Status::OK();
  }

  // Shape is fully defined and the output is a vector.
  const int64_t num_indices = index_rank ? c->Value(c->Dim(index_shape, 0)) : 1;
  const int64_t num_seeds =
      seed_rank == 2 ? c->Value(c->Dim(seed_shape, 0)) : 1;
  const int64_t num_max_indices =
      max_index_rank ? c->Value(c->Dim(max_index_shape, 0)) : 1;
  const int64_t num_outputs =
      std::max(std::max(num_indices, num_seeds), num_max_indices);
  if (num_indices != 1 && num_indices != num_outputs) {
    return errors::InvalidArgument("Index has shape [", num_indices,
                                   "] but must have shape [", num_outputs,
                                   "].");
  }
  if (num_seeds != 1 && num_seeds != num_outputs) {
    return errors::InvalidArgument("Seed has shape [", num_seeds,
                                   "3, ] but must have shape [", num_outputs,
                                   ", 3].");
  }
  if (num_max_indices != 1 && num_max_indices != num_outputs) {
    return errors::InvalidArgument("Max index has shape [", num_max_indices,
                                   "] but must have shape [", num_outputs,
                                   "].");
  }
  c->set_output(0, c->Vector(num_outputs));
  return Status::OK();
}

REGISTER_OP("RandomIndexShuffle")
    .Input("index: dtype")
    .Input("seed: Tseed")
    .Input("max_index: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, uint32, int64, uint64}")
    .Attr("Tseed: {int32, uint32, int64, uint64}")
    .SetShapeFn(StatelessRandomPermuteShape);

}  // namespace
}  // namespace tensorflow
