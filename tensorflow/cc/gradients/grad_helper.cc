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

#include "tensorflow/cc/gradients/grad_helper.h"

#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {

using tensorflow::ops::Add;
using tensorflow::ops::Const;
using tensorflow::ops::DynamicStitch;
using tensorflow::ops::Mod;
using tensorflow::ops::OnesLike;
using tensorflow::ops::Range;
using tensorflow::ops::Size;

Output ReducedShapeHelper(const Scope& scope, const Output& input_shape,
                          const Output& reduction_axes) {
  auto zero = Const(scope, 0);
  auto one = Const(scope, 1);

  // Running example in comments
  // input_shape = [2, 3, 5, 7]
  // axes = [1, 2]
  // The result (a shape after a reduction with keep_dims=True)
  // [2, 1, 1, 7]
  //
  // We can treat each entry in axes as an index into input_shape that
  // should be replaced by 1.
  // We use DynamicStitch to do this.

  // input_rank = 4
  auto input_rank = Size(scope, input_shape);

  // Normalize any negative indices in the reduction_axes to positive
  // values.
  auto axes = Mod(scope, Add(scope, reduction_axes, input_rank), input_rank);

  // This [0..input_rank) range of integers is used in DynamicStitch to
  // first copy input_shape to the result.
  // input_rank_range = [0, 1, 2, 3]
  auto input_rank_range = Range(scope, zero, input_rank, one);

  // A 1-filled tensor with the same shape as axes. DynamicStitch will
  // merge these 1s (using axes for indices) to the correct
  // position in the result.
  // axes_ones = [1, 1]
  auto axes_ones = OnesLike(scope, axes);

  // using DynamicStitch:
  // indices = { input_rank_range, axes }
  //         = { [0, 1, 2, 3], [1, 2] }
  // data = { input_shape, axes_ones }
  //      = { [2, 3, 5, 7], [1, 1] }
  // The input_rank_range entry in indices first replicates the
  // input_shape to the result.
  // The axes entry in indices then moves a 1 to each of its entries,
  // resulting in
  // [2, 1, 1, 7]
  std::vector<Output> indices = {input_rank_range, axes};
  std::vector<Output> data = {input_shape, axes_ones};
  return DynamicStitch(scope, indices, data);
}

}  // namespace tensorflow
