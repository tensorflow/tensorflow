/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/data_format.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

absl::StatusOr<xla::XlaOp> Contract(xla::XlaOp input, int64_t dim) {
  xla::XlaBuilder* builder = input.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(input));

  if (input_shape.dimensions().back() != 4) {
    return errors::InvalidArgument("Expected last dimension to be 4; got ",
                                   input_shape.dimensions().back());
  }

  // Transpose the input so C is directly followed by VECT_C.
  std::vector<int64_t> permutation;
  auto rank = input_shape.rank();
  permutation.reserve(rank);
  for (int64_t i = 0; i != rank - 1; ++i) {
    permutation.push_back(i);
    if (i == dim) {
      permutation.push_back(rank - 1);
    }
  }

  // Now merge the adjacent dimensions with a reshape.
  std::vector<int64_t> contracted_shape(input_shape.dimensions().begin(),
                                        input_shape.dimensions().end() - 1);
  contracted_shape[dim] *= 4;

  return xla::Reshape(xla::Transpose(input, permutation), contracted_shape);
}

absl::StatusOr<xla::XlaOp> Expand(xla::XlaOp input, int64_t dim) {
  xla::XlaBuilder* builder = input.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(input));

  if (input_shape.dimensions(dim) % 4 != 0) {
    return errors::InvalidArgument(
        "Expected vectorized dimension to be evenly divisible by 4; got ",
        input_shape.dimensions(dim));
  }

  // Split the `dim` into two dimensions with a reshape. The size of the new
  // dimension is always 4.
  std::vector<int64_t> expanded_shape =
      xla::SpanToVector(input_shape.dimensions());
  expanded_shape[dim] /= 4;
  expanded_shape.insert(expanded_shape.begin() + dim + 1, 4);

  // Move the newly created dimension to the end with a transpose.
  std::vector<int64_t> permutation;
  const int64_t expanded_shape_size = expanded_shape.size();
  permutation.reserve(expanded_shape_size);
  for (int64_t i = 0; i != expanded_shape_size; ++i) {
    permutation.push_back(i);
    if (i == dim) {
      ++i;
    }
  }
  permutation.push_back(dim + 1);

  return xla::Transpose(xla::Reshape(input, expanded_shape), permutation);
}

}  // namespace

absl::StatusOr<xla::XlaOp> NCHW_VECT_CToNCHW(xla::XlaOp input) {
  return Contract(input, 1);
}

absl::StatusOr<xla::XlaOp> NCHWToNCHW_VECT_C(xla::XlaOp input) {
  return Expand(input, 1);
}

}  // namespace tensorflow
