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

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace tensorflow {

xla::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                      absl::Span<int64 const> output_dims) {
  xla::XlaBuilder* builder = input.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(input));
  absl::Span<int64 const> input_dims =
      xla::AsInt64Slice(input_shape.dimensions());

  if (input_dims == output_dims) {
    return input;
  }

  if (input_dims.size() > output_dims.size()) {
    return errors::InvalidArgument(
        "Input shape (", xla::ShapeUtil::HumanString(input_shape),
        ") must have rank less than or equal to the output shape [",
        absl::StrJoin(output_dims, ","), "]");
  }

  std::vector<int64> broadcast_dims;
  std::vector<int64> broadcast_shape;
  auto input_it = input_dims.rbegin();
  for (auto output_it = output_dims.rbegin(); output_it != output_dims.rend();
       ++output_it) {
    if (input_it != input_dims.rend()) {
      if (!(*output_it == 0 && *input_it == 0) &&
          !(*input_it != 0 && *output_it % *input_it == 0)) {
        return errors::InvalidArgument("Invalid shape broadcast from ",
                                       xla::ShapeUtil::HumanString(input_shape),
                                       " to [", absl::StrJoin(output_dims, ","),
                                       "]");
      }

      broadcast_dims.push_back(broadcast_shape.size());
      if (*output_it == *input_it) {
        broadcast_shape.push_back(*output_it);
      } else if (*output_it != *input_it) {
        // Add dimensions [I, O/I], which we will later flatten to just
        // [O]. We must do this in two phases since XLA broadcasting does not
        // support tiling.
        broadcast_shape.push_back(*input_it);
        broadcast_shape.push_back(*output_it / *input_it);
      }
      ++input_it;
    } else {
      broadcast_shape.push_back(*output_it);
    }
  }
  TF_RET_CHECK(input_it == input_dims.rend());

  absl::c_reverse(broadcast_dims);
  int broadcast_shape_size = broadcast_shape.size();
  for (int64& broadcast_dim : broadcast_dims) {
    broadcast_dim = broadcast_shape_size - broadcast_dim - 1;
  }
  absl::c_reverse(broadcast_shape);
  xla::XlaOp output =
      xla::BroadcastInDim(input, broadcast_shape, broadcast_dims);
  if (broadcast_shape != output_dims) {
    output = xla::Reshape(output, output_dims);
  }
  return output;
}

}  // namespace tensorflow
