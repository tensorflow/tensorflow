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

#include "tensorflow/compiler/xla/client/lib/broadcast.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

StatusOr<XlaOp> BroadcastTo(XlaOp input,
                            absl::Span<int64_t const> output_dims) {
  XlaBuilder* builder = input.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
  absl::Span<int64_t const> input_dims = input_shape.dimensions();

  if (input_dims == output_dims) {
    return input;
  }

  if (input_dims.size() > output_dims.size()) {
    return tensorflow::errors::InvalidArgument(
        "Input shape (", ShapeUtil::HumanString(input_shape),
        ") must have rank less than or equal to the output shape [",
        absl::StrJoin(output_dims, ","), "]");
  }

  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> broadcast_shape;
  auto input_it = input_dims.rbegin();
  for (auto output_it = output_dims.rbegin(); output_it != output_dims.rend();
       ++output_it) {
    if (input_it != input_dims.rend()) {
      if (!(*output_it == 0 && *input_it == 0) &&
          !(*input_it != 0 && *output_it % *input_it == 0)) {
        return tensorflow::errors::InvalidArgument(
            "Invalid shape broadcast from ",
            ShapeUtil::HumanString(input_shape), " to [",
            absl::StrJoin(output_dims, ","), "]");
      }

      broadcast_dims.push_back(broadcast_shape.size());
      if (*output_it == *input_it || *input_it == 1) {
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
  for (int64_t& broadcast_dim : broadcast_dims) {
    broadcast_dim = broadcast_shape_size - broadcast_dim - 1;
  }
  absl::c_reverse(broadcast_shape);
  XlaOp output = BroadcastInDim(input, broadcast_shape, broadcast_dims);
  if (broadcast_shape != output_dims) {
    output = Reshape(output, output_dims);
  }
  return output;
}

}  // namespace xla
