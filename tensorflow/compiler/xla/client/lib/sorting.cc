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

#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"

namespace xla {

XlaOp TopK(XlaOp input, int64 k) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;
    int last_dim_size = input_shape.dimensions(last_dim);

    XlaOp iota_s32 = Iota(builder, S32, last_dim_size);
    auto input_dims = input_shape.dimensions();
    std::vector<int64> broadcast_dims(input_dims.begin(), input_dims.end() - 1);
    XlaOp broadcast_s32 = Broadcast(iota_s32, broadcast_dims);
    XlaOp sort_result = Sort(Neg(input), broadcast_s32);
    std::vector<int64> start_indices(input_shape.dimensions_size(), 0);
    std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
    limit_indices[last_dim] = k;
    std::vector<int64> strides(input_shape.dimensions_size(), 1);

    XlaOp values = Neg(Slice(GetTupleElement(sort_result, 0), start_indices,
                             limit_indices, strides));
    XlaOp indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                          limit_indices, strides);
    return Tuple(builder, {values, indices});
  });
}

}  // namespace xla
