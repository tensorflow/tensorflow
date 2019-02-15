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
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp TopK(XlaOp input, int64 k) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;

    Shape iota_shape =
        ShapeUtil::MakeShape(S32, AsInt64Slice(input_shape.dimensions()));
    XlaOp iota_s32 = Iota(builder, iota_shape, last_dim);
    auto input_dims = input_shape.dimensions();
    // TODO(b/122298745): Get rid of Neg() and use CreateScalarGtComputation
    // once the TPU backend supports the comparison computations.
    XlaOp sort_result =
        Sort({Neg(input), iota_s32},
             CreateScalarLtComputation({input_shape.element_type(), S32},
                                       iota_s32.builder()),
             last_dim, /*is_stable=*/true);
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
