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
    for (int64 i = 0; i < input_shape.rank(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota_s32 = SetDimensionSize(iota_s32, GetDimensionSize(input, i), i);
      }
    }
    auto input_dims = input_shape.dimensions();
    XlaOp sort_result =
        Sort({input, iota_s32},
             CreateScalarGtComputation({input_shape.element_type(), S32},
                                       iota_s32.builder()),
             last_dim, /*is_stable=*/true);
    std::vector<int64> start_indices(input_shape.dimensions_size(), 0);
    std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
    limit_indices[last_dim] = k;
    std::vector<int64> strides(input_shape.dimensions_size(), 1);

    XlaOp values = Slice(GetTupleElement(sort_result, 0), start_indices,
                         limit_indices, strides);
    XlaOp indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                          limit_indices, strides);
    return Tuple(builder, {values, indices});
  });
}

XlaOp TopKWithPartitions(XlaOp input, int64 k, int64 num_partitions) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;
    // Calculate per partition size.
    auto input_dims = input_shape.dimensions();
    int64 last_dim_size = input_shape.dimensions(last_dim);
    const int64 per_partition_size = CeilOfRatio(last_dim_size, num_partitions);
    // Do normal TopK when per partition size is smaller than or equal to k.
    if (k >= per_partition_size) {
      return TopK(input, k);
    }

    Shape iota_shape =
        ShapeUtil::MakeShape(S32, AsInt64Slice(input_shape.dimensions()));
    XlaOp iota_s32 = Iota(builder, iota_shape, last_dim);
    for (int64 i = 0; i < input_shape.rank(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota_s32 = SetDimensionSize(iota_s32, GetDimensionSize(input, i), i);
      }
    }

    XlaOp values, indices;
    for (int64 partition = 0; partition < num_partitions; partition++) {
      std::vector<int64> start_indices(input_shape.dimensions_size(), 0);
      std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
      std::vector<int64> strides(input_shape.dimensions_size(), 1);
      start_indices[last_dim] = partition * per_partition_size;
      limit_indices[last_dim] =
          std::min((partition + 1) * per_partition_size, last_dim_size);
      // Slice value and indices for this partition..
      XlaOp sliced_input = Slice(input, start_indices, limit_indices, strides);
      XlaOp sliced_indices =
          Slice(iota_s32, start_indices, limit_indices, strides);
      // Concat with previous results.
      if (partition > 0) {
        sliced_input = ConcatInDim(builder, {values, sliced_input}, last_dim);
        sliced_indices =
            ConcatInDim(builder, {indices, sliced_indices}, last_dim);
      }
      // Sort this slice
      XlaOp sort_result =
          Sort({sliced_input, sliced_indices},
               CreateScalarGtComputation({input_shape.element_type(), S32},
                                         sliced_indices.builder()),
               last_dim, /*is_stable=*/true);
      // Slice topk.
      start_indices[last_dim] = 0;
      limit_indices[last_dim] = k;
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
    }
    return Tuple(builder, {values, indices});
  });
}

}  // namespace xla
