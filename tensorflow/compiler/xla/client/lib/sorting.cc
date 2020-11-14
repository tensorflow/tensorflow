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
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp TopK(XlaOp input, int64 k) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;
    int64 last_dim_size = input_shape.dimensions(last_dim);
    // TODO(b/148796364): tune these constants for better performance.
    const int64 kPerPartitionSize = 8192;        // 2^13
    const int64 kLastDimSizeThreshold = 524288;  // 2^19
    const int64 kMinNumPartitions = 8;
    const int64 kMinimalK = 1000;
    if ((k >= kMinimalK) && (k < kPerPartitionSize) &&
        (kPerPartitionSize / k > 2) && last_dim_size >= kLastDimSizeThreshold) {
      int64 num_partitions =
          CeilOfRatio(last_dim_size - k, kPerPartitionSize - k);
      if (num_partitions >= kMinNumPartitions) {
        return TopKWithPartitions(input, k, num_partitions);
      }
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

    auto topk_body_fn =
        [&](XlaOp partition, absl::Span<const XlaOp> values_and_indices,
            XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
      auto values = values_and_indices[0];
      auto indices = values_and_indices[1];
      auto input = values_and_indices[2];
      auto iota_s32 = values_and_indices[3];

      // Slice value and indices for this partition.
      XlaOp start = Mul(Add(partition, ConstantR0<int32>(builder, 1)),
                        ConstantR0<int32>(builder, per_partition_size));
      XlaOp sliced_input =
          DynamicSliceInMinorDims(input, {start}, {per_partition_size});
      XlaOp sliced_indices =
          DynamicSliceInMinorDims(iota_s32, {start}, {per_partition_size});
      // Concat with previous results.
      sliced_input = ConcatInDim(builder, {values, sliced_input}, last_dim);
      sliced_indices =
          ConcatInDim(builder, {indices, sliced_indices}, last_dim);
      // Sort this slice
      XlaOp sort_result =
          Sort({sliced_input, sliced_indices},
               CreateScalarGtComputation({input_shape.element_type(), S32},
                                         sliced_indices.builder()),
               last_dim, true);

      std::vector<int64> start_indices(input_shape.dimensions_size(), 0);
      std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
      std::vector<int64> strides(input_shape.dimensions_size(), 1);
      // Slice topk.
      start_indices[last_dim] = 0;
      limit_indices[last_dim] = k;
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
      return std::vector<XlaOp>{values, indices, input, iota_s32};
    };

    // Get the values and indices for the first topk so that they can
    // be passed to the while loop.
    std::vector<int64> start_indices(input_shape.dimensions_size(), 0);
    std::vector<int64> limit_indices(input_dims.begin(), input_dims.end());
    std::vector<int64> strides(input_shape.dimensions_size(), 1);
    start_indices[last_dim] = 0;
    limit_indices[last_dim] = per_partition_size;
    // Slice value and indices for the first partition.
    XlaOp sliced_input = Slice(input, start_indices, limit_indices, strides);
    XlaOp sliced_indices =
        Slice(iota_s32, start_indices, limit_indices, strides);
    // Sort this slice
    XlaOp sort_result =
        Sort({sliced_input, sliced_indices},
             CreateScalarGtComputation({input_shape.element_type(), S32},
                                       sliced_indices.builder()),
             last_dim, /*is_stable=*/true);

    // Slice topk.
    start_indices[last_dim] = 0;
    limit_indices[last_dim] = k;
    XlaOp values = Slice(GetTupleElement(sort_result, 0), start_indices,
                         limit_indices, strides);
    XlaOp indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                          limit_indices, strides);

    // Pass the result of the first TopK to the while loop and do
    // num_partition - 1 iterations.
    TF_ASSIGN_OR_RETURN(auto values_and_indices,
                        ForEachIndex(num_partitions - 1, S32, topk_body_fn,
                                     {values, indices, input, iota_s32},
                                     "topk_with_partition", builder));
    return Tuple(builder, {values_and_indices[0], values_and_indices[1]});
  });
}

}  // namespace xla
