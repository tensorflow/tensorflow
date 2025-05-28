/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/sorting.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/loops.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

XlaOp TopK(XlaOp input, int64_t k, PrimitiveType index_type) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions().size() - 1;
    int64_t last_dim_size = input_shape.dimensions(last_dim);
    // TODO(b/148796364): tune these constants for better performance.
    const int64_t kPerPartitionSize = 8192;        // 2^13
    const int64_t kLastDimSizeThreshold = 524288;  // 2^19
    const int64_t kMinNumPartitions = 8;
    const int64_t kMinimalK = 1000;
    if ((k >= kMinimalK) && (k < kPerPartitionSize) &&
        (kPerPartitionSize / k > 2) && last_dim_size >= kLastDimSizeThreshold) {
      int64_t num_partitions =
          CeilOfRatio(last_dim_size - k, kPerPartitionSize - k);
      if (num_partitions >= kMinNumPartitions) {
        return TopKWithPartitions(input, k, num_partitions, index_type);
      }
    }

    Shape iota_shape =
        ShapeUtil::MakeValidatedShape(index_type, input_shape.dimensions())
            .value();
    XlaOp iota = Iota(builder, iota_shape, last_dim);
    for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota = SetDimensionSize(iota, GetDimensionSize(input, i), i);
      }
    }
    auto input_dims = input_shape.dimensions();

    // We can pack BF16 values to be sorted along with their index values into a
    // single 32-bit value in some cases.
    constexpr int32_t kLow16BitsLimit = int32_t{1} << 16;
    constexpr int32_t kLow16BitsMask = kLow16BitsLimit - 1;
    constexpr int32_t kHigh16BitsMask = ~kLow16BitsMask;

    // Whether to use the packed sorting algorithm for BF16 data. This change is
    // good in general, and enables a separate TPU optimization for common cases
    // as well (top-k for small k).
    constexpr int kMaxLastDimSizeForSmallBatches = 1500;
    constexpr int kSmallBatchSizeThreshold = 8;
    const bool use_packed_bf16_sort =
        (input_shape.element_type() == BF16 &&
         last_dim_size < kLow16BitsLimit &&
         (last_dim_size < kMaxLastDimSizeForSmallBatches ||
          (input_shape.dimensions().size() == 2 &&
           input_shape.dimensions(0) >= kSmallBatchSizeThreshold)));

    std::vector<int64_t> start_indices(input_shape.dimensions().size(), 0);
    std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
    limit_indices[last_dim] = k;
    std::vector<int64_t> strides(input_shape.dimensions().size(), 1);

    XlaOp values;
    XlaOp indices;
    if (use_packed_bf16_sort) {
      // Converts a 32-bit value from sign-magnitude (used for floats) to one's
      // complement (easy to compare using integer operations) or vice versa.
      auto sign_magnitude_to_from_ones_complement = [builder](const XlaOp in) {
        constexpr int32_t kAllNonSignBits = 0x7fffffff;
        XlaOp in_s32 = BitcastConvertType(in, S32);
        return Xor(
            And(in_s32, ConstantR0<int32_t>(builder, kAllNonSignBits)),
            ShiftRightArithmetic(in_s32, ConstantR0<int32_t>(builder, 31)));
      };

      // Move input values to the high 16 bits of each 32-bit element, convert
      // them to allow integer comparisons, set the low 16 bits to one (in order
      // to reverse the sort order of the element indices), then XOR in the iota
      // result. This leads to the ones' complement version of the BF16 input in
      // the high 16 bits and the ones' complement of the indices in the low 16
      // bits.
      XlaOp input_f32_trimmed =
          Or(sign_magnitude_to_from_ones_complement(
                 BitcastConvertType(ConvertElementType(input, F32), S32)),
             ConstantR0<int32_t>(builder, kLow16BitsMask));
      XlaOp input_and_iota = Xor(input_f32_trimmed, iota);

      // Sort in reverse order so the largest elements are at the beginning.
      // Breaking ties here is why the index bits need to be inverted.
      XlaOp sort_result_raw =
          Sort({input_and_iota},
               CreateScalarGtComputation({index_type}, builder), last_dim,
               /*is_stable=*/false);

      // Slice off the first k values.
      sort_result_raw =
          Slice(sort_result_raw, start_indices, limit_indices, strides);
      // The k in TopK is static so we shouldn't generate a dynamic dimension
      // even if input is dynamic.
      sort_result_raw = RemoveDynamicDimension(sort_result_raw, last_dim);

      // Get the high 16 bits of each value from the sorted result and convert
      // them back to BF16.
      values = ConvertElementType(
          BitcastConvertType(
              And(sign_magnitude_to_from_ones_complement(sort_result_raw),
                  ConstantR0<int32_t>(builder, kHigh16BitsMask)),
              F32),
          BF16);

      // Get the index values from the low 16 bits of each value and invert them
      // again.
      indices = And(
          Xor(sort_result_raw, ConstantR0<int32_t>(builder, kLow16BitsMask)),
          ConstantR0<int32_t>(builder, kLow16BitsMask));
    } else {
      XlaOp sort_result =
          Sort({input, iota},
               CreateScalarGtComputation(
                   {input_shape.element_type(), index_type}, iota.builder()),
               last_dim, /*is_stable=*/true);
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      // The k in TopK is static so we shouldn't generate a dynamic dimension
      // even if input is dynamic.
      values = RemoveDynamicDimension(values, last_dim);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
      indices = RemoveDynamicDimension(indices, last_dim);
    }

    return Tuple(builder, {values, indices});
  });
}

XlaOp TopKWithPartitions(XlaOp input, int64_t k, int64_t num_partitions,
                         PrimitiveType index_type) {
  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions().size() - 1;
    // Calculate per partition size.
    auto input_dims = input_shape.dimensions();
    int64_t last_dim_size = input_shape.dimensions(last_dim);
    const int64_t per_partition_size =
        CeilOfRatio(last_dim_size, num_partitions);
    // Do normal TopK when per partition size is smaller than or equal to k.
    if (k >= per_partition_size) {
      return TopK(input, k, index_type);
    }

    Shape iota_shape =
        ShapeUtil::MakeValidatedShape(index_type, input_shape.dimensions())
            .value();
    XlaOp iota = Iota(builder, iota_shape, last_dim);
    for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota = SetDimensionSize(iota, GetDimensionSize(input, i), i);
      }
    }

    auto topk_body_fn =
        [&](XlaOp partition, absl::Span<const XlaOp> values_and_indices,
            XlaBuilder* builder) -> absl::StatusOr<std::vector<XlaOp>> {
      auto values = values_and_indices[0];
      auto indices = values_and_indices[1];
      auto input = values_and_indices[2];
      auto iota = values_and_indices[3];

      // Slice value and indices for this partition.
      XlaOp start =
          Mul(Add(partition, One(builder, index_type)),
              ConstantR0WithType(builder, index_type, per_partition_size));
      XlaOp sliced_input =
          DynamicSliceInMinorDims(input, {start}, {per_partition_size});
      XlaOp sliced_indices =
          DynamicSliceInMinorDims(iota, {start}, {per_partition_size});
      // Concat with previous results.
      sliced_input = ConcatInDim(builder, {values, sliced_input}, last_dim);
      sliced_indices =
          ConcatInDim(builder, {indices, sliced_indices}, last_dim);
      // Sort this slice
      XlaOp sort_result = Sort(
          {sliced_input, sliced_indices},
          CreateScalarGtComputation({input_shape.element_type(), index_type},
                                    sliced_indices.builder()),
          last_dim, true);

      std::vector<int64_t> start_indices(input_shape.dimensions().size(), 0);
      std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
      std::vector<int64_t> strides(input_shape.dimensions().size(), 1);
      // Slice topk.
      start_indices[last_dim] = 0;
      limit_indices[last_dim] = k;
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
      return std::vector<XlaOp>{values, indices, input, iota};
    };

    // Get the values and indices for the first topk so that they can
    // be passed to the while loop.
    std::vector<int64_t> start_indices(input_shape.dimensions().size(), 0);
    std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
    std::vector<int64_t> strides(input_shape.dimensions().size(), 1);
    start_indices[last_dim] = 0;
    limit_indices[last_dim] = per_partition_size;
    // Slice value and indices for the first partition.
    XlaOp sliced_input = Slice(input, start_indices, limit_indices, strides);
    XlaOp sliced_indices = Slice(iota, start_indices, limit_indices, strides);
    // Sort this slice
    XlaOp sort_result =
        Sort({sliced_input, sliced_indices},
             CreateScalarGtComputation({input_shape.element_type(), index_type},
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
    TF_ASSIGN_OR_RETURN(
        auto values_and_indices,
        ForEachIndex(num_partitions - 1, index_type, topk_body_fn,
                     {values, indices, input, iota}, "topk_with_partition",
                     builder));
    return Tuple(builder, {values_and_indices[0], values_and_indices[1]});
  });
}

}  // namespace xla
