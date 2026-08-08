/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_TENSOR_SUMMARY_UTIL_H_
#define XLA_HLO_TOOLS_COMPARISON_TENSOR_SUMMARY_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"

namespace xla {
namespace comparison {

constexpr int64_t kNumStats = 9;

template <typename T>
struct BlockSummary {
  // A vector containing the block indices along each partitioned dimension.
  // The size of this vector is the same as the size of the split_spec
  // vector in the Summary struct.
  std::vector<int64_t> block_indices;
  // A single F32 scalar containing the minimum value of the block.
  T min = {};
  // A single F32 scalar containing the maximum value of the block.
  T max = {};
  // A single F32 scalar containing the mean value of the block.
  T mean = {};
  // A single F32 scalar containing the standard deviation of the block.
  T stddev = {};
  // A single F32 scalar containing the number of elements in the block.
  T count = {};
  // A single F32 scalar containing the number of elements with NaN values in
  // the block.
  T nan_count = {};
  // A single F32 scalar containing the number of elements with positive
  // infinity values in the block.
  T pos_inf_count = {};
  // A single F32 scalar containing the number of elements with negative
  // infinity values in the block.
  T neg_inf_count = {};
  // A single F32 scalar containing the number of elements with zero values in
  // the block.
  T zero_count = {};
};

using FloatBlockSummary = BlockSummary<float>;

struct DimSplitSpec {
  // The logical dimension number indicating which dimension to split.
  int64_t dim_index;
  // The number of blocks along the dimension.
  int64_t block_count;

  bool operator==(const DimSplitSpec& other) const {
    return dim_index == other.dim_index && block_count == other.block_count;
  }
};

template <typename T>
struct Summary {
  // A vector containing the block summaries for each block of the tensor.
  std::vector<BlockSummary<T>> block_summaries;

  // A vector containing the split spec for each dimension. The order of each
  // DimSplitSpec and the `block_count` in them determines the shape of the
  // blocks.
  std::vector<DimSplitSpec> split_spec;
};

using FloatSummary = Summary<float>;

// Combines multiple block summaries into a single block summary. This is
// useful for aggregating summaries, for example, when unsharding.
FloatBlockSummary CombineBlockSummaries(
    absl::Span<const int64_t> new_block_indices,
    absl::Span<const FloatBlockSummary> block_summaries);

// Extracts the float summary from the logged literal and dim split spec.
absl::StatusOr<FloatSummary> GetFloatSummary(
    const Literal& literal, absl::Span<const DimSplitSpec> split_spec);

}  // namespace comparison
}  // namespace xla

#endif  // XLA_HLO_TOOLS_COMPARISON_TENSOR_SUMMARY_UTIL_H_
