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

#include "xla/hlo/tools/comparison/tensor_summary_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"

namespace xla {
namespace comparison {

FloatBlockSummary CombineBlockSummaries(
    absl::Span<const int64_t> new_block_indices,
    absl::Span<const FloatBlockSummary> block_summaries) {
  CHECK_GE(block_summaries.size(), 1);

  double total_count = 0;
  double mean = 0;
  double m2 = 0;
  float min_val = std::numeric_limits<float>::infinity();
  float max_val = -std::numeric_limits<float>::infinity();
  double total_nan_count = 0;
  double total_pos_inf_count = 0;
  double total_neg_inf_count = 0;
  double total_zero_count = 0;

  for (const auto& summary : block_summaries) {
    double n1 = total_count;
    double n2 = summary.count;

    if (n2 == 0) {
      continue;
    }

    min_val = std::min(min_val, summary.min);
    max_val = std::max(max_val, summary.max);

    double new_total_count = n1 + n2;
    double delta = summary.mean - mean;
    mean = (n1 * mean + n2 * summary.mean) / new_total_count;

    double var2 = static_cast<double>(summary.stddev) * summary.stddev;
    double m2_2 = var2 * n2;
    m2 = m2 + m2_2 + delta * delta * n1 * n2 / new_total_count;
    total_count = new_total_count;

    total_nan_count += summary.nan_count;
    total_pos_inf_count += summary.pos_inf_count;
    total_neg_inf_count += summary.neg_inf_count;
    total_zero_count += summary.zero_count;
  }

  float final_stddev =
      (total_count > 0) ? static_cast<float>(std::sqrt(m2 / total_count)) : 0;

  FloatBlockSummary summary;
  summary.block_indices =
      std::vector<int64_t>(new_block_indices.begin(), new_block_indices.end());
  summary.min = min_val;
  summary.max = max_val;
  summary.mean = static_cast<float>(mean);
  summary.stddev = final_stddev;
  summary.count = static_cast<float>(total_count);
  summary.nan_count = static_cast<float>(total_nan_count);
  summary.pos_inf_count = static_cast<float>(total_pos_inf_count);
  summary.neg_inf_count = static_cast<float>(total_neg_inf_count);
  summary.zero_count = static_cast<float>(total_zero_count);
  return summary;
}

absl::StatusOr<FloatSummary> GetFloatSummary(
    const Literal& literal, absl::Span<const DimSplitSpec> split_spec) {
  if (!literal.shape().IsArray() ||
      ShapeUtil::IsZeroElementArray(literal.shape())) {
    return absl::InvalidArgumentError(
        "Literal shape is not an array or is a zero element array.");
  }
  absl::Span<const int64_t> dimensions = literal.shape().dimensions();
  if (dimensions.size() - 1 != split_spec.size()) {
    return absl::InvalidArgumentError(
        "Literal shape dimensions size does not match dim split spec size.");
  }
  if (dimensions.back() != kNumStats) {
    return absl::InvalidArgumentError(
        "Literal shape last dimension is not 9. It should be 9 for min, max, "
        "mean, stddev, count, nan_count, pos_inf_count, neg_inf_count, and "
        "zero_count.");
  }

  std::vector<int64_t> block_dims(dimensions.begin(), dimensions.end() - 1);
  std::vector<int64_t> block_indices_vec(block_dims.size(), 0);
  FloatSummary float_summary;
  float_summary.split_spec =
      std::vector<DimSplitSpec>(split_spec.begin(), split_spec.end());

  do {
    std::vector<int64_t> literal_indices = block_indices_vec;
    literal_indices.push_back(0);
    float min_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 1;
    float max_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 2;
    float mean_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 3;
    float stddev_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 4;
    float count_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 5;
    float nan_count_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 6;
    float pos_inf_count_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 7;
    float neg_inf_count_val = literal.Get<float>(literal_indices);
    literal_indices.back() = 8;
    float zero_count_val = literal.Get<float>(literal_indices);
    FloatBlockSummary block_summary;
    block_summary.block_indices = block_indices_vec;
    block_summary.min = min_val;
    block_summary.max = max_val;
    block_summary.mean = mean_val;
    block_summary.stddev = stddev_val;
    block_summary.count = count_val;
    block_summary.nan_count = nan_count_val;
    block_summary.pos_inf_count = pos_inf_count_val;
    block_summary.neg_inf_count = neg_inf_count_val;
    block_summary.zero_count = zero_count_val;
    float_summary.block_summaries.push_back(block_summary);
  } while (IndexUtil::BumpIndices(literal.shape(),
                                  absl::MakeSpan(block_indices_vec)));

  return float_summary;
}

}  // namespace comparison
}  // namespace xla
