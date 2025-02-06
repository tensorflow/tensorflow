/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/shape_partition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

namespace xla {
namespace cpu {

std::vector<int64_t> ShapePartitionAssigner::Run(
    int64_t target_partition_count) {
  // Gather outermost dims where dim_size >= 'target_partition_count'.
  // This may include the inner-dim as LLVM can vectorize loops with dynamic
  // bounds.
  std::vector<int64_t> outer_dims;
  int64_t outer_dim_size = 1;
  // TODO(b/27458679) Consider reserving enough minor dimensions (based on
  // target vector register width) to enable vector instructions.
  for (int i = shape_.layout().minor_to_major_size() - 1; i >= 0; --i) {
    const int64_t dimension = shape_.layout().minor_to_major(i);
    outer_dims.push_back(dimension);
    outer_dim_size *= shape_.dimensions(dimension);
    if (outer_dim_size >= target_partition_count) {
      break;
    }
  }

  // Clip target partition count if outer dim size is insufficient to cover.
  target_partition_count = std::min(outer_dim_size, target_partition_count);

  // Calculate the target number of partitions per-dimension, by factoring
  // 'target_partition_count' into 'num_outer_dims' equal terms.
  // EX:
  // *) target_partition_count = 16
  // *) out_dim_count = 2
  // *) target_dim_partition_count = 16 ^ (1.0 / 2) == 4
  const int64_t target_dim_partition_count = std::pow(
      static_cast<double>(target_partition_count), 1.0 / outer_dims.size());

  // Assign feasible dimension partitions based on 'target_dim_partition_count'
  // and actual dimension sizes from 'shape_'.
  std::vector<int64_t> dimension_partition_counts(outer_dims.size());
  for (int64_t i = 0; i < outer_dims.size(); ++i) {
    dimension_partition_counts[i] =
        std::min(static_cast<int64_t>(shape_.dimensions(outer_dims[i])),
                 target_dim_partition_count);
  }

  // Check if total partition count is below 'target_partition_count'.
  // This can occur if some dimensions in 'shape_' are below the
  // 'target_dim_partition_count' threshold.
  if (GetTotalPartitionCount(dimension_partition_counts) <
      target_partition_count) {
    // Assign additional partitions (greedily to outer dimensions), if doing
    // so would keep the total number of partitions <= 'target_partition_count',
    // using one pass over 'dimension_partition_counts'.
    for (int64_t i = 0; i < dimension_partition_counts.size(); ++i) {
      const int64_t current_dim_partition_count = dimension_partition_counts[i];
      const int64_t other_dims_partition_count =
          GetTotalPartitionCount(dimension_partition_counts) /
          current_dim_partition_count;
      // Constraint: (current + additional) * other <= target
      // Calculate: additional = target / other - current
      int64_t additional_partition_count =
          target_partition_count / other_dims_partition_count -
          current_dim_partition_count;
      // Clip 'additional_partition_count' by current dimension size.
      additional_partition_count = std::min(
          shape_.dimensions(outer_dims[i]) - dimension_partition_counts[i],
          additional_partition_count);
      if (additional_partition_count > 0) {
        dimension_partition_counts[i] += additional_partition_count;
      }
    }
  }

  return dimension_partition_counts;
}

int64_t ShapePartitionAssigner::GetTotalPartitionCount(
    const std::vector<int64_t>& dimension_partition_counts) {
  int64_t total_partition_count = 1;
  for (int64_t dim_partition_count : dimension_partition_counts) {
    total_partition_count *= dim_partition_count;
  }
  return total_partition_count;
}

ShapePartitionIterator::ShapePartitionIterator(
    const Shape& shape, absl::Span<const int64_t> dimension_partition_counts)
    : shape_(shape),
      dimension_partition_counts_(dimension_partition_counts.begin(),
                                  dimension_partition_counts.end()),
      dimensions_(dimension_partition_counts_.size()),
      dimension_partition_sizes_(dimension_partition_counts_.size()),
      dimension_partition_strides_(dimension_partition_counts_.size()) {
  // Store partitioned outer dimensions from 'shape_'.
  for (int i = 0; i < dimensions_.size(); ++i) {
    dimensions_[i] = shape_.layout().minor_to_major(
        shape_.layout().minor_to_major_size() - 1 - i);
  }

  // Calculate partition size for each dimension (note that the size of
  // the last partition in each dimension may be different if the dimension
  // size is not a multiple of partition size).
  for (int i = 0; i < dimension_partition_sizes_.size(); ++i) {
    const int64_t dim_size = shape_.dimensions(dimensions_[i]);
    dimension_partition_sizes_[i] =
        std::max(int64_t{1}, dim_size / dimension_partition_counts_[i]);
  }

  // Calculate the partition strides for each dimension.
  dimension_partition_strides_[dimension_partition_strides_.size() - 1] = 1;
  for (int i = dimension_partition_strides_.size() - 2; i >= 0; --i) {
    dimension_partition_strides_[i] = dimension_partition_strides_[i + 1] *
                                      dimension_partition_counts_[i + 1];
  }
}

std::vector<std::pair<int64_t, int64_t>> ShapePartitionIterator::GetPartition(
    int64_t index) const {
  // Calculate and return the partition for 'index'.
  // Returns for each dimension: (partition_start, partition_size).
  std::vector<std::pair<int64_t, int64_t>> partition(dimensions_.size());
  for (int64_t i = 0; i < partition.size(); ++i) {
    // Calculate the index for dimension 'i'.
    const int64_t partition_index = index / dimension_partition_strides_[i];
    // Calculate dimension partition start at 'partition_index'.
    partition[i].first = partition_index * dimension_partition_sizes_[i];
    // Calculate dimension partition size (note that the last partition size
    // may be adjusted if dimension size is not a multiple of partition size).
    if (partition_index == dimension_partition_counts_[i] - 1) {
      // Last partition in this dimension.
      partition[i].second =
          shape_.dimensions(dimensions_[i]) - partition[i].first;
    } else {
      partition[i].second = dimension_partition_sizes_[i];
    }
    CHECK_GT(partition[i].second, 0);
    // Update index to remove contribution from current dimension.
    index -= partition_index * dimension_partition_strides_[i];
  }
  return partition;
}

int64_t ShapePartitionIterator::GetTotalPartitionCount() const {
  return ShapePartitionAssigner::GetTotalPartitionCount(
      dimension_partition_counts_);
}

}  // namespace cpu
}  // namespace xla
