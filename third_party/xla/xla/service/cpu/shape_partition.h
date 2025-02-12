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

#ifndef XLA_SERVICE_CPU_SHAPE_PARTITION_H_
#define XLA_SERVICE_CPU_SHAPE_PARTITION_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/shape.h"

namespace xla {
namespace cpu {

// ShapePartitionAssigner partitions the most-major dimensions of 'shape' such
// that the total partition count <= 'target_partition_count'.
//
// Example 1:
//
//   Let 'shape' = [8, 16, 32] and 'target_partition_count' = 6.
//
//   Because the most-major dimension size is <= 'target_partition_count', we
//   can generate our target number of partitions by partition the most-major
//   dimensions.
//
//   This will result in the following partitions of the most-major dimension:
//
//     [0, 1), [1, 2), [2, 3), [3, 4), [4, 5) [5, 8)
//
//   Note that the last partition has residual because the dimension size is
//   not a multiple of the partition count.
//
//
// Example 2:
//
//   Let 'shape' = [8, 16, 32] and 'target_partition_count' = 16.
//
//   Because the most-major dimension only has size 8, we must also partition
//   the next most-major dimension to generate the target of 16 partitions.
//   We factor 'target_partition_count' by the number of most-major dimensions
//   we need to partition, to get a per-dimension target partition count:
//
//     target_dimension_partition_count = 16 ^ (1 / 2) == 4
//
//   This will result in the following partitions of the most-major dimension:
//
//     [0, 2), [2, 4), [4, 6), [6, 8)
//
//   This will result in the following partitions of the second most-major
//   dimension:
//
//     [0, 4), [4, 8), [8, 12), [12, 16)
//
class ShapePartitionAssigner {
 public:
  explicit ShapePartitionAssigner(const Shape& shape) : shape_(shape) {}

  // Returns dimension partition counts (starting at outermost dimension).
  std::vector<int64_t> Run(int64_t target_partition_count);

  // Returns the total partition count based on 'dimension_partition_counts'.
  static int64_t GetTotalPartitionCount(
      const std::vector<int64_t>& dimension_partition_counts);

 private:
  const Shape& shape_;
};

// ShapePartitionIterator iterates through outer-dimension partitions of
// 'shape' as specified by 'dimension_partition_counts'.
class ShapePartitionIterator {
 public:
  ShapePartitionIterator(const Shape& shape,
                         absl::Span<const int64_t> dimension_partition_counts);

  // Returns a partition [start, size] for each dimension.
  // Partitions are listed starting from outer-most dimension first.
  std::vector<std::pair<int64_t, int64_t>> GetPartition(int64_t index) const;

  int64_t GetTotalPartitionCount() const;

 private:
  const Shape& shape_;
  const std::vector<int64_t> dimension_partition_counts_;

  std::vector<int64_t> dimensions_;
  std::vector<int64_t> dimension_partition_sizes_;
  std::vector<int64_t> dimension_partition_strides_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_SHAPE_PARTITION_H_
