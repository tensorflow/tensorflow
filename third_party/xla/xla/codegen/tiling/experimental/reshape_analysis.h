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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_RESHAPE_ANALYSIS_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_RESHAPE_ANALYSIS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/shape.h"

namespace xla::gpu::experimental {

// Categorizes a "minimal" reshape.
enum class MinimalReshapeCategory {
  kIdentity,       // E.g. [8] -> [8]
  kIncreaseRank,   // E.g. [8] -> [1, 8, 1]
  kDecreaseRank,   // E.g. [1, 8, 1] -> [8]
  kExpandShape,    // E.g. [8] -> [4, 2]
  kCollapseShape,  // E.g. [4, 2] -> [8]
  kGeneric,        // E.g. [2, 5, 7] -> [7, 5, 2] or [8, 16] -> [4, 32]
};

// Represents a contiguous range of dimension IDs.
struct DimensionRange {
  int64_t start;
  int64_t count;

  DimensionRange(int64_t start, int64_t count) : start(start), count(count) {}

  explicit DimensionRange(llvm::ArrayRef<int64_t> ids)
      : start(ids.empty() ? 0 : ids.front()),
        count(static_cast<int64_t>(ids.size())) {}

  bool operator==(const DimensionRange& other) const {
    return start == other.start && count == other.count;
  }

  int64_t end() const { return start + count - 1; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DimensionRange& range) {
    absl::Format(&sink, "[%d, %d]", range.start, range.count);
  }
};

// Represents a "minimal" reshape (subshape from reshape), i.e. a reshape that
// cannot be decomposed into a series of smaller reshapes.
// For example, [8, 4] -> [8, 2, 2] is not a minimal reshape; it has matching
// subshapes [8] -> [8] and [4] -> [2, 2].
struct MinimalReshape {
  DimensionRange input_dim_ids;
  DimensionRange output_dim_ids;
  MinimalReshapeCategory category;

  bool operator==(const MinimalReshape& other) const {
    return input_dim_ids == other.input_dim_ids &&
           output_dim_ids == other.output_dim_ids && category == other.category;
  }

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MinimalReshape& reshape) {
    sink.Append(reshape.ToString());
  }
};

// Returns the category for a single minimal reshape segment.
MinimalReshapeCategory GetMinimalReshapeCategory(
    const Shape& input_shape, const Shape& output_shape,
    llvm::ArrayRef<int64_t> input_ids, llvm::ArrayRef<int64_t> output_ids);

// Scans input and output shapes from left to right in an attempt to find
// subshapes with the same number of elements (minimal reshapes).
//
// For example, [4, 8, 12] -> [32, 3, 4] can be represented as a composition of
// two minimal reshapes:
// 1. [4, 8] -> [32] (CollapseShape)
// 2. [12] -> [3, 4] (ExpandShape)
//
// Size-1 dimension handling:
// Dimensions of size 1 are grouped with adjacent dimensions to form minimal
// reshapes. They are prioritized to be grouped with the subsequent dimension.
// If no subsequent dimension (e.g., they are trailing), they are grouped with
// the preceding dimension.
//
// Example: [8, 1, 8, 1] -> [8, 8] results in:
// 1. [8] -> [8]       (Category: kIdentity)
// 2. [1, 8, 1] -> [8] (Category: kDecreaseRank)
std::vector<MinimalReshape> GetMinimalReshapes(const Shape& input_shape,
                                               const Shape& output_shape);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_RESHAPE_ANALYSIS_H_
