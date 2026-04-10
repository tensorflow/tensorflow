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

#include "xla/codegen/tiling/experimental/reshape_analysis.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::SmallVector;

// Returns dimensions with size > 1.
llvm::SmallVector<int64_t> GetNonOneDimensions(const Shape& shape,
                                               llvm::ArrayRef<int64_t> ids) {
  llvm::SmallVector<int64_t> non_one_dims;
  for (int64_t id : ids) {
    if (shape.dimensions(id) > 1) {
      non_one_dims.push_back(shape.dimensions(id));
    }
  }
  return non_one_dims;
}

}  // namespace

MinimalReshapeCategory GetMinimalReshapeCategory(
    const Shape& input_shape, const Shape& output_shape,
    llvm::ArrayRef<int64_t> input_ids, llvm::ArrayRef<int64_t> output_ids) {
  llvm::SmallVector<int64_t> input_non_one =
      GetNonOneDimensions(input_shape, input_ids);
  llvm::SmallVector<int64_t> output_non_one =
      GetNonOneDimensions(output_shape, output_ids);

  if (input_non_one == output_non_one) {
    if (input_ids.size() < output_ids.size()) {
      return MinimalReshapeCategory::kIncreaseRank;
    }
    if (input_ids.size() > output_ids.size()) {
      return MinimalReshapeCategory::kDecreaseRank;
    }
    return MinimalReshapeCategory::kIdentity;
  }

  if (input_non_one.size() == 1 && output_non_one.size() > 1) {
    return MinimalReshapeCategory::kExpandShape;
  }

  if (input_non_one.size() > 1 && output_non_one.size() == 1) {
    return MinimalReshapeCategory::kCollapseShape;
  }

  return MinimalReshapeCategory::kGeneric;
}

std::vector<MinimalReshape> GetMinimalReshapes(const Shape& input_shape,
                                               const Shape& output_shape) {
  CHECK_EQ(ShapeUtil::ElementsIn(input_shape),
           ShapeUtil::ElementsIn(output_shape))
      << "Input and output shapes must have the same number of elements.";

  std::vector<MinimalReshape> reshapes;
  int64_t input_dim_id = 0;
  int64_t output_dim_id = 0;
  int64_t input_num_elements = 1;
  int64_t output_num_elements = 1;
  SmallVector<int64_t> input_ids;
  SmallVector<int64_t> output_ids;

  const int64_t input_rank = input_shape.dimensions().size();
  const int64_t output_rank = output_shape.dimensions().size();
  while (input_dim_id < input_rank || output_dim_id < output_rank ||
         !input_ids.empty() || !output_ids.empty()) {
    if (input_dim_id < input_rank &&
        (input_ids.empty() || input_num_elements < output_num_elements)) {
      input_num_elements *= input_shape.dimensions(input_dim_id);
      input_ids.push_back(input_dim_id++);
      continue;
    }

    if (output_dim_id < output_rank &&
        (output_ids.empty() || output_num_elements < input_num_elements)) {
      output_num_elements *= output_shape.dimensions(output_dim_id);
      output_ids.push_back(output_dim_id++);
      continue;
    }

    // Once one shape is fully processed, absorb any remaining size-1 dimensions
    // from the other into the current minimal reshape. This groups trivial
    // dimensions, e.g., [1, 8, 1] -> [8] is a single minimal reshape.
    if (input_dim_id == input_rank) {
      while (output_dim_id < output_rank &&
             output_shape.dimensions(output_dim_id) == 1) {
        output_ids.push_back(output_dim_id++);
      }
    }
    if (output_dim_id == output_rank) {
      while (input_dim_id < input_rank &&
             input_shape.dimensions(input_dim_id) == 1) {
        input_ids.push_back(input_dim_id++);
      }
    }

    MinimalReshapeCategory category = GetMinimalReshapeCategory(
        input_shape, output_shape, input_ids, output_ids);
    reshapes.push_back(
        {DimensionRange(input_ids), DimensionRange(output_ids), category});
    input_ids.clear();
    output_ids.clear();
    input_num_elements = 1;
    output_num_elements = 1;
  }

  return reshapes;
}

}  // namespace xla::gpu::experimental
