/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/optimize_batch_matmul_utils.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

BatchMatMulDimensionsInfo::BatchMatMulDimensionsInfo(mlir::ShapedType type,
                                                     bool is_lhs)
    : is_lhs_(is_lhs) {
  // BatchMatMulOp has the following shape pattern: B0,...,Bn,L,C and
  // B0,...,Bn,C,R. So, there is only one Contracting dimension and one
  // output dimension.
  const int64_t rank = type.getRank();

  if (is_lhs) {
    contracting_dimensions_.axes.push_back(rank - 1);
    contracting_dimensions_.sizes.push_back(type.getDimSize(rank - 1));
    out_dimensions_.axes.push_back(rank - 2);
    out_dimensions_.sizes.push_back(type.getDimSize(rank - 2));
  } else {
    contracting_dimensions_.axes.push_back(rank - 2);
    contracting_dimensions_.sizes.push_back(type.getDimSize(rank - 2));
    out_dimensions_.axes.push_back(rank - 1);
    out_dimensions_.sizes.push_back(type.getDimSize(rank - 1));
  }
  // Dims 0 and 1 are contracting and output dimensions, hence skipped.
  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    batch_dimensions_.axes.push_back(dim);
    batch_dimensions_.sizes.push_back(type.getDimSize(dim));
  }
}

const DimensionVector& BatchMatMulDimensionsInfo::batch_dimensions() const {
  return batch_dimensions_;
}
const DimensionVector& BatchMatMulDimensionsInfo::contracting_dimensions()
    const {
  return contracting_dimensions_;
}

const DimensionVector& BatchMatMulDimensionsInfo::out_dimensions() const {
  return out_dimensions_;
}

bool BatchMatMulDimensionsInfo::is_lhs() const { return is_lhs_; }

BatchMatMulDimensionsInfo GetBatchMatMulLhsDimensionsInfo(
    mlir::ShapedType type) {
  return BatchMatMulDimensionsInfo(type, /*is_lhs=*/true);
}

BatchMatMulDimensionsInfo GetBatchMatMulRhsDimensionsInfo(
    mlir::ShapedType type) {
  return BatchMatMulDimensionsInfo(type, /*is_lhs=*/false);
}

bool HasFlattenedContractingDims(
    llvm::ArrayRef<int32_t> reshape_input_shape,
    const BatchMatMulDimensionsInfo& bmm_dimensions_info) {
  // Batch dimensions are not flattened and need to match the LHS/RHS of
  // BatchMatMulOp.
  auto batch_dimensions = bmm_dimensions_info.batch_dimensions().SizesArray();
  // The batch dimensions are at the front of the input shape.
  auto reshape_input_shape_batch_dims =
      reshape_input_shape.take_front(batch_dimensions.size());

  if (!llvm::all_of(
          llvm::zip(batch_dimensions, reshape_input_shape_batch_dims),
          [](auto dims) { return std::get<0>(dims) == std::get<1>(dims); })) {
    return false;
  }

  // Out dimensions are assumed to be unflattened and need to match the LHS/RHS
  // of BatchMatMulOp.
  auto out_dimensions = bmm_dimensions_info.out_dimensions().SizesArray();
  llvm::ArrayRef<int32_t> reshape_input_shape_out_dims;
  // The out dimensions are at the end of the input shape for LHS and
  // at the front for RHS.
  if (bmm_dimensions_info.is_lhs()) {
    reshape_input_shape_out_dims =
        reshape_input_shape.slice(batch_dimensions.size(), 1);
  } else {
    reshape_input_shape_out_dims =
        reshape_input_shape.take_back(out_dimensions.size());
  }
  if (!llvm::all_of(
          llvm::zip(out_dimensions, reshape_input_shape_out_dims),
          [](auto dims) { return std::get<0>(dims) == std::get<1>(dims); })) {
    return false;
  }

  auto contracting_dimensions =
      bmm_dimensions_info.contracting_dimensions().SizesArray();
  // The contracting dimensions are at the end of the input shape for
  // LHS and at the front for RHS.
  llvm::ArrayRef<int32_t> reshape_input_shape_contracting_dims;
  size_t num_contracting_dims = reshape_input_shape.size() -
                                batch_dimensions.size() - out_dimensions.size();
  if (bmm_dimensions_info.is_lhs()) {
    reshape_input_shape_contracting_dims =
        reshape_input_shape.take_back(num_contracting_dims);
  } else {
    reshape_input_shape_contracting_dims = reshape_input_shape.slice(
        batch_dimensions.size(), num_contracting_dims);
  }

  return (std::accumulate(reshape_input_shape_contracting_dims.begin(),
                          reshape_input_shape_contracting_dims.end(), 1,
                          std::multiplies<int64_t>()) ==
          contracting_dimensions[0]);
}

bool HasFlattenedOutDims(llvm::ArrayRef<int32_t> reshape_input_shape,
                         const BatchMatMulDimensionsInfo& bmm_dimensions_info) {
  // Batch dimensions are not flattened and need to match the LHS/RHS of
  // BatchMatMulOp.
  auto batch_dimensions = bmm_dimensions_info.batch_dimensions().SizesArray();
  // The batch dimensions are at the front of the input shape.
  auto reshape_input_shape_batch_dims =
      reshape_input_shape.take_front(batch_dimensions.size());
  if (!llvm::all_of(
          llvm::zip(batch_dimensions, reshape_input_shape_batch_dims),
          [](auto dims) { return std::get<0>(dims) == std::get<1>(dims); })) {
    return false;
  }

  auto contracting_dimensions =
      bmm_dimensions_info.contracting_dimensions().SizesArray();
  // The contracting dimensions are at the end of the input shape for
  // LHS and at the front for RHS.
  llvm::ArrayRef<int32_t> reshape_input_shape_contracting_dims;
  if (bmm_dimensions_info.is_lhs()) {
    reshape_input_shape_contracting_dims =
        reshape_input_shape.take_back(contracting_dimensions.size());
  } else {
    reshape_input_shape_contracting_dims =
        reshape_input_shape.slice(batch_dimensions.size(), 1);
  }
  if (!llvm::all_of(
          llvm::zip(contracting_dimensions,
                    reshape_input_shape_contracting_dims),
          [](auto dims) { return std::get<0>(dims) == std::get<1>(dims); })) {
    return false;
  }

  auto out_dimensions = bmm_dimensions_info.out_dimensions().SizesArray();
  // The out dimensions are at the end of the input shape for LHS and
  // at the front for RHS.
  llvm::ArrayRef<int32_t> reshape_input_shape_out_dims;
  size_t num_out_dims = reshape_input_shape.size() - batch_dimensions.size() -
                        contracting_dimensions.size();
  if (bmm_dimensions_info.is_lhs()) {
    reshape_input_shape_out_dims =
        reshape_input_shape.slice(batch_dimensions.size(), num_out_dims);
  } else {
    reshape_input_shape_out_dims = reshape_input_shape.take_back(num_out_dims);
  }

  return (std::accumulate(reshape_input_shape_out_dims.begin(),
                          reshape_input_shape_out_dims.end(), 1,
                          std::multiplies<int64_t>()) == out_dimensions[0]);
}

std::tuple<std::pair<int, int>, std::pair<int, int>>
GetTransposedGroupsIndexRange(llvm::ArrayRef<int32_t> transpose_permutation) {
  // If the input vector is empty, return None for both pairs.
  if (transpose_permutation.empty()) {
    return {{-1, -1}, {-1, -1}};  // Use -1 to indicate None
  }

  int group_one_end_idx = -1;
  for (int i = 0; i < transpose_permutation.size(); ++i) {
    if (transpose_permutation[i] == i) {
      group_one_end_idx = i;
    } else {
      break;
    }
  }

  // If all dimensions are batch dimensions, i.e. the first group is a
  // monotonically increasing sequence, return None for both remaining groups.
  if (group_one_end_idx == transpose_permutation.size() - 1) {
    return {{-1, -1}, {-1, -1}};
  }

  int group_two_start_idx = group_one_end_idx + 1;
  int group_two_end_idx = group_two_start_idx;
  int group_three_start_idx = -1;
  int group_three_end_idx = -1;

  int group_two_end_idx_value = transpose_permutation.size() - 1;
  int group_three_start_idx_value = group_one_end_idx + 1;

  for (int i = group_two_start_idx + 1; i < transpose_permutation.size(); ++i) {
    if (transpose_permutation[i] > group_two_end_idx_value ||
        transpose_permutation[i] <= group_three_start_idx_value ||
        (transpose_permutation[i] != transpose_permutation[i - 1] + 1)) {
      break;
    }
    group_two_end_idx = i;
  }

  group_three_start_idx = group_two_end_idx + 1;
  group_three_end_idx = transpose_permutation.size() - 1;
  // Fail if the last group is not a monotonically increasing sequence.
  for (int i = group_three_start_idx + 1; i < transpose_permutation.size();
       ++i) {
    if (transpose_permutation[i] != transpose_permutation[i - 1] + 1) {
      return {{-1, -1}, {-1, -1}};
    }
  }

  // Handle edge cases where start index might be greater than end index.
  if (group_two_start_idx > group_two_end_idx) {
    group_two_start_idx = group_two_end_idx;
  }

  if (group_three_start_idx > group_three_end_idx) {
    group_three_start_idx = group_three_end_idx;
  }
  if (group_three_start_idx >= transpose_permutation.size()) {
    group_three_start_idx = -1;
    group_three_end_idx = -1;
  }

  return {{group_two_start_idx, group_two_end_idx},
          {group_three_start_idx, group_three_end_idx}};
}

bool HasTransposedContractingAndOutDims(
    llvm::ArrayRef<int32_t> transpose_input_shape,
    llvm::ArrayRef<int32_t> transpose_permutation,
    const BatchMatMulDimensionsInfo& bmm_dimensions_info) {
  std::tuple<std::pair<int, int>, std::pair<int, int>>
      transposed_groups_index_range =
          GetTransposedGroupsIndexRange(transpose_permutation);
  // Return false if the transpose_permutation is not valid.
  if (std::get<0>(transposed_groups_index_range).first == -1 ||
      std::get<0>(transposed_groups_index_range).second == -1 ||
      std::get<1>(transposed_groups_index_range).first == -1 ||
      std::get<1>(transposed_groups_index_range).second == -1) {
    return false;
  }

  // Check if the broadcast dimensions match the batch dimensions of
  // BatchMatMulOp.
  if (!bmm_dimensions_info.batch_dimensions().AxesArray().empty() &&
      bmm_dimensions_info.batch_dimensions().AxesArray().back() !=
          std::get<0>(transposed_groups_index_range).first - 1) {
    return false;
  }

  // Accumulating the sizes of the transposed groups should match the sizes of
  // the contracting and out dimensions of BatchMatMulOp.
  int64_t group_two_dims_size = 1;
  int64_t group_three_dims_size = 1;
  for (int i = std::get<0>(transposed_groups_index_range).first;
       i <= std::get<0>(transposed_groups_index_range).second; ++i) {
    group_two_dims_size *= transpose_input_shape[transpose_permutation[i]];
  }
  for (int i = std::get<1>(transposed_groups_index_range).first;
       i <= std::get<1>(transposed_groups_index_range).second; ++i) {
    group_three_dims_size *= transpose_input_shape[transpose_permutation[i]];
  }

  const auto& out_dims = bmm_dimensions_info.out_dimensions().SizesArray()[0];
  const auto& contracting_dims =
      bmm_dimensions_info.contracting_dimensions().SizesArray()[0];

  return bmm_dimensions_info.is_lhs()
             ? (group_two_dims_size == out_dims &&
                group_three_dims_size == contracting_dims)
             : (group_two_dims_size == contracting_dims &&
                group_three_dims_size == out_dims);
}
}  // namespace TFL
}  // namespace mlir
