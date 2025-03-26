/* Copyright 2022 The OpenXLA Authors.

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

// This file implements utilities for the canonicalization of ScatterOp and
// GatherOp.

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_GATHER_UTILS_H_
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_GATHER_UTILS_H_

#include <utility>

#include "mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {

// Checks if the scatter has the following characteristics:
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [0, 1, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
bool isCanonicalScatter(ScatterOp scatterOp);

// Checks if the gather has the following characteristics:
// - start_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - collapsed_slice_dims is []
// - offset_dims is [1, 2, ...]
// - start_index_map is [0, 1, ...]
bool isCanonicalGather(GatherOp gatherOp);

// Expands the shape of `tensor`, inserting degenerate dimensions.
//
// For example tensor<10x4xf32> and dimsToInsert = {0, 2}
// will result in tensor<1x10x1x4xf32>.
Value insertDegenerateDimensions(OpBuilder& b, Location loc, Value tensor,
                                 ArrayRef<int64_t> dimsToInsert);

// Given a map from index vector positions to dimension numbers, creates a
// permutation that when applied to the operand, let you replace the map with
// the identity permutation. Also returns its inverse. In gather, the map is
// called `start_index_map`. In scatter, it's `scatter_dims_to_operand_dims`.
std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
makeOperandStartIndexPermutations(ArrayRef<int64_t> dimMap, int operandRank);

// Insert transposes and reshapes to bring `indices` to the 2D shape, where
// the dim0 is the product of all dimensions that are not equal to
// `indexVectorDim` and dim1 is the index vector dim.
//
// Examples.
//
// [a, I, b] will be transposed to [a, b, I], then reshaped into [ab, I].
// [a, b] will be reshaped to [a, b, I(1)] and then reshaped into [ab, I(1)].
Value canonicalizeStartIndices(OpBuilder& b, Location loc, Value indices,
                               int64_t indexVectorDim);

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MHLO_SCATTER_GATHER_UTILS_H_
