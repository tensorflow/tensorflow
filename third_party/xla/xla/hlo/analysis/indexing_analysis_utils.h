/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_UTILS_H_
#define XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_UTILS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/shape.h"

namespace xla {

struct HloInstructionIndexing;

// Computes the indexing map for a Pad operation.
IndexingMap ComputePadIndexingMap(absl::Span<const int64_t> output_dims,
                                  absl::Span<const int64_t> padding_low,
                                  absl::Span<const int64_t> padding_high,
                                  absl::Span<const int64_t> padding_interior,
                                  mlir::MLIRContext* mlir_context);

// Computes the indexing map for a window-based operation (e.g. ReduceWindow,
// Convolution).
IndexingMap ComposeWindowIndexingMap(absl::Span<const int64_t> input_dims,
                                     absl::Span<const int64_t> output_dims,
                                     absl::Span<const int64_t> window_dims,
                                     absl::Span<const int64_t> window_strides,
                                     absl::Span<const int64_t> window_dilations,
                                     absl::Span<const int64_t> base_dilations,
                                     absl::Span<const int64_t> padding,
                                     mlir::MLIRContext* mlir_context);

// Creates an elementwise indexing for num_operands operands with the given
// output shape. All operands use an identity mapping.
HloInstructionIndexing CreateElementwiseIndexing(
    int64_t num_operands, const Shape& output_shape,
    mlir::MLIRContext* mlir_context);

// Creates a scalar (empty) indexing map for the given output shape.
// Used for scalar operands like init values or padding values.
IndexingMap CreateScalarIndexingMap(const Shape& output_shape,
                                    mlir::MLIRContext* mlir_context);

IndexingMap ComputeBroadcastIndexingMap(
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> broadcast_dims, mlir::MLIRContext* mlir_context);

IndexingMap ComputeSliceIndexingMap(absl::Span<const int64_t> output_shape_dims,
                                    absl::Span<const int64_t> slice_starts,
                                    absl::Span<const int64_t> slice_strides,
                                    mlir::MLIRContext* mlir_context);

IndexingMap ComputeReverseIndexingMap(
    absl::Span<const int64_t> output_shape_dims,
    absl::Span<const int64_t> reverse_dims, mlir::MLIRContext* mlir_context);

mlir::AffineMap ComputeTransposeIndexingMap(
    absl::Span<const int64_t> permutation, mlir::MLIRContext* mlir_context);

HloInstructionIndexing ComputeConcatenateIndexing(
    int64_t rank, int64_t concat_dim, absl::Span<const int64_t> output_dims,
    const std::vector<int64_t>& operand_concat_dim_sizes,
    mlir::MLIRContext* mlir_context);

// Computes indexing maps for DotGeneral operands.
// Returns a pair of (lhs_indexing_map, rhs_indexing_map).
std::pair<IndexingMap, IndexingMap> ComputeDotOperandsIndexing(
    absl::Span<const int64_t> lhs_dims, absl::Span<const int64_t> rhs_dims,
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims,
    absl::Span<const int64_t> rhs_contracting_dims,
    mlir::MLIRContext* mlir_context);

// Computes indexing map for reduce input operands.
IndexingMap ComputeReduceInputIndexingMap(absl::Span<const int64_t> input_dims,
                                          absl::Span<const int64_t> output_dims,
                                          absl::Span<const int64_t> reduce_dims,
                                          mlir::MLIRContext* mlir_context);

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_UTILS_H_
