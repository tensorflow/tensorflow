/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/analysis/indexing_analysis_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/shape.h"

namespace xla {

using llvm::SmallVector;
using mlir::MLIRContext;

IndexingMap ComputeBroadcastIndexingMap(
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> broadcast_dims, MLIRContext* mlir_context) {
  SmallVector<SymbolicExpr> exprs;
  exprs.reserve(broadcast_dims.size());
  for (int64_t bcast_dim : broadcast_dims) {
    exprs.push_back(CreateDimExpr(bcast_dim, mlir_context));
  }
  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, output_dims.size(), /*num_symbols=*/0,
                       std::move(exprs)),
      output_dims, {});
}

IndexingMap ComputeSliceIndexingMap(absl::Span<const int64_t> output_shape_dims,
                                    absl::Span<const int64_t> slice_starts,
                                    absl::Span<const int64_t> slice_strides,
                                    mlir::MLIRContext* mlir_context) {
  auto rank = output_shape_dims.size();
  SmallVector<SymbolicExpr> exprs;
  exprs.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    SymbolicExpr dim_expr = CreateDimExpr(dim, mlir_context);
    exprs.push_back(dim_expr * slice_strides[dim] + slice_starts[dim]);
  }
  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, rank, /*num_symbols=*/0, std::move(exprs)),
      output_shape_dims, {});
}

IndexingMap ComputeReverseIndexingMap(
    absl::Span<const int64_t> output_shape_dims,
    absl::Span<const int64_t> reverse_dims, mlir::MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reverse_dims_set(reverse_dims.begin(),
                                                reverse_dims.end());
  SmallVector<SymbolicExpr> exprs;
  exprs.reserve(output_shape_dims.size());
  for (auto [output_dim_id, output_dim] : llvm::enumerate(output_shape_dims)) {
    SymbolicExpr dim_expr = CreateDimExpr(output_dim_id, mlir_context);
    exprs.push_back(reverse_dims_set.contains(output_dim_id)
                        ? -dim_expr + (output_dim - 1)
                        : dim_expr);
  }
  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, output_shape_dims.size(),
                       /*num_symbols=*/0, std::move(exprs)),
      output_shape_dims, {});
}

HloInstructionIndexing ComputeConcatenateIndexing(
    int64_t rank, int64_t concat_dim, absl::Span<const int64_t> output_dims,
    const std::vector<int64_t>& operand_concat_dim_sizes,
    mlir::MLIRContext* mlir_context) {
  SymbolicMap symbolic_map =
      SymbolicMap::GetMultiDimIdentityMap(rank, mlir_context);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromTensorSizes(output_dims);

  HloInstructionIndexing concat_indexing;
  concat_indexing.indexing_maps.resize(operand_concat_dim_sizes.size());
  SymbolicExpr concat_dim_expr = CreateDimExpr(concat_dim, mlir_context);
  int64_t offset = 0;
  for (const auto [operand_id, operand_concat_dim] :
       llvm::enumerate(operand_concat_dim_sizes)) {
    symbolic_map.SetResult(concat_dim, concat_dim_expr - offset);
    dim_vars[concat_dim] =
        IndexingMap::Variable{{offset, offset + operand_concat_dim - 1}};
    concat_indexing.indexing_maps[operand_id].insert(
        OperandIndexing(IndexingMap(symbolic_map, dim_vars, /*range_vars=*/{},
                                    /*rt_vars=*/{})));
    offset += operand_concat_dim;
  }
  return concat_indexing;
}

std::pair<IndexingMap, IndexingMap> ComputeDotOperandsIndexing(
    absl::Span<const int64_t> lhs_dims, absl::Span<const int64_t> rhs_dims,
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims,
    absl::Span<const int64_t> rhs_contracting_dims, MLIRContext* mlir_context) {
  SmallVector<SymbolicExpr> lhs_exprs(lhs_dims.size());
  SmallVector<SymbolicExpr> rhs_exprs(rhs_dims.size());
  int64_t output_dim_id = 0;

  // Batch dimensions
  for (auto [lhs_batch_dim, rhs_batch_dim] :
       llvm::zip(lhs_batch_dims, rhs_batch_dims)) {
    SymbolicExpr output_dim_expr = CreateDimExpr(output_dim_id, mlir_context);
    lhs_exprs[lhs_batch_dim] = output_dim_expr;
    rhs_exprs[rhs_batch_dim] = output_dim_expr;
    ++output_dim_id;
  }

  // LHS non-contracting dims
  absl::flat_hash_set<int64_t> lhs_batch_set(lhs_batch_dims.begin(),
                                             lhs_batch_dims.end());
  absl::flat_hash_set<int64_t> lhs_contracting_set(lhs_contracting_dims.begin(),
                                                   lhs_contracting_dims.end());
  for (int64_t i = 0; i < lhs_dims.size(); ++i) {
    if (!lhs_batch_set.contains(i) && !lhs_contracting_set.contains(i)) {
      lhs_exprs[i] = CreateDimExpr(output_dim_id++, mlir_context);
    }
  }

  // RHS non-contracting dims
  absl::flat_hash_set<int64_t> rhs_batch_set(rhs_batch_dims.begin(),
                                             rhs_batch_dims.end());
  absl::flat_hash_set<int64_t> rhs_contracting_set(rhs_contracting_dims.begin(),
                                                   rhs_contracting_dims.end());
  for (int64_t i = 0; i < rhs_dims.size(); ++i) {
    if (!rhs_batch_set.contains(i) && !rhs_contracting_set.contains(i)) {
      rhs_exprs[i] = CreateDimExpr(output_dim_id++, mlir_context);
    }
  }

  // Contracting dimensions (as symbols)
  int64_t symbol_id = 0;
  std::vector<int64_t> symbol_sizes;
  symbol_sizes.reserve(lhs_contracting_dims.size());
  int64_t output_rank = output_dims.size();
  for (auto [lhs_contract, rhs_contract] :
       llvm::zip(lhs_contracting_dims, rhs_contracting_dims)) {
    SymbolicExpr symbol_expr =
        CreateSymbolExpr(symbol_id, output_rank, mlir_context);
    lhs_exprs[lhs_contract] = symbol_expr;
    rhs_exprs[rhs_contract] = symbol_expr;
    symbol_sizes.push_back(lhs_dims[lhs_contract]);
    ++symbol_id;
  }

  return std::make_pair(
      IndexingMap::FromTensorSizes(
          SymbolicMap::Get(mlir_context, output_rank, symbol_sizes.size(),
                           std::move(lhs_exprs)),
          output_dims, symbol_sizes),
      IndexingMap::FromTensorSizes(
          SymbolicMap::Get(mlir_context, output_rank, symbol_sizes.size(),
                           std::move(rhs_exprs)),
          output_dims, symbol_sizes));
}

IndexingMap ComputeReduceInputIndexingMap(absl::Span<const int64_t> input_dims,
                                          absl::Span<const int64_t> output_dims,
                                          absl::Span<const int64_t> reduce_dims,
                                          MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_set(reduce_dims.begin(),
                                               reduce_dims.end());
  std::vector<int64_t> parallel_dims_sizes;
  int64_t output_dim_id = 0;
  SmallVector<SymbolicExpr> exprs;
  exprs.reserve(input_dims.size());

  for (auto [input_dim_id, input_dim] : llvm::enumerate(input_dims)) {
    if (reduce_dims_set.contains(input_dim_id)) {
      exprs.push_back(CreateSymbolExpr(parallel_dims_sizes.size(),
                                       output_dims.size(), mlir_context));
      parallel_dims_sizes.push_back(input_dim);
      continue;
    }
    exprs.push_back(CreateDimExpr(output_dim_id++, mlir_context));
  }

  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, output_dims.size(), reduce_dims_set.size(),
                       std::move(exprs)),
      output_dims, parallel_dims_sizes);
}

IndexingMap ComputePadIndexingMap(absl::Span<const int64_t> output_dims,
                                  absl::Span<const int64_t> padding_low,
                                  absl::Span<const int64_t> padding_high,
                                  absl::Span<const int64_t> padding_interior,
                                  MLIRContext* mlir_context) {
  int64_t output_rank = output_dims.size();

  SmallVector<SymbolicExpr> exprs;
  std::vector<std::pair<SymbolicExpr, Interval>> constraints;
  std::vector<IndexingMap::Variable> dim_vars;
  exprs.reserve(output_rank);
  constraints.reserve(output_rank);
  int64_t output_dim_id = 0;
  for (const auto [output_dim, pad_low, pad_high, pad_interior] :
       llvm::zip(output_dims, padding_low, padding_high, padding_interior)) {
    SymbolicExpr dim_expr = CreateDimExpr(output_dim_id, mlir_context);
    dim_vars.push_back({IndexingMap::Variable{
        std::max(int64_t{0}, pad_low),
        std::min(output_dim - 1, output_dim - 1 - pad_high)}});
    if (pad_interior == 0) {
      exprs.push_back(dim_expr - pad_low);
    } else {
      exprs.push_back((dim_expr - pad_low).floorDiv(pad_interior + 1));
      constraints.push_back(
          {(dim_expr - pad_low) % (pad_interior + 1), Interval{0, 0}});
    }
    ++output_dim_id;
  }
  return IndexingMap(SymbolicMap::Get(mlir_context, output_rank,
                                      /*num_symbols=*/0, std::move(exprs)),
                     std::move(dim_vars),
                     /*range_vars = */ {},
                     /*rt_vars = */ {}, absl::MakeSpan(constraints));
}

IndexingMap ComposeWindowIndexingMap(absl::Span<const int64_t> input_dims,
                                     absl::Span<const int64_t> output_dims,
                                     absl::Span<const int64_t> window_dims,
                                     absl::Span<const int64_t> window_strides,
                                     absl::Span<const int64_t> window_dilations,
                                     absl::Span<const int64_t> base_dilations,
                                     absl::Span<const int64_t> padding,
                                     MLIRContext* mlir_context) {
  size_t rank = input_dims.size();

  // Compute shape of the padded input and the indexing map of pad op required
  // to pad the input.
  SmallVector<int64_t> padding_low, padding_high, padding_interior,
      padded_input_dimensions;
  SmallVector<SymbolicExpr, 4> exprs;
  std::vector<IndexingMap::Variable> dim_vars;
  std::vector<IndexingMap::Variable> range_vars;
  exprs.reserve(rank);
  dim_vars.reserve(rank);
  range_vars.reserve(rank);

  for (size_t dim_id = 0; dim_id < rank; ++dim_id) {
    int64_t pad_low = padding[dim_id * 2];
    int64_t pad_high = padding[dim_id * 2 + 1];
    int64_t base_dilation = base_dilations[dim_id];
    int64_t window_dilation = window_dilations[dim_id];
    int64_t window_stride = window_strides[dim_id];
    int64_t output_dim = output_dims[dim_id];
    int64_t window_dim = window_dims[dim_id];
    int64_t input_dim_size = input_dims[dim_id];

    padding_low.push_back(pad_low);
    padding_high.push_back(pad_high);
    // For some reason interior_padding in HLO pad is offset from base_dilations
    // in HLO reduce-window by 1.
    padding_interior.push_back(base_dilation - 1);
    padded_input_dimensions.push_back(input_dim_size + pad_low + pad_high +
                                      (input_dim_size - 1) *
                                          (base_dilation - 1));
    SymbolicExpr dim_expr = CreateDimExpr(dim_id, mlir_context);
    SymbolicExpr symbol_expr = CreateSymbolExpr(dim_id, rank, mlir_context);

    exprs.push_back(symbol_expr * window_dilation + window_stride * dim_expr);
    dim_vars.push_back({IndexingMap::Variable{0, output_dim - 1}});
    range_vars.push_back({IndexingMap::Variable{0, window_dim - 1}});
  }
  // Indexing map for pad op that pads the input.
  IndexingMap padded_input_indexing =
      ComputePadIndexingMap(padded_input_dimensions, padding_low, padding_high,
                            padding_interior, mlir_context);
  // Indexing map for reduce-window, that does not do any padding.
  IndexingMap input_indexing_no_padding(
      SymbolicMap::Get(mlir_context, rank, rank, std::move(exprs)), dim_vars,
      range_vars,
      /*rt_vars=*/{});

  // Composed indexing.
  IndexingMap result =
      ComposeIndexingMaps(input_indexing_no_padding, padded_input_indexing);
  result.Simplify();
  result.RemoveUnusedSymbols();
  return result;
}

HloInstructionIndexing CreateElementwiseIndexing(int64_t num_operands,
                                                 const Shape& output_shape,
                                                 MLIRContext* mlir_context) {
  IndexingMap identity_map = IndexingMap::FromTensorSizes(
      SymbolicMap::GetMultiDimIdentityMap(output_shape.dimensions().size(),
                                          mlir_context),
      output_shape.dimensions(), {});
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(num_operands);
  for (int64_t i = 0; i < num_operands; ++i) {
    indexing.indexing_maps[i].insert(OperandIndexing{identity_map});
  }
  return indexing;
}

IndexingMap CreateScalarIndexingMap(const Shape& output_shape,
                                    MLIRContext* mlir_context) {
  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, output_shape.dimensions().size(),
                       /*num_symbols=*/0, {}),
      output_shape.dimensions(), /*symbol_upper_bounds=*/{});
}

SymbolicMap ComputeTransposeIndexingMap(absl::Span<const int64_t> permutation,
                                        MLIRContext* mlir_context) {
  SmallVector<SymbolicExpr> exprs;
  exprs.reserve(permutation.size());
  for (int64_t dim : permutation) {
    exprs.push_back(CreateDimExpr(dim, mlir_context));
  }
  return SymbolicMap::Get(mlir_context, permutation.size(), 0,
                          std::move(exprs));
}

}  // namespace xla
