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

#include "xla/service/gpu/model/indexing_analysis.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::getAffineSymbolExpr;
using mlir::MLIRContext;

HloInstructionIndexing CreateUnknownIndexing(int64_t count = 1) {
  HloInstructionIndexing indexing;
  indexing.indexing_maps = std::vector<absl::flat_hash_set<IndexingMap>>(
      count, {IndexingMap::GetUndefined()});
  return indexing;
}

IndexingMap CreateIdentityMap(const Shape& shape, MLIRContext* ctx) {
  if (shape.IsTuple()) {
    // Should happen only for variadic reduce. In that case all tuple shapes are
    // equal.
    return CreateIdentityMap(shape.tuple_shapes(0), ctx);
  }

  auto dims = shape.dimensions();
  IndexingMap identity_map = IndexingMap::FromTensorSizes(
      AffineMap::getMultiDimIdentityMap(dims.size(), ctx), dims, {});
  return identity_map;
}

HloInstructionIndexing ComputeOutputToInputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(instr->operand_count());
  int64_t operand_count = instr->operand_count();
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    instr_indexing.indexing_maps[operand_id].insert(identity_map);
  }
  return instr_indexing;
}

HloInstructionIndexing ComputeInputToOutputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({identity_map});
}

HloInstructionIndexing ComputeOutputToInputBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  auto output_dims = bcast->shape().dimensions();

  std::vector<AffineExpr> exprs;
  exprs.reserve(bcast->dimensions().size());
  for (int64_t bcast_dim : bcast->dimensions()) {
    exprs.push_back(getAffineDimExpr(bcast_dim, mlir_context));
  }
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                     mlir_context),
      output_dims, {});
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeInputToOutputBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  absl::Span<const int64_t> bcast_dims = bcast->dimensions();

  const Shape& input_shape = bcast->operand(0)->shape();
  const Shape& output_shape = bcast->shape();

  std::vector<int64_t> added_dims_sizes;
  std::vector<AffineExpr> exprs;
  exprs.reserve(output_shape.rank());
  for (auto [output_dim_id, output_dim] :
       llvm::enumerate(output_shape.dimensions())) {
    auto bcast_dim =
        std::find(bcast_dims.begin(), bcast_dims.end(), output_dim_id);
    if (bcast_dim == bcast_dims.end()) {
      exprs.push_back(
          getAffineSymbolExpr(added_dims_sizes.size(), mlir_context));
      added_dims_sizes.push_back(output_dim);
      continue;
    }
    exprs.push_back(getAffineDimExpr(
        std::distance(bcast_dims.begin(), bcast_dim), mlir_context));
  }
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(input_shape.rank(), added_dims_sizes.size(), exprs,
                     mlir_context),
      input_shape.dimensions(), added_dims_sizes);

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

std::vector<Range> RangesFromUpperBounds(absl::Span<const int64_t> bounds) {
  std::vector<Range> dim_ranges;
  dim_ranges.reserve(bounds.size());
  for (int64_t dim : bounds) {
    dim_ranges.push_back(Range{0, dim - 1});
  }
  return dim_ranges;
}

HloInstructionIndexing ComputeOutputToInputConcatenateOpIndexing(
    const HloConcatenateInstruction* concat, MLIRContext* mlir_context) {
  const auto& operand_0_dims = concat->operand(0)->shape().dimensions();

  // Initialize affine map and domain. Only concat_dim elements of both have to
  // be adjusted for a particular operand_id.
  mlir::MutableAffineMap affine_map =
      AffineMap::getMultiDimIdentityMap(operand_0_dims.size(), mlir_context);
  std::vector<Range> dim_ranges = RangesFromUpperBounds(operand_0_dims);

  HloInstructionIndexing concat_indexing;
  concat_indexing.indexing_maps.resize(concat->operand_count());
  int64_t concat_dim = concat->concatenate_dimension();
  AffineExpr concat_dim_expr = getAffineDimExpr(concat_dim, mlir_context);
  int64_t offset = 0;
  for (const auto [operand_id, operand] : llvm::enumerate(concat->operands())) {
    affine_map.setResult(concat_dim, concat_dim_expr - offset);
    int64_t operand_concat_dim = operand->shape().dimensions()[concat_dim];
    dim_ranges[concat_dim] = Range{offset, offset + operand_concat_dim - 1};
    concat_indexing.indexing_maps[operand_id].insert(
        IndexingMap(affine_map.getAffineMap(), dim_ranges,
                    /*symbol_ranges=*/{}));
    offset += operand_concat_dim;
  }
  return concat_indexing;
}

HloInstructionIndexing ComputeInputToOutputConcatenateOpIndexing(
    const HloConcatenateInstruction* concat, int input_id,
    MLIRContext* mlir_context) {
  int64_t concat_dim = concat->concatenate_dimension();
  int64_t offset = 0;
  for (int64_t operand_id = 0; operand_id < input_id; ++operand_id) {
    offset += concat->operand(operand_id)->shape().dimensions()[concat_dim];
  }
  // Initialize affine map. Only concat_dim element has to be adjusted for a
  // particular operand_id.
  const auto& operand_dims = concat->operand(input_id)->shape().dimensions();
  mlir::MutableAffineMap affine_map =
      AffineMap::getMultiDimIdentityMap(operand_dims.size(), mlir_context);
  affine_map.setResult(concat_dim,
                       getAffineDimExpr(concat_dim, mlir_context) + offset);
  IndexingMap indexing_map =
      IndexingMap::FromTensorSizes(affine_map.getAffineMap(), operand_dims, {});
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

// Composes instruction indexing maps starting at the root instruction
// until the HloParameterInstruction is found.
HloInstructionIndexing ComputeOutputToInputFusionOpIndexing(
    const HloFusionInstruction* fusion, int output_id,
    MLIRContext* mlir_context) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);
  auto grouped_indexing_maps = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[output_id], mlir_context);

  // After the traversal, `grouped_indexing_maps` is keyed by
  // HloParameterInstructions. Convert them back to the operand id and return.
  HloInstructionIndexing fusion_indexing;
  fusion_indexing.indexing_maps.resize(fusion->operand_count());
  for (auto [operand_id, operand] : llvm::enumerate(fusion->operands())) {
    fusion_indexing.indexing_maps[operand_id] = grouped_indexing_maps[operand];
  }
  return fusion_indexing;
}

HloInstructionIndexing ComputeOutputToInputDotOpIndexing(
    const HloDotInstruction* dot, MLIRContext* mlir_context) {
  CHECK_NE(dot, nullptr);
  const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();
  absl::Span<const int64_t> lhs_contracting_dims(
      dim_numbers.lhs_contracting_dimensions());
  absl::Span<const int64_t> rhs_contracting_dims =
      dim_numbers.rhs_contracting_dimensions();

  absl::Span<const int64_t> lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dims = dim_numbers.rhs_batch_dimensions();

  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  // According to the StableHLO specification, the dimensions of the output
  // shape are ordered as follows:
  //   lhs_batch_dims | lhs_non_contracting_dims | rhs_non_contracting_dims
  SmallVector<AffineExpr> lhs_exprs(lhs_shape.rank());
  SmallVector<AffineExpr> rhs_exprs(rhs_shape.rank());
  int64_t output_dim_id = 0;

  // lhs_batch_dims
  for (auto [lhs_batch_dim, rhs_batch_dim] :
       llvm::zip(lhs_batch_dims, rhs_batch_dims)) {
    AffineExpr output_dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
    lhs_exprs[lhs_batch_dim] = output_dim_expr;
    rhs_exprs[rhs_batch_dim] = output_dim_expr;
    ++output_dim_id;
  }

  // lhs_non_contracting_dims
  auto lhs_non_contracting_dims =
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_contracting_dims);
  assert(lhs_non_contracting_dims.ok());

  for (int64_t lhs_non_contracting_dim : lhs_non_contracting_dims.value()) {
    lhs_exprs[lhs_non_contracting_dim] =
        getAffineDimExpr(output_dim_id++, mlir_context);
  }

  // rhs_non_contracting_dims
  auto rhs_non_contracting_dims =
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_contracting_dims);
  assert(rhs_non_contracting_dims.ok());
  for (int64_t rhs_non_contracting_dim : rhs_non_contracting_dims.value()) {
    rhs_exprs[rhs_non_contracting_dim] =
        getAffineDimExpr(output_dim_id++, mlir_context);
  }

  int64_t input_dim_id = 0;
  std::vector<int64_t> input_dim_sizes;
  input_dim_sizes.reserve(lhs_contracting_dims.size());

  for (auto [lhs_contracting_dim, rhs_contracting_dim] :
       llvm::zip(lhs_contracting_dims, rhs_contracting_dims)) {
    AffineExpr input_dim_expr = getAffineSymbolExpr(input_dim_id, mlir_context);
    lhs_exprs[lhs_contracting_dim] = input_dim_expr;
    rhs_exprs[rhs_contracting_dim] = input_dim_expr;
    ++input_dim_id;

    // LHS and RHS contracting dimensions must match pairwise, and we therefore
    // need only populate a single input_dim_sizes vector.
    input_dim_sizes.push_back(lhs_shape.dimensions(lhs_contracting_dim));
  }

  IndexingMap lhs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(dot->shape().rank(), input_dim_sizes.size(), lhs_exprs,
                     mlir_context),
      dot->shape().dimensions(), input_dim_sizes);

  IndexingMap rhs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(dot->shape().rank(), input_dim_sizes.size(), rhs_exprs,
                     mlir_context),
      dot->shape().dimensions(), input_dim_sizes);
  return HloInstructionIndexing::FromIndexingMaps(
      {lhs_indexing_map, rhs_indexing_map});
}

IndexingMap ComputeOutputToInputPadOpIndexingImpl(
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> padding_low,
    absl::Span<const int64_t> padding_high,
    absl::Span<const int64_t> padding_interior, MLIRContext* mlir_context) {
  int64_t output_rank = output_dims.size();

  std::vector<AffineExpr> exprs;
  std::vector<std::pair<AffineExpr, Range>> constraints;
  std::vector<Range> dimension_ranges;
  exprs.reserve(output_rank);
  constraints.reserve(output_rank);
  int64_t output_dim_id = 0;
  for (const auto [output_dim, pad_low, pad_high, pad_interior] :
       llvm::zip(output_dims, padding_low, padding_high, padding_interior)) {
    AffineExpr dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
    dimension_ranges.push_back(
        Range{std::max(int64_t{0}, pad_low),
              std::min(output_dim - 1, output_dim - 1 - pad_high)});
    if (pad_interior == 0) {
      exprs.push_back(dim_expr - pad_low);
    } else {
      exprs.push_back((dim_expr - pad_low).floorDiv(pad_interior + 1));
      constraints.push_back(
          {(dim_expr - pad_low) % (pad_interior + 1), Range{0, 0}});
    }
    ++output_dim_id;
  }
  return IndexingMap{
      AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      dimension_ranges, /*symbol_ranges = */ {}, absl::MakeSpan(constraints)};
}

HloInstructionIndexing ComputeOutputToInputPadOpIndexing(
    const HloPadInstruction* pad, MLIRContext* mlir_context) {
  const Shape& output_shape = pad->shape();
  int64_t rank = output_shape.rank();
  SmallVector<int64_t> padding_low, padding_high, padding_interior;
  padding_low.reserve(rank);
  padding_high.reserve(rank);
  padding_interior.reserve(rank);
  for (const auto& dim_config : pad->padding_config().dimensions()) {
    padding_low.push_back(dim_config.edge_padding_low());
    padding_high.push_back(dim_config.edge_padding_high());
    padding_interior.push_back(dim_config.interior_padding());
  }
  IndexingMap input_indexing_map = ComputeOutputToInputPadOpIndexingImpl(
      output_shape.dimensions(), padding_low, padding_high, padding_interior,
      mlir_context);
  IndexingMap padding_value_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.rank(), /*symbolCount=*/0, {}, mlir_context),
      output_shape.dimensions(), /*symbol_upper_bounds=*/{});
  return HloInstructionIndexing::FromIndexingMaps(
      {input_indexing_map, padding_value_indexing_map});
}

HloInstructionIndexing ComputeOutputToInputReduceOpIndexing(
    const HloReduceInstruction* reduce, int output_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());

  const Shape& input_shape = reduce->operand(output_id)->shape();
  const Shape& output_shape = GetOutputShape(reduce, 0);

  std::vector<int64_t> parallel_dims_sizes;
  int64_t output_dim_id = 0;
  std::vector<AffineExpr> exprs;
  exprs.reserve(input_shape.rank());
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (reduce_dims_ids.contains(input_dim_id)) {
      exprs.push_back(
          getAffineSymbolExpr(parallel_dims_sizes.size(), mlir_context));
      parallel_dims_sizes.push_back(input_dim);
      continue;
    }
    exprs.push_back(getAffineDimExpr(output_dim_id++, mlir_context));
  }
  IndexingMap inputs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.rank(), reduce_dims_ids.size(), exprs,
                     mlir_context),
      output_shape.dimensions(), parallel_dims_sizes);
  IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.rank(), /*symbolCount=*/0, {}, mlir_context),
      output_shape.dimensions(), {});

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(reduce->operand_count());
  for (int64_t id = 0; id < reduce->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing_map);
  }
  for (int64_t id = reduce->input_count(); id < reduce->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inits_indexing_map);
  }
  return instr_indexing;
}

HloInstructionIndexing ComputeInputToOutputReduceOpIndexing(
    const HloReduceInstruction* reduce, int input_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());
  const Shape& input_shape = reduce->operand(input_id)->shape();
  const Shape& output_shape = GetOutputShape(reduce, 0);
  int64_t output_rank = output_shape.rank();

  int64_t output_dim_id = 0;
  std::vector<AffineExpr> inputs_exprs, inits_exprs;
  inputs_exprs.reserve(output_rank);
  inits_exprs.reserve(output_rank);
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (reduce_dims_ids.contains(input_dim_id)) {
      continue;
    }
    inputs_exprs.push_back(getAffineDimExpr(input_dim_id, mlir_context));
    inits_exprs.push_back(getAffineSymbolExpr(output_dim_id++, mlir_context));
  }
  IndexingMap inputs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(input_shape.rank(), /*symbolCount=*/0, inputs_exprs,
                     mlir_context),
      input_shape.dimensions(), {});
  IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(0, /*symbolCount=*/output_rank, inits_exprs, mlir_context),
      {}, output_shape.dimensions());

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(reduce->operand_count());
  for (int64_t id = 0; id < reduce->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing_map);
  }
  for (int64_t id = reduce->input_count(); id < reduce->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inits_indexing_map);
  }
  return instr_indexing;
}

// Indexing for reduce-window with dilations and non-trivial padding can be
// represented as a composition of pad op and reduce-window that never goes out
// of bounds.
HloInstructionIndexing ComputeOutputToInputReduceWindowOpIndexing(
    const HloReduceWindowInstruction* reduce_window, int output_id,
    MLIRContext* mlir_context) {
  const Shape& input_shape = reduce_window->operand(0)->shape();
  const Shape& output_shape = GetOutputShape(reduce_window, 0);
  int64_t rank = input_shape.rank();

  // Compute shape of the padded input and the indexing map of pad op required
  // to pad the input.
  SmallVector<int64_t> padding_low, padding_high, padding_interior,
      padded_input_dimensions;
  padding_low.reserve(rank);
  padding_high.reserve(rank);
  padding_interior.reserve(rank);
  padded_input_dimensions.reserve(rank);
  SmallVector<AffineExpr, 4> exprs;
  std::vector<Range> dim_ranges, symbol_ranges;
  exprs.reserve(rank);
  dim_ranges.reserve(rank);
  symbol_ranges.reserve(rank);
  for (const auto& [dim_id, window_config] :
       llvm::enumerate(reduce_window->window().dimensions())) {
    padding_low.push_back(window_config.padding_low());
    padding_high.push_back(window_config.padding_high());
    // For some reason interior_padding in HLO pad is offset from base_dilations
    // in HLO reduce-window by 1.
    padding_interior.push_back(window_config.base_dilation() - 1);
    padded_input_dimensions.push_back(input_shape.dimensions(dim_id) +
                                      window_config.padding_low() +
                                      window_config.padding_high() +
                                      (input_shape.dimensions(dim_id) - 1) *
                                          (window_config.base_dilation() - 1));
    AffineExpr dim_expr = getAffineDimExpr(dim_id, mlir_context);
    AffineExpr symbol_expr = getAffineSymbolExpr(dim_id, mlir_context);

    exprs.push_back(symbol_expr + window_config.stride() * dim_expr);
    dim_ranges.push_back(Range{0, output_shape.dimensions(dim_id) - 1});
    symbol_ranges.push_back(Range{0, window_config.size() - 1});
  }
  // Indexing map for pad op that pads the input.
  IndexingMap padded_input_indexing = ComputeOutputToInputPadOpIndexingImpl(
      padded_input_dimensions, padding_low, padding_high, padding_interior,
      mlir_context);
  // Indexing map for reduce-window, that does not do any padding.
  IndexingMap reduce_window_indexing_no_padding(
      AffineMap::get(rank, rank, exprs, mlir_context), dim_ranges,
      symbol_ranges);

  // Composed indexing.
  IndexingMap inputs_indexing = ComposeIndexingMaps(
      reduce_window_indexing_no_padding, padded_input_indexing);
  inputs_indexing.Simplify();
  inputs_indexing.RemoveUnusedSymbols();

  // Indexing map for the init value.
  IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.rank(), /*symbolCount=*/0, {}, mlir_context),
      output_shape.dimensions(), /*symbol_upper_bounds=*/{});

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(reduce_window->operand_count());
  for (int64_t id = 0; id < reduce_window->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing);
  }
  for (int64_t id = reduce_window->input_count();
       id < reduce_window->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inits_indexing_map);
  }
  return instr_indexing;
}

// Computes strides for a shape.
std::vector<int64_t> ComputeStrides(absl::Span<const int64_t> dims) {
  int rank = static_cast<int>(dims.size());
  std::vector<int64_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = dims[i + 1] * strides[i + 1];
  }
  return strides;
}

// Computes 1D index given a shape and N-d indexing expressions.
AffineExpr LinearizeShape(absl::Span<const int64_t> dims,
                          absl::Span<const AffineExpr> dimension_exprs,
                          MLIRContext* mlir_context) {
  AffineExpr linear_index = getAffineConstantExpr(0, mlir_context);

  auto strides = ComputeStrides(dims);
  for (auto [stride, dimension_expr] : llvm::zip(strides, dimension_exprs)) {
    linear_index = linear_index + dimension_expr * stride;
  }
  return linear_index;
}

// Computes N-d indexing expressions given a linear index and a shape.
std::vector<AffineExpr> DelinearizeIndex(absl::Span<const int64_t> dims,
                                         AffineExpr linear_index,
                                         MLIRContext* mlir_context) {
  std::vector<AffineExpr> multi_index;
  multi_index.reserve(dims.size());

  AffineExpr remainder = linear_index;
  for (int64_t stride : ComputeStrides(dims)) {
    multi_index.push_back(remainder.floorDiv(stride));
    remainder = remainder % stride;
  }
  return multi_index;
}

// Computes indexing for "minimal" reshapes, i.e. reshapes that cannot be
// represented by a series of composed reshapes, i.e. when there are no
// subshapes in input and output that have the same number of elements.
// For example, [8, 4] -> [8, 2, 2] is not a minimal reshape, it has matching
// subshapes [8] -> [8] and [4] -> [2, 2].
//
// There are only 4 types of "minimal" reshapes considers only 4 cases:
//   1. Dimension is not changed, e.g. [8] -> [8]
//   2. Dimension is expanded, e.g. [8] -> [4, 2]
//   3. Dimension is collapsed, e.g. [4, 2] -> [8]
//   4. Dimension is collapsed and expanded, e.g. [8, 16] -> [4, 32]
//
// The function computes indexing maps for these 4 cases, i.e. considers given
// input/output shapes and checks if the shapes are the same, expanded or
// collapsed. Otherwise, performs linearization/delinearization.
void ComputeMinimalReshapeIndexing(
    absl::Span<const int64_t> input_dims, absl::Span<const int64_t> output_dims,
    absl::Span<const AffineExpr> output_dims_exprs,
    std::vector<AffineExpr>* exprs, MLIRContext* mlir_context) {
  // The shape does not change.
  if (input_dims.size() == 1 && output_dims.size() == 1) {
    absl::c_copy(output_dims_exprs, std::back_inserter(*exprs));
    return;
  }
  // Expand shape.
  if (input_dims.size() == 1) {
    exprs->push_back(
        LinearizeShape(output_dims, output_dims_exprs, mlir_context));
    return;
  }
  // Collapse shape.
  if (output_dims.size() == 1) {
    auto multi_index =
        DelinearizeIndex(input_dims, output_dims_exprs.front(), mlir_context);
    absl::c_copy(multi_index, std::back_inserter(*exprs));
    return;
  }
  // Generic case.
  AffineExpr linear_index =
      LinearizeShape(output_dims, output_dims_exprs, mlir_context);
  auto multi_index = DelinearizeIndex(input_dims, linear_index, mlir_context);
  absl::c_copy(multi_index, std::back_inserter(*exprs));
}

// Scans input and output shapes from left to right in an attempt to find
// subshapes with the same number of elements and then computes indexing map for
// every pair of subshapes.
//
// Example:
//   p0 = f32[4, 8, 12] parameter(0)
//   reshape = f32[32, 3, 4] reshape(p0)
//
// This reshape can be represented as a composition of two reshapes.
// The first reshape collapses dimensions first two input dimensions [4, 8] onto
// the output dimension [32].
// The second reshape expands the input dimension [12] into two output
// dimensions [3, 4].
// This is an optimization that allows us to construct simpler affine maps,
// otherwise we would need to linearize/delinearize even some of the simpler
// cases.
AffineMap ComputeReshapeIndexingMap(const Shape& input, const Shape& output,
                                    MLIRContext* mlir_context) {
  absl::Span<const int64_t> input_dims = input.dimensions();
  absl::Span<const int64_t> output_dims = output.dimensions();

  std::vector<AffineExpr> exprs;
  exprs.reserve(input.rank());

  // If the input shape has no elements (e.g. 1000x10x0 -> 100x100x0), just set
  // everything to 0.
  if (ShapeUtil::ElementsIn(input) == 0) {
    for (int i = 0; i < input.rank(); ++i) {
      exprs.push_back(getAffineConstantExpr(0, mlir_context));
    }
    return AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                          mlir_context);
  }

  std::vector<AffineExpr> output_dims_exprs;

  // Find subshapes with the same element count and compute indexing for them.
  int64_t input_num_elements = 1;
  int64_t output_num_elements = 1;
  std::vector<int64_t> input_subshape, output_subshape;
  size_t input_dim_id = 0, output_dim_id = 0;
  while (input_dim_id < input.rank() || output_dim_id < output.rank() ||
         !input_subshape.empty()) {
    if (input_dim_id < input.rank() &&
        (input_subshape.empty() || input_num_elements < output_num_elements ||
         input_dims[input_dim_id] == 1)) {
      input_num_elements *= input_dims[input_dim_id];
      input_subshape.push_back(input_dims[input_dim_id]);
      ++input_dim_id;
      continue;
    }
    if (output_dim_id < output.rank() &&
        (output_subshape.empty() || output_num_elements < input_num_elements ||
         output_dims[output_dim_id] == 1)) {
      output_num_elements *= output_dims[output_dim_id];
      output_subshape.push_back(output_dims[output_dim_id]);
      output_dims_exprs.push_back(
          getAffineDimExpr(output_dim_id, mlir_context));
      ++output_dim_id;
      continue;
    }
    ComputeMinimalReshapeIndexing(input_subshape, output_subshape,
                                  output_dims_exprs, &exprs, mlir_context);
    input_num_elements = 1;
    output_num_elements = 1;
    input_subshape.clear();
    output_subshape.clear();
    output_dims_exprs.clear();
  }
  return AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                        mlir_context);
};

HloInstructionIndexing ComputeOutputToInputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  const auto& input = reshape->operand(0)->shape();
  const auto& output = reshape->shape();

  IndexingMap reshape_indexing_map = IndexingMap::FromTensorSizes(
      ComputeReshapeIndexingMap(input, output, mlir_context),
      output.dimensions(), {});
  reshape_indexing_map.Simplify();
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}
HloInstructionIndexing ComputeInputToOutputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  const auto& input = reshape->operand(0)->shape();
  const auto& output = reshape->shape();

  IndexingMap reshape_indexing_map = IndexingMap::FromTensorSizes(
      ComputeReshapeIndexingMap(output, input, mlir_context),
      input.dimensions(), {});
  reshape_indexing_map.Simplify();
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}

HloInstructionIndexing ComputeReverseOpIndexing(
    const HloReverseInstruction* reverse, MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reverse_dims(reverse->dimensions().begin(),
                                            reverse->dimensions().end());
  auto output_dims = reverse->shape().dimensions();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_dims.size());
  for (auto [output_dim_id, output_dim] : llvm::enumerate(output_dims)) {
    auto dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
    if (!reverse_dims.contains(output_dim_id)) {
      exprs.push_back(dim_expr);
      continue;
    }
    exprs.push_back(-dim_expr + output_dim - 1);
  }

  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                     mlir_context),
      output_dims, {});

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeOutputToInputSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  auto output_rank = slice->shape().rank();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);
  for (int64_t dim = 0; dim < output_rank; ++dim) {
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);
    exprs.push_back(dim_expr * slice->slice_strides()[dim] +
                    slice->slice_starts()[dim]);
  }
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      slice->shape().dimensions(), {});
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

AffineMap ComputeTransposeIndexingMap(absl::Span<const int64_t> permutation,
                                      MLIRContext* mlir_context) {
  return AffineMap::getPermutationMap(
      std::vector<unsigned>(permutation.begin(), permutation.end()),
      mlir_context);
}

HloInstructionIndexing ComputeOutputToInputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  AffineMap inverse_permutation = ComputeTransposeIndexingMap(
      InversePermutation(transpose->dimensions()), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({IndexingMap::FromTensorSizes(
      inverse_permutation, transpose->shape().dimensions(), {})});
}

HloInstructionIndexing ComputeInputToOutputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  AffineMap forward_permutation =
      ComputeTransposeIndexingMap(transpose->dimensions(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({IndexingMap::FromTensorSizes(
      forward_permutation, transpose->operand(0)->shape().dimensions(), {})});
}

}  // namespace

IndexingMap GetBitcastMap(const Shape& input_shape, const Shape& output_shape,
                          MLIRContext* ctx) {
  ShapeUtil::BitcastDecomposition decomposed_bitcast =
      ShapeUtil::DecomposeBitcast(input_shape, output_shape);

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          decomposed_bitcast)) {
    auto permutation = ShapeUtil::DeduceTransposeDimensionsForBitcast(
        input_shape, output_shape);
    CHECK(permutation.has_value())
        << "Failed to deduce permutation for a bitcast.";

    return IndexingMap::FromTensorSizes(
        ComputeTransposeIndexingMap(permutation.value(), ctx),
        input_shape.dimensions(), {});
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposed_bitcast)) {
    // Note: ComputeReshapeIndexingMap assumes it's computing an output->input
    // indexing, so input and output are reversed.
    return IndexingMap::FromTensorSizes(
        ComputeReshapeIndexingMap(output_shape, input_shape, ctx),
        input_shape.dimensions(), {});
  }
  // `trt` stands for transpose-reshape-transpose decomposition of bitcast.
  auto trt = std::get<ShapeUtil::BitcastDecompositionTrt>(decomposed_bitcast);
  auto transpose_map_1 = ComputeTransposeIndexingMap(trt.transpose1_dims, ctx);
  auto reshape_map =
      ComputeReshapeIndexingMap(trt.reshape_shape, trt.transpose1_shape, ctx);
  auto transpose_map_2 = ComputeTransposeIndexingMap(trt.transpose2_dims, ctx);
  auto bitcast_map =
      transpose_map_2.compose(reshape_map).compose(transpose_map_1);
  return IndexingMap::FromTensorSizes(bitcast_map, input_shape.dimensions(),
                                      {});
}

namespace {

HloInstructionIndexing ComputeOutputToInputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  auto bitcast_map = GetBitcastMap(bitcast->shape(),
                                   bitcast->operand(0)->shape(), mlir_context);
  bitcast_map.Simplify();

  return HloInstructionIndexing::FromIndexingMaps({bitcast_map});
}

HloInstructionIndexing ComputeInputToOutputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  auto bitcast_map = GetBitcastMap(bitcast->operand(0)->shape(),
                                   bitcast->shape(), mlir_context);
  bitcast_map.Simplify();

  return HloInstructionIndexing::FromIndexingMaps({bitcast_map});
}

// Converts a layout to a dimensions transposition necessary to get to that
// layout from identity.
std::vector<int64_t> ToTransposeDimensions(const Layout& l) {
  std::vector<int64_t> out(l.minor_to_major().begin(),
                           l.minor_to_major().end());
  absl::c_reverse(out);
  return out;
}

AffineMap GetTilingAffineMap(llvm::ArrayRef<AffineExpr> exprs,
                             const Tiling& tiling) {
  return AffineMap::get(
      /*dimCount=*/6, /*symbolCount=*/tiling.GetShape().size(), exprs,
      exprs[0].getContext());
}

}  // namespace

llvm::SmallVector<AffineExpr, 4> DelinearizeInBoundsIndex(
    AffineExpr linear, absl::Span<const int64_t> sizes,
    absl::Span<const int64_t> strides) {
  llvm::SmallVector<AffineExpr, 4> result;
  result.reserve(sizes.size());
  if (absl::c_linear_search(sizes, 0)) {
    for (int dim = 0; dim < sizes.size(); ++dim) {
      result.push_back(mlir::getAffineConstantExpr(0, linear.getContext()));
    }
    return result;
  }

  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    result.push_back(linear.floorDiv(stride) % size);
  }
  for (int dim = 0; dim < sizes.size(); ++dim) {
    if (sizes[dim] > 1) {
      // We assume the linear index is in bounds, so no mod for the first major
      // non-degenerate dimension. Degenerate dimensions are already rewritten
      // to 0 by operator%.
      result[dim] = linear.floorDiv(strides[dim]);
      break;
    }
  }
  return result;
}

IndexingMap GetIndexingMapFromPhysicalLayoutToLogical(const Shape& shape,
                                                      MLIRContext* ctx) {
  if (shape.rank() == 0) {
    return IndexingMap(AffineMap::get(ctx), {}, {});
  }
  return IndexingMap::FromTensorSizes(
      ComputeTransposeIndexingMap(
          InversePermutation(ToTransposeDimensions(shape.layout())), ctx),
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape)
          .dimensions(),
      {});
}

IndexingMap GetIndexingMapFromLogicalToPhysicalLayout(const Shape& shape,
                                                      MLIRContext* ctx) {
  if (shape.rank() == 0) {
    return IndexingMap(AffineMap::get(ctx), {}, {});
  }
  return IndexingMap::FromTensorSizes(
      ComputeTransposeIndexingMap(ToTransposeDimensions(shape.layout()), ctx),
      shape.dimensions(), {});
}

AffineMap GetBlockOffsetsForTiling(const Tiling& tiling,
                                   mlir::MLIRContext* ctx) {
  auto offsets = DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx),
                                          tiling.GetBlockCounts(),
                                          tiling.GetBlockStrides());
  for (auto&& [offset, tile_size] :
       llvm::zip(offsets, tiling.GetBlockTileSize())) {
    offset = offset * tile_size;
  }
  return GetTilingAffineMap(offsets, tiling);
}

AffineMap GetThreadOffsetsForTiling(const Tiling& tiling,
                                    mlir::MLIRContext* ctx) {
  auto offsets = DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx),
                                          tiling.GetThreadsPerBlock(),
                                          tiling.GetThreadStrides());
  for (int dim = 0; dim < tiling.GetShape().size(); ++dim) {
    if (tiling.GetThreadTileSize()[dim] > 1) {
      offsets[dim] = offsets[dim] + getAffineSymbolExpr(dim, ctx) *
                                        tiling.GetThreadsPerBlock()[dim];
    }
  }
  return GetTilingAffineMap(offsets, tiling);
}

IndexingMap GetIndexingMapForTiling(const Tiling& tiling,
                                    mlir::MLIRContext* ctx) {
  return GetIndexingMapForTiling(GetBlockOffsetsForTiling(tiling, ctx),
                                 GetThreadOffsetsForTiling(tiling, ctx),
                                 tiling);
}

IndexingMap GetIndexingMapForTiling(AffineMap block_offsets,
                                    AffineMap thread_offsets,
                                    const Tiling& tiling) {
  llvm::SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(block_offsets.getNumResults());
  for (auto [block, thread] :
       llvm::zip(block_offsets.getResults(), thread_offsets.getResults())) {
    offsets.push_back(block + thread);
  }

  // TODO(jreiffers): Use general constraints for symbols: in the last blocks
  // in each each dimension, the bounds can be different if we don't have a
  // perfect tiling.
  std::vector<Range> dimension_ranges{
      {0, tiling.GetNumThreadsPerBlock() - 1}, {}, {},
      {0, tiling.GetNumBlocks() - 1},          {}, {},
  };
  return {GetTilingAffineMap(offsets, tiling), dimension_ranges,
          RangesFromUpperBounds(tiling.GetThreadTileSize())};
}

bool HloInstructionIndexing::Simplify() {
  bool any_simplified = false;
  for (auto& operand_indexing : indexing_maps) {
    std::vector<IndexingMap> to_remove, to_add;
    for (IndexingMap map : operand_indexing) {
      to_remove.push_back(map);
      if (map.IsUndefined()) {
        to_add.push_back(map);
      } else if (map.Simplify()) {
        map.RemoveUnusedSymbols();
      } else {
        to_remove.pop_back();
      }
    }
    for (auto& map : to_remove) {
      operand_indexing.erase(map);
    }
    for (auto& map : to_add) {
      operand_indexing.insert(map);
    }
    any_simplified |= !to_remove.empty();
  }
  return any_simplified;
}

HloInstructionIndexing HloInstructionIndexing::FromIndexingMaps(
    absl::Span<const IndexingMap> indexing_maps) {
  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(indexing_maps.size());
  for (const auto& [index, map] : llvm::enumerate(indexing_maps)) {
    instr_indexing.indexing_maps[index].insert(map);
  }
  return instr_indexing;
}

std::string HloInstructionIndexing::ToString(
    const AffineMapPrinter& printer) const {
  std::string s;
  std::stringstream ss(s);
  Print(ss, printer);
  return ss.str();
}

void HloInstructionIndexing::Print(std::ostream& out,
                                   const AffineMapPrinter& printer) const {
  for (const auto& [operand_id, indexing_maps] :
       llvm::enumerate(indexing_maps)) {
    out << "operand id = " << operand_id << ' ';
    for (const auto& indexing_map : indexing_maps) {
      if (indexing_map.IsUndefined()) {
        out << "unknown indexing";
        continue;
      }
      indexing_map.Print(out, printer);
    }
  }
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  AffineMapPrinter printer;
  instr_indexing.Print(out, printer);
  return out;
}

const Shape& GetOutputShape(const HloInstruction* instr, int64_t output_id) {
  return instr->shape().IsTuple()
             ? ShapeUtil::GetSubshape(instr->shape(), {output_id})
             : instr->shape();
}

GroupedByOpIndexingMap GroupIndexingMapsByProducers(
    const HloInstructionIndexing& indexing, const HloInstruction* instr) {
  GroupedByOpIndexingMap result;
  for (const auto& [operand_id, indexing_maps] :
       llvm::enumerate(indexing.indexing_maps)) {
    result[instr->operand(operand_id)].insert(indexing_maps.begin(),
                                              indexing_maps.end());
  }
  return result;
}

GroupedByOpIndexingMap ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, HloInstructionAdaptor target_instr,
    MLIRContext* ctx) {
  auto initial_map = CreateIdentityMap(target_instr.instruction().shape(), ctx);

  GroupedByOpIndexingMap grouped_indexing_maps;
  // If target_instr is a parameter of a fusion, then we create an identity map
  // for the fusion operand.
  if (fusion_adaptor.ContainsInstruction(target_instr)) {
    if (auto parameter_instr =
            DynCast<HloParameterInstruction>(&target_instr.instruction())) {
      const HloInstruction* user = parameter_instr->users().front();
      auto fusion_operand = HloInstructionAdaptor(*user).GetOperand(
          parameter_instr->parameter_number());
      grouped_indexing_maps[&fusion_operand.instruction()] = {initial_map};
      return grouped_indexing_maps;
    }
  }
  grouped_indexing_maps[&target_instr.instruction()].insert(initial_map);

  auto post_order = fusion_adaptor.MakeInstructionPostOrder();

  // Iterator in reversed post-order (use-before-def).
  auto it = std::find(post_order.rbegin(), post_order.rend(), target_instr);
  for (; it != post_order.rend(); ++it) {
    auto producer_indexing = ComputeOutputToInputIndexing(&it->instruction(),
                                                          /*output_id=*/0, ctx);
    auto consumer_indexing_maps =
        grouped_indexing_maps.find(&it->instruction());
    if (consumer_indexing_maps == grouped_indexing_maps.end()) {
      continue;
    }
    // Indexing maps have to be copied because of rehashing. Consider using a
    // different container to get better performance.
    IndexingMapSet consumer_indexing_maps_copy = consumer_indexing_maps->second;
    for (const auto& [producer_operand_id, producer_operand_indexing] :
         llvm::enumerate(producer_indexing.indexing_maps)) {
      auto producer_operand_adaptor = it->GetOperand(producer_operand_id);
      for (const IndexingMap& producer_map : producer_operand_indexing) {
        for (const IndexingMap& consumer_map : consumer_indexing_maps_copy) {
          auto composed_map = ComposeIndexingMaps(consumer_map, producer_map);
          composed_map.Simplify();
          composed_map.RemoveUnusedSymbols();
          grouped_indexing_maps[&producer_operand_adaptor.instruction()].insert(
              composed_map);
        }
      }
    }
  }
  return grouped_indexing_maps;
}

bool FuseProducerConsumerOutputToInputIndexing(
    const HloInstruction* producer_instr,
    absl::flat_hash_map<const HloInstruction*, IndexingMapSet>*
        consumer_indexing,
    MLIRContext* mlir_context) {
  auto producer_indexing = ComputeOutputToInputIndexing(
      producer_instr, /*output_id=*/0, mlir_context);
  auto consumer_indexing_maps = (*consumer_indexing)[producer_instr];
  for (const auto& [producer_operand_id, producer_operand_indexing] :
       llvm::enumerate(producer_indexing.indexing_maps)) {
    const HloInstruction* producer_operand_instr =
        producer_instr->operand(producer_operand_id);
    for (const IndexingMap& producer_map : producer_operand_indexing) {
      for (const IndexingMap& consumer_map : consumer_indexing_maps) {
        (*consumer_indexing)[producer_operand_instr].insert(
            ComposeIndexingMaps(producer_map, consumer_map));
      }
    }
  }
  consumer_indexing->erase(producer_instr);
  return true;
}

HloInstructionIndexing ComputeOutputToInputIndexing(const HloInstruction* instr,
                                                    int output_id,
                                                    MLIRContext* ctx) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeOutputToInputCwiseOpIndexing(instr, ctx);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeOutputToInputBitcastOpIndexing(instr, ctx);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeOutputToInputBroadcastOpIndexing(broadcast, ctx);
  }
  if (auto concat = DynCast<HloConcatenateInstruction>(instr)) {
    return ComputeOutputToInputConcatenateOpIndexing(concat, ctx);
  }
  if (auto constant = DynCast<HloConstantInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto dot = DynCast<HloDotInstruction>(instr)) {
    return ComputeOutputToInputDotOpIndexing(dot, ctx);
  }
  if (auto fusion = DynCast<HloFusionInstruction>(instr)) {
    return ComputeOutputToInputFusionOpIndexing(fusion, output_id, ctx);
  }
  if (auto iota = DynCast<HloIotaInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto pad = DynCast<HloPadInstruction>(instr)) {
    return ComputeOutputToInputPadOpIndexing(pad, ctx);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeOutputToInputReduceOpIndexing(reduce, output_id, ctx);
  }
  if (auto reduce_window = DynCast<HloReduceWindowInstruction>(instr)) {
    return ComputeOutputToInputReduceWindowOpIndexing(reduce_window, output_id,
                                                      ctx);
  }
  if (auto reshape = DynCast<HloReshapeInstruction>(instr)) {
    return ComputeOutputToInputReshapeOpIndexing(reshape, ctx);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, ctx);
  }
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeOutputToInputSliceOpIndexing(slice, ctx);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeOutputToInputTransposeOpIndexing(transpose, ctx);
  }
  // If we cannot compute output-to-input indexing, we return std::nullopt for
  // every op parameter.
  return CreateUnknownIndexing(instr->operand_count());
}

HloInstructionIndexing ComputeInputToOutputIndexing(const HloInstruction* instr,
                                                    int input_id,
                                                    MLIRContext* ctx) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeInputToOutputCwiseOpIndexing(instr, ctx);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeInputToOutputBitcastOpIndexing(instr, ctx);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeInputToOutputBroadcastOpIndexing(broadcast, ctx);
  }
  if (auto concat = DynCast<HloConcatenateInstruction>(instr)) {
    return ComputeInputToOutputConcatenateOpIndexing(concat, input_id, ctx);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeInputToOutputReduceOpIndexing(reduce, input_id, ctx);
  }
  if (auto reshape = DynCast<HloReshapeInstruction>(instr)) {
    return ComputeInputToOutputReshapeOpIndexing(reshape, ctx);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, ctx);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeInputToOutputTransposeOpIndexing(transpose, ctx);
  }
  // If we cannot compute input-to-output indexing, we return std::nullopt for
  // every op result.
  int64_t num_results =
      instr->shape().IsTuple() ? instr->shape().tuple_shapes_size() : 1;
  return CreateUnknownIndexing(num_results);
}

IndexingMap ComputeEpilogueInputToOutputIndexing(
    const HloInstruction* epilogue_root, mlir::MLIRContext* ctx,
    std::function<bool(const HloInstruction*)> is_root) {
  auto* instr = epilogue_root;
  auto root_indexing = CreateIdentityMap(instr->shape(), ctx);
  while (!is_root(instr)) {
    // There can be multiple users, but they must have compatible indexing maps.
    auto* user = instr->users().front();
    auto user_indexing =
        ComputeInputToOutputIndexing(user, user->operand_index(instr), ctx);
    root_indexing = root_indexing * *user_indexing.indexing_maps[0].begin();
    root_indexing.Simplify();
    instr = user;
  }
  return root_indexing;
}

}  // namespace gpu
}  // namespace xla
