/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/indexing_map_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::AffineBinaryOpExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

IndexingMap CreateIdentityMap(const Shape& shape, MLIRContext* ctx) {
  auto dims = shape.dimensions();
  IndexingMap identity_map{
      .affine_map = AffineMap::getMultiDimIdentityMap(dims.size(), ctx),
      .domain = Domain::FromUpperBounds(dims, {})};
  return identity_map;
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);

  HloInstructionIndexing instr_indexing;
  int64_t operand_count = instr->operand_count();
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    instr_indexing.indexing_maps[operand_id].insert(identity_map);
  }
  return instr_indexing;
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({identity_map});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  auto output_dims = bcast->shape().dimensions();

  std::vector<AffineExpr> exprs;
  exprs.reserve(bcast->dimensions().size());
  for (int64_t bcast_dim : bcast->dimensions()) {
    exprs.push_back(getAffineDimExpr(bcast_dim, mlir_context));
  }
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .domain = Domain::FromUpperBounds(output_dims, {})};
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputBroadcastOpIndexing(
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
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(input_shape.rank(), added_dims_sizes.size(),
                                   exprs, mlir_context),
      .domain =
          Domain::FromUpperBounds(input_shape.dimensions(), added_dims_sizes)};

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

// Composes affine maps, i.e. consumer_map ∘ producer_map.
IndexingMap ComposeIndexingMaps(const IndexingMap& producer_map,
                                const IndexingMap& consumer_map) {
  // AffineMap::compose(some_affine_map) actually computes some_affine_map ∘
  // this.
  AffineMap composed_map = mlir::simplifyAffineMap(
      producer_map.affine_map.compose(consumer_map.affine_map));

  // After the composition some of the symbols might become unused, e.g. when a
  // dimension was added by broadcasting as then reduced. We should remove these
  // dimensions from the composed affine map and also from the resulting
  // `domain.symbol_ranges`.
  //
  // For example, if there is a reduction(broadcast):
  //
  //   param = f32[15] parameter(0)
  //   bcast = f32[15, 20] broadcast(p0), dimensions={0}
  //   reduce = f32[15, 20] reduce(bcast, init) dimensions={1}
  //
  // then `reduce` has (d0)[s0] -> (d0, s0) with s0 in [0, 20).
  // and  `bcast` has (d0, d1) -> (d0) indexing map.
  //
  // The composition of there two maps yields (d0)[s0] -> (d0),
  // although `s0` is not used in the mapping. In order to remove such symbols,
  // we get the indices of unused symbols and remove them from the composed
  // affine map and the `domain.symbol_ranges`.
  auto unused_symbols_bit_vector =
      mlir::getUnusedSymbolsBitVector({composed_map});
  composed_map = mlir::compressSymbols(composed_map, unused_symbols_bit_vector);

  // The symbols in the composed map, i.e. combined
  // producer_map.compose(consumer_map) are packed as [symbols(producer_map) |
  // symbols(consumer_map)]. In that order we are adding the symbol ranges while
  // skipping the symbols that are unused.
  std::vector<Range> combined_symbol_ranges;
  combined_symbol_ranges.reserve(producer_map.domain.symbol_ranges.size() +
                                 consumer_map.domain.symbol_ranges.size());
  int64_t symbol_id = 0;
  for (const Range& symbol_range :
       llvm::concat<const Range>(producer_map.domain.symbol_ranges,
                                 consumer_map.domain.symbol_ranges)) {
    if (unused_symbols_bit_vector[symbol_id++]) continue;
    combined_symbol_ranges.push_back(symbol_range);
  }
  IndexingMap composed_indexing_map{
      .affine_map = std::move(composed_map),
      .domain = Domain{.dimension_ranges = consumer_map.domain.dimension_ranges,
                       .symbol_ranges = combined_symbol_ranges}};
  composed_indexing_map.Simplify();
  return composed_indexing_map;
}

// Composes instruction indexing maps starting at the root instruction
// until the HloParameterInstruction is found.
StatusOr<HloInstructionIndexing> ComputeOutputToInputFusionOpIndexing(
    const HloFusionInstruction* fusion, int output_id,
    MLIRContext* mlir_context) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);
  TF_ASSIGN_OR_RETURN(auto grouped_indexing_maps,
                      ComputeGroupedOutputToInputIndexing(
                          *fusion_adaptor, output_id, mlir_context));

  // After the traversal, `grouped_indexing_maps` is keyed by
  // HloParameterInstructions. Convert them back to the operand id and return.
  HloInstructionIndexing fusion_indexing;
  for (auto [operand_id, operand] : llvm::enumerate(fusion->operands())) {
    fusion_indexing.indexing_maps[operand_id] = grouped_indexing_maps[operand];
  }
  return fusion_indexing;
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputDotOpIndexing(
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
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dims,
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_contracting_dims));

  for (int64_t lhs_non_contracting_dim : lhs_non_contracting_dims) {
    lhs_exprs[lhs_non_contracting_dim] =
        getAffineDimExpr(output_dim_id++, mlir_context);
  }

  // rhs_non_contracting_dims
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dims,
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_contracting_dims));

  for (int64_t rhs_non_contracting_dim : rhs_non_contracting_dims) {
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

  IndexingMap lhs_indexing_map{
      .affine_map = AffineMap::get(dot->shape().rank(), input_dim_sizes.size(),
                                   lhs_exprs, mlir_context),
      .domain =
          Domain::FromUpperBounds(dot->shape().dimensions(), input_dim_sizes)};

  IndexingMap rhs_indexing_map{
      .affine_map = AffineMap::get(dot->shape().rank(), input_dim_sizes.size(),
                                   rhs_exprs, mlir_context),
      .domain =
          Domain::FromUpperBounds(dot->shape().dimensions(), input_dim_sizes)};
  return HloInstructionIndexing::FromIndexingMaps(
      {lhs_indexing_map, rhs_indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputReduceOpIndexing(
    const HloReduceInstruction* reduce, int output_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());

  const Shape& input_shape = reduce->operand(output_id)->shape();
  const Shape& output_shape = reduce->shape().IsTuple()
                                  ? ShapeUtil::GetSubshape(reduce->shape(), {0})
                                  : reduce->shape();

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
  IndexingMap inputs_indexing_map{
      .affine_map = AffineMap::get(output_shape.rank(), reduce_dims_ids.size(),
                                   exprs, mlir_context),
      .domain = Domain::FromUpperBounds(output_shape.dimensions(),
                                        parallel_dims_sizes)};
  IndexingMap inits_indexing_map{
      .affine_map = AffineMap::get(output_shape.rank(), /*symbolCount=*/0, {},
                                   mlir_context),
      .domain = Domain::FromUpperBounds(output_shape.dimensions(), {})};

  HloInstructionIndexing instr_indexing;
  for (int64_t id = 0; id < reduce->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing_map);
  }
  for (int64_t id = reduce->input_count(); id < reduce->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inits_indexing_map);
  }
  return instr_indexing;
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputReduceOpIndexing(
    const HloReduceInstruction* reduce, int input_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());
  const Shape& input_shape = reduce->operand(input_id)->shape();
  const Shape& output_shape = reduce->shape().IsTuple()
                                  ? ShapeUtil::GetSubshape(reduce->shape(), {0})
                                  : reduce->shape();
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
    inits_exprs.push_back(
        mlir::getAffineSymbolExpr(output_dim_id++, mlir_context));
  }
  IndexingMap inputs_indexing_map{
      .affine_map = AffineMap::get(input_shape.rank(), /*symbolCount=*/0,
                                   inputs_exprs, mlir_context),
      .domain = Domain::FromUpperBounds(input_shape.dimensions(), {})};
  IndexingMap inits_indexing_map{
      .affine_map = AffineMap::get(0, /*symbolCount=*/output_rank, inits_exprs,
                                   mlir_context),
      .domain = Domain::FromUpperBounds({}, output_shape.dimensions())};

  HloInstructionIndexing instr_indexing;
  for (int64_t id = 0; id < reduce->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing_map);
  }
  for (int64_t id = reduce->input_count(); id < reduce->operand_count(); ++id) {
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
    linear_index = getAffineBinaryOpExpr(
        AffineExprKind::Add, linear_index,
        getAffineBinaryOpExpr(AffineExprKind::Mul,
                              getAffineConstantExpr(stride, mlir_context),
                              dimension_expr));
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
    AffineExpr stride_expr = getAffineConstantExpr(stride, mlir_context);
    multi_index.push_back(getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                remainder, stride_expr));
    remainder =
        getAffineBinaryOpExpr(AffineExprKind::Mod, remainder, stride_expr);
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
AffineMap ComputeReshapeIndexingMap(absl::Span<const int64_t> input_dims,
                                    absl::Span<const int64_t> output_dims,
                                    MLIRContext* mlir_context) {
  size_t input_rank = input_dims.size();
  size_t output_rank = output_dims.size();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);

  std::vector<AffineExpr> output_dims_exprs;

  // Find subshapes with the same element count and compute indexing for them.
  int64_t input_num_elements = 1;
  int64_t output_num_elements = 1;
  std::vector<int64_t> input_subshape, output_subshape;
  size_t input_dim_id = 0, output_dim_id = 0;
  while (input_dim_id < input_rank || output_dim_id < output_rank ||
         !input_subshape.empty()) {
    if (input_dim_id < input_rank &&
        (input_subshape.empty() || input_num_elements < output_num_elements ||
         input_dims[input_dim_id] == 1)) {
      input_num_elements *= input_dims[input_dim_id];
      input_subshape.push_back(input_dims[input_dim_id]);
      ++input_dim_id;
      continue;
    }
    if (output_dim_id < output_rank &&
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

StatusOr<HloInstructionIndexing> ComputeOutputToInputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  auto input_dims = reshape->operand(0)->shape().dimensions();
  auto output_dims = reshape->shape().dimensions();

  IndexingMap reshape_indexing_map{
      .affine_map =
          ComputeReshapeIndexingMap(input_dims, output_dims, mlir_context),
      .domain = Domain::FromUpperBounds(output_dims, {})};
  reshape_indexing_map.Simplify();
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}
StatusOr<HloInstructionIndexing> ComputeInputToOutputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  auto input_dims = reshape->operand(0)->shape().dimensions();
  auto output_dims = reshape->shape().dimensions();

  IndexingMap reshape_indexing_map{
      .affine_map =
          ComputeReshapeIndexingMap(output_dims, input_dims, mlir_context),
      .domain = Domain::FromUpperBounds(input_dims, {})};
  reshape_indexing_map.Simplify();
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeReverseOpIndexing(
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
    auto dim_bound = getAffineConstantExpr(output_dim - 1, mlir_context);
    auto neg_dim_expr = getAffineBinaryOpExpr(
        AffineExprKind::Mul, getAffineConstantExpr(-1, mlir_context), dim_expr);
    exprs.push_back(
        getAffineBinaryOpExpr(AffineExprKind::Add, neg_dim_expr, dim_bound));
  }

  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .domain = Domain::FromUpperBounds(output_dims, {})};

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  auto output_rank = slice->shape().rank();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);
  for (int64_t dim = 0; dim < output_rank; ++dim) {
    AffineExpr offset =
        getAffineConstantExpr(slice->slice_starts()[dim], mlir_context);
    AffineExpr stride =
        getAffineConstantExpr(slice->slice_strides()[dim], mlir_context);
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);

    AffineExpr mul =
        getAffineBinaryOpExpr(AffineExprKind::Mul, stride, dim_expr);
    exprs.push_back(getAffineBinaryOpExpr(AffineExprKind::Add, offset, mul));
  }
  IndexingMap indexing_map{
      .affine_map =
          AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      .domain = Domain::FromUpperBounds(slice->shape().dimensions(), {})};
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  AffineMap inverse_permutation = ComputeTransposeIndexingMap(
      InversePermutation(transpose->dimensions()), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({IndexingMap{
      .affine_map = inverse_permutation,
      .domain = Domain::FromUpperBounds(transpose->shape().dimensions(), {})}});
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  AffineMap forward_permutation =
      ComputeTransposeIndexingMap(transpose->dimensions(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps(
      {IndexingMap{.affine_map = forward_permutation,
                   .domain = Domain::FromUpperBounds(
                       transpose->operand(0)->shape().dimensions(), {})}});
}

StatusOr<AffineMap> ComputeOutputToInputBitcastOpIndexingImpl(
    const Shape& input_shape, const Shape& output_shape,
    MLIRContext* mlir_context) {
  ShapeUtil::BitcastDecomposition decomposed_bitcast =
      ShapeUtil::DecomposeBitcast(input_shape, output_shape);

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          decomposed_bitcast)) {
    auto permutation = ShapeUtil::DeduceTransposeDimensionsForBitcast(
        input_shape, output_shape);
    CHECK(permutation.has_value())
        << "Failed to deduce permutation for a bitcast.";

    return ComputeTransposeIndexingMap(InversePermutation(permutation.value()),
                                       mlir_context);
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposed_bitcast)) {
    return ComputeReshapeIndexingMap(input_shape.dimensions(),
                                     output_shape.dimensions(), mlir_context);
  }
  // `trt` stands for transpose-reshape-transpose decomposition of bitcast.
  auto trt = std::get<ShapeUtil::BitcastDecompositionTrt>(decomposed_bitcast);
  AffineMap transpose_map_1 = ComputeTransposeIndexingMap(
      InversePermutation(trt.transpose1_dims), mlir_context);
  AffineMap reshape_map =
      ComputeReshapeIndexingMap(trt.transpose1_shape.dimensions(),
                                trt.reshape_shape.dimensions(), mlir_context);
  AffineMap transpose_map_2 = ComputeTransposeIndexingMap(
      InversePermutation(trt.transpose2_dims), mlir_context);
  return transpose_map_1.compose(reshape_map).compose(transpose_map_2);
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  const Shape& input_shape = bitcast->operand(0)->shape();
  const Shape& output_shape = bitcast->shape();
  TF_ASSIGN_OR_RETURN(auto bitcast_affine_map,
                      ComputeOutputToInputBitcastOpIndexingImpl(
                          input_shape, output_shape, mlir_context));
  IndexingMap bitcast_indexing_map{
      .affine_map = bitcast_affine_map,
      .domain = Domain::FromUpperBounds(output_shape.dimensions(), {})};
  bitcast_indexing_map.Simplify();

  return HloInstructionIndexing::FromIndexingMaps({bitcast_indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  const Shape& input_shape = bitcast->operand(0)->shape();
  const Shape& output_shape = bitcast->shape();

  TF_ASSIGN_OR_RETURN(auto bitcast_affine_map,
                      ComputeOutputToInputBitcastOpIndexingImpl(
                          output_shape, input_shape, mlir_context));

  IndexingMap bitcast_indexing_map{
      .affine_map = bitcast_affine_map,
      .domain = Domain::FromUpperBounds(input_shape.dimensions(), {})};
  bitcast_indexing_map.Simplify();

  return HloInstructionIndexing::FromIndexingMaps({bitcast_indexing_map});
}

}  // namespace

bool IndexingMap::Simplify() {
  auto* mlir_context = affine_map.getContext();
  IndexingMapSimplifier simplifier{mlir_context};
  for (const auto& [index, range] : llvm::enumerate(domain.dimension_ranges)) {
    simplifier.SetInclusiveBounds(getAffineDimExpr(index, mlir_context),
                                  range.lower_bound, range.upper_bound - 1);
  }
  for (const auto& [index, range] : llvm::enumerate(domain.symbol_ranges)) {
    simplifier.SetInclusiveBounds(getAffineSymbolExpr(index, mlir_context),
                                  range.lower_bound, range.upper_bound - 1);
  }
  AffineMap simplified_affine_map = simplifier.Simplify(affine_map);
  if (simplified_affine_map == affine_map) {
    return false;
  }
  affine_map = simplified_affine_map;
  return true;
}

bool HloInstructionIndexing::Simplify() {
  bool any_simplified = false;
  for (auto& operand_indexing : indexing_maps) {
    std::vector<IndexingMap> to_remove;
    std::vector<IndexingMap> to_add;
    absl::flat_hash_set<IndexingMap>& indexing_maps = operand_indexing.second;
    for (IndexingMap map : indexing_maps) {
      to_remove.push_back(map);
      if (map.Simplify()) {
        to_add.push_back(map);
      } else {
        to_remove.pop_back();
      }
    }
    for (auto& map : to_remove) {
      indexing_maps.erase(map);
    }
    for (auto& map : to_add) {
      indexing_maps.insert(map);
    }
    any_simplified |= !to_remove.empty();
  }
  return any_simplified;
}

bool operator==(const Range& lhs, const Range& rhs) {
  return lhs.lower_bound == rhs.lower_bound &&
         lhs.upper_bound == rhs.upper_bound;
}

bool operator==(const Domain& lhs, const Domain& rhs) {
  return lhs.dimension_ranges == rhs.dimension_ranges &&
         lhs.symbol_ranges == rhs.symbol_ranges;
}

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.affine_map == rhs.affine_map && lhs.domain == rhs.domain;
}

std::ostream& operator<<(std::ostream& out, const Range& range) {
  out << '[' << range.lower_bound << ", " << range.upper_bound << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Domain& domain) {
  for (const auto& [index, range] : llvm::enumerate(domain.dimension_ranges)) {
    out << 'd' << index << " in " << range << '\n';
  }
  for (const auto& [index, range] : llvm::enumerate(domain.symbol_ranges)) {
    out << 's' << index << " in " << range << '\n';
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map) {
  out << ToString(indexing_map.affine_map) << " with domain\n"
      << indexing_map.domain << "\n";
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  for (const auto& [operand_id, indexing_maps] : instr_indexing.indexing_maps) {
    out << "operand id = " << operand_id << ' ';
    for (const auto& indexing_map : indexing_maps) {
      out << indexing_map;
    }
  }
  return out;
}

std::string Range::ToString() const { return ToStringImpl(*this); }

std::string Domain::ToString() const { return ToStringImpl(*this); }

Domain Domain::FromUpperBounds(absl::Span<const int64_t> dimension_upper_bounds,
                               absl::Span<const int64_t> symbol_upper_bounds) {
  Domain domain;
  domain.dimension_ranges.reserve(dimension_upper_bounds.size());
  for (const int64_t ub : dimension_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.dimension_ranges.push_back({.lower_bound = 0, .upper_bound = ub});
  }
  domain.symbol_ranges.reserve(symbol_upper_bounds.size());
  for (const int64_t ub : symbol_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.symbol_ranges.push_back({.lower_bound = 0, .upper_bound = ub});
  }
  return domain;
}

std::string IndexingMap::ToString() const { return ToStringImpl(*this); }

HloInstructionIndexing HloInstructionIndexing::FromIndexingMaps(
    absl::Span<const IndexingMap> indexing_maps) {
  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.reserve(indexing_maps.size());
  for (const auto& [index, map] : llvm::enumerate(indexing_maps)) {
    instr_indexing.indexing_maps[index].insert(map);
  }
  return instr_indexing;
}

std::string HloInstructionIndexing::ToString() const {
  return ToStringImpl(*this);
}

absl::flat_hash_map<const HloInstruction*, absl::flat_hash_set<IndexingMap>>
GroupIndexingMapsByProducers(const HloInstructionIndexing& indexing,
                             const HloInstruction* instr) {
  absl::flat_hash_map<const HloInstruction*, absl::flat_hash_set<IndexingMap>>
      result;
  for (const auto& [operand_id, indexing_maps] : indexing.indexing_maps) {
    result[instr->operand(operand_id)].insert(indexing_maps.begin(),
                                              indexing_maps.end());
  }
  return result;
}

AffineMap ComputeTransposeIndexingMap(absl::Span<const int64_t> permutation,
                                      MLIRContext* mlir_context) {
  return AffineMap::getPermutationMap(
      std::vector<unsigned>(permutation.begin(), permutation.end()),
      mlir_context);
}

StatusOr<GroupedByOpIndexingMap> ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, int output_id, MLIRContext* ctx) {
  auto root = fusion_adaptor.GetRoots()[output_id];

  auto initial_map = CreateIdentityMap(root.instruction().shape(), ctx);

  GroupedByOpIndexingMap grouped_indexing_maps;
  grouped_indexing_maps[&root.instruction()].insert(initial_map);

  auto post_order = fusion_adaptor.MakeInstructionPostOrder();

  // Iterator in reversed post-order (use-before-def).
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    TF_ASSIGN_OR_RETURN(auto producer_indexing,
                        ComputeOutputToInputIndexing(&it->instruction(),
                                                     /*output_id=*/0, ctx));

    auto consumer_indexing_maps = grouped_indexing_maps[&it->instruction()];
    for (const auto& [producer_operand_id, producer_operand_indexing] :
         producer_indexing.indexing_maps) {
      auto producer_operand_adaptor = it->GetOperand(producer_operand_id);
      for (const IndexingMap& producer_map : producer_operand_indexing) {
        for (const IndexingMap& consumer_map : consumer_indexing_maps) {
          grouped_indexing_maps[&producer_operand_adaptor.instruction()].insert(
              ComposeIndexingMaps(producer_map, consumer_map));
        }
      }
    }
  }

  return grouped_indexing_maps;
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputIndexing(
    const HloInstruction* instr, int output_id, MLIRContext* ctx) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeOutputToInputCwiseOpIndexing(instr, ctx);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeOutputToInputBitcastOpIndexing(instr, ctx);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeOutputToInputBroadcastOpIndexing(broadcast, ctx);
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
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeOutputToInputReduceOpIndexing(reduce, output_id, ctx);
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
  return InvalidArgument("Unsupported instruction type");
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputIndexing(
    const HloInstruction* instr, int input_id, MLIRContext* ctx) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeInputToOutputCwiseOpIndexing(instr, ctx);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeInputToOutputBitcastOpIndexing(instr, ctx);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeInputToOutputBroadcastOpIndexing(broadcast, ctx);
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
  return InvalidArgument("Unsupported instruction type");
}

}  // namespace gpu
}  // namespace xla
