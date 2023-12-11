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

#include "xla/service/gpu/model/tile_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/matmul_utils.h"
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
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

StatusOr<HloInstructionIndexing> ComputeOutputToInputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  auto dims = instr->shape().dimensions();
  IndexingMap identity_map{.affine_map = AffineMap::getMultiDimIdentityMap(
                               dims.size(), mlir_context),
                           .input_dims_sizes = {}};

  HloInstructionIndexing instr_indexing;
  int64_t operand_count = instr->operand_count();
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    instr_indexing.indexing_maps[operand_id].insert(identity_map);
  }
  return instr_indexing;
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  auto dims = instr->shape().dimensions();
  IndexingMap identity_map{.affine_map = AffineMap::getMultiDimIdentityMap(
                               dims.size(), mlir_context),
                           .input_dims_sizes = {}};
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
      .input_dims_sizes = {}};
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
      .input_dims_sizes = std::move(added_dims_sizes)};
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
  // `input_dim_sizes`.
  //
  // For example, if there is a reduction(broadcast):
  //
  //   param = f32[15] parameter(0)
  //   bcast = f32[15, 20] broadcast(p0), dimensions={0}
  //   reduce = f32[15, 20] reduce(bcast, init) dimensions={1}
  //
  // then `reduce` has (d0)[s0] -> (d0, s0) with size(s0) = 20
  // and  `bcast` has (d0, d1) -> (d0) indexing map.
  //
  // The composition of there two maps yields (d0)[s0] -> (d0) with size(s0),
  // although `s0` is not used in the mapping. In order to remove such symbols,
  // we get the indices of unused symbols and remove them from the composed
  // affine map and the `input_dim_sizes`.
  auto unused_symbols_bit_vector =
      mlir::getUnusedSymbolsBitVector({composed_map});
  composed_map = mlir::compressSymbols(composed_map, unused_symbols_bit_vector);

  // The input dims symbols in the composed map, i.e. combined
  // producer_map.compose(consumer_map) are packed as [symbols(producer_map) |
  // symbols(consumer_map)]. In that order we are adding the sizes for the input
  // dims while skipping the symbols that are unused.
  std::vector<int64_t> combined_sizes;
  combined_sizes.reserve(producer_map.input_dims_sizes.size() +
                         consumer_map.input_dims_sizes.size());
  int64_t symbol_id = 0;
  for (int64_t dim : llvm::concat<const int64_t>(
           producer_map.input_dims_sizes, consumer_map.input_dims_sizes)) {
    if (unused_symbols_bit_vector[symbol_id++]) continue;
    combined_sizes.push_back(dim);
  }
  return IndexingMap{.affine_map = std::move(composed_map),
                     .input_dims_sizes = std::move(combined_sizes)};
}

// Composes instruction indexing maps starting at the root instruction
// until the HloParameterInstruction is found.
StatusOr<HloInstructionIndexing> ComputeOutputToInputFusionOpIndexing(
    const HloFusionInstruction* fusion, int output_id,
    MLIRContext* mlir_context) {
  const HloInstruction* root =
      fusion->shape().IsTuple()
          ? fusion->fused_expression_root()->operand(output_id)
          : fusion->fused_expression_root();
  TF_ASSIGN_OR_RETURN(auto root_indexing, ComputeOutputToInputIndexing(
                                              root, output_id, mlir_context));

  auto grouped_indexing_maps =
      GroupIndexingMapsByProducers(root_indexing, root);

  // `bfs` is initialized with all producer instructions of the fusion root that
  // are not parameters of the fusion.
  std::queue<const HloInstruction*> bfs;
  for (const auto& [instr, indexing_maps] : grouped_indexing_maps) {
    if (instr->opcode() == HloOpcode::kParameter) continue;
    bfs.push(instr);
  }
  while (!bfs.empty()) {
    const HloInstruction* producer_instr = bfs.front();
    bfs.pop();
    TF_CHECK_OK(FuseProducerConsumerOutputToInputIndexing(
        producer_instr, &grouped_indexing_maps, mlir_context));

    for (const HloInstruction* producer_operand_instr :
         producer_instr->operands()) {
      if (producer_operand_instr->opcode() != HloOpcode::kParameter) {
        bfs.push(producer_operand_instr);
      }
    }
  }
  // After the traversal, `grouped_indexing_maps` is keyed by
  // HloParameterInstructions. Convert them back to the operand id and return.
  HloInstructionIndexing fusion_indexing;
  for (auto& [instr, indexing_maps] : grouped_indexing_maps) {
    fusion_indexing.indexing_maps[instr->parameter_number()] =
        std::move(indexing_maps);
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
      .input_dims_sizes = input_dim_sizes};

  IndexingMap rhs_indexing_map{
      .affine_map = AffineMap::get(dot->shape().rank(), input_dim_sizes.size(),
                                   rhs_exprs, mlir_context),
      .input_dims_sizes = input_dim_sizes};
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
      .input_dims_sizes = parallel_dims_sizes};
  IndexingMap inits_indexing_map{
      .affine_map = AffineMap::get(output_shape.rank(), /*symbolCount=*/0, {},
                                   mlir_context),
      .input_dims_sizes = {}};

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
      .input_dims_sizes = {}};
  IndexingMap inits_indexing_map{
      .affine_map = AffineMap::get(0, /*symbolCount=*/output_rank, inits_exprs,
                                   mlir_context),
      .input_dims_sizes = {output_shape.dimensions().begin(),
                           output_shape.dimensions().end()}};

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
IndexingMap ComputeReshapeIndexingMap(absl::Span<const int64_t> input_dims,
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
  return IndexingMap{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .input_dims_sizes = {}};
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  auto input_dims = reshape->operand(0)->shape().dimensions();
  auto output_dims = reshape->shape().dimensions();
  IndexingMap reshape_indexing_map =
      ComputeReshapeIndexingMap(input_dims, output_dims, mlir_context);

  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  auto input_dims = reshape->operand(0)->shape().dimensions();
  auto output_dims = reshape->shape().dimensions();
  IndexingMap reshape_indexing_map =
      ComputeReshapeIndexingMap(output_dims, input_dims, mlir_context);

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
      .input_dims_sizes = {}};

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
      .input_dims_sizes = {}};
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

IndexingMap ComputeTransposeIndexingMap(absl::Span<const int64_t> permutation,
                                        MLIRContext* mlir_context) {
  auto forward_permutation = AffineMap::getPermutationMap(
      std::vector<unsigned>(permutation.begin(), permutation.end()),
      mlir_context);
  return IndexingMap{
      .affine_map = mlir::inversePermutation(forward_permutation),
      .input_dims_sizes = {}};
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  return HloInstructionIndexing::FromIndexingMaps(
      {ComputeTransposeIndexingMap(transpose->dimensions(), mlir_context)});
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  auto forward_permutation = AffineMap::getPermutationMap(
      std::vector<unsigned>(transpose->dimensions().begin(),
                            transpose->dimensions().end()),
      mlir_context);
  return HloInstructionIndexing::FromIndexingMaps(
      {IndexingMap{.affine_map = forward_permutation, .input_dims_sizes = {}}});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputBitcastOpIndexingImpl(
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
    return HloInstructionIndexing::FromIndexingMaps(
        {ComputeTransposeIndexingMap(*permutation, mlir_context)});
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposed_bitcast)) {
    IndexingMap reshape_indexing_map = ComputeReshapeIndexingMap(
        input_shape.dimensions(), output_shape.dimensions(), mlir_context);
    return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
  }
  // `trt` stands for transpose-reshape-transpose decomposition of bitcast.
  auto trt = std::get<ShapeUtil::BitcastDecompositionTrt>(decomposed_bitcast);
  IndexingMap transpose_map_1 =
      ComputeTransposeIndexingMap(trt.transpose1_dims, mlir_context);
  IndexingMap reshape_map =
      ComputeReshapeIndexingMap(trt.transpose1_shape.dimensions(),
                                trt.reshape_shape.dimensions(), mlir_context);
  IndexingMap transpose_map_2 =
      ComputeTransposeIndexingMap(trt.transpose2_dims, mlir_context);
  IndexingMap composed_map = ComposeIndexingMaps(
      ComposeIndexingMaps(transpose_map_1, reshape_map), transpose_map_2);
  return HloInstructionIndexing::FromIndexingMaps({composed_map});
}

StatusOr<HloInstructionIndexing> ComputeOutputToInputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  const Shape& input_shape = bitcast->operand(0)->shape();
  const Shape& output_shape = bitcast->shape();
  return ComputeOutputToInputBitcastOpIndexingImpl(input_shape, output_shape,
                                                   mlir_context);
}

StatusOr<HloInstructionIndexing> ComputeInputToOutputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  const Shape& input_shape = bitcast->operand(0)->shape();
  const Shape& output_shape = bitcast->shape();
  return ComputeOutputToInputBitcastOpIndexingImpl(output_shape, input_shape,
                                                   mlir_context);
}

template <typename T>
std::string ToStringImpl(const T& value) {
  std::string s;
  std::stringstream ss(s);
  ss << value;
  return ss.str();
}

int64_t FloorDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor -
         (((dividend >= 0) != (divisor >= 0) && dividend % divisor) ? 1 : 0);
}

struct IndexingMapSimplifier {
  struct Bounds {
    int64_t lower;
    int64_t upper;
  };

  Bounds BoundsInclusive(AffineExpr expr) {
    auto bound = bounds.find(expr);
    if (bound != bounds.end()) return bound->second;

    switch (expr.getKind()) {
      case AffineExprKind::Constant: {
        int64_t value = mlir::cast<mlir::AffineConstantExpr>(expr).getValue();
        return bounds[expr] = {value, value};
      }
      case AffineExprKind::DimId: {
        int64_t size =
            dimension_sizes[mlir::cast<AffineDimExpr>(expr).getPosition()];
        return bounds[expr] = {0, size - 1};
      }
      case AffineExprKind::SymbolId: {
        int64_t size =
            symbol_sizes[mlir::cast<AffineSymbolExpr>(expr).getPosition()];
        return bounds[expr] = {0, size - 1};
      }
      default:
        auto binary_op = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
        CHECK(binary_op);
        auto lhs = BoundsInclusive(binary_op.getLHS());
        auto rhs = BoundsInclusive(binary_op.getRHS());

        auto& result = bounds[expr];
        switch (expr.getKind()) {
          case AffineExprKind::Add:
            return result = {lhs.lower + rhs.lower, lhs.upper + rhs.upper};
          case AffineExprKind::Mul: {
            int64_t a = lhs.lower * rhs.lower;
            int64_t b = lhs.upper * rhs.upper;
            return result = {std::min(a, b), std::max(a, b)};
          }
          case AffineExprKind::Mod: {
            CHECK_EQ(rhs.lower, rhs.upper) << "RHS of mod must be a constant";
            int64_t m = rhs.lower;
            if (0 <= lhs.lower && lhs.upper < m) {
              return result = lhs;
            }
            return result = {0, m - 1};
          }
          case AffineExprKind::FloorDiv: {
            CHECK_EQ(rhs.lower, rhs.upper)
                << "RHS of floor_div must be a constant";
            int64_t d = rhs.lower;
            int a = FloorDiv(lhs.lower, d);
            int b = FloorDiv(lhs.upper, d);
            return result = {std::min(a, b), std::max(a, b)};
          }
          default:
            // We don't use ceildiv, so we don't support it.
            LOG(FATAL) << "Unsupported expression";
        }
    }
  }

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  AffineExpr RewriteMod(AffineBinaryOpExpr mod) {
    auto lhs_simplified = SimplifyOnce(mod.getLHS());

    auto lhs = BoundsInclusive(lhs_simplified);
    auto rhs = BoundsInclusive(mod.getRHS());

    // a % b where b is always larger than a?
    if (0 <= lhs.lower && lhs.upper < rhs.lower) return lhs_simplified;

    // The logic below assumes we have a constant RHS.
    if (rhs.lower != rhs.upper) return mod;
    int64_t m = rhs.lower;

    auto new_lhs = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
      if (expr.getKind() != AffineExprKind::Mul) {
        return true;
      }

      auto mul_rhs =
          BoundsInclusive(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
      bool remove = mul_rhs.lower == mul_rhs.upper && (mul_rhs.lower % m) == 0;
      return !remove;  // We keep it if we don't remove it!
    });

    // If we weren't able to remove or simplify anything, return the original
    // expression.
    if (new_lhs == mod.getLHS()) {
      return mod;
    }
    // If we removed everything, return 0.
    if (!new_lhs) {
      return getAffineConstantExpr(0, mlir_context);
    }
    // Otherwise, return new_sum % m.
    return getAffineBinaryOpExpr(AffineExprKind::Mod, new_lhs, mod.getRHS());
  }

  // Simplifier for floordiv.
  // - Rewrites (a * 100 + ...) / 100 to a + (...) / 100
  // - Rewrites a / 100 to 0 when a is known to be less than 100.
  AffineExpr RewriteFloorDiv(AffineBinaryOpExpr div) {
    auto lhs_simplified = SimplifyOnce(div.getLHS());
    auto lhs = BoundsInclusive(lhs_simplified);
    auto rhs = BoundsInclusive(div.getRHS());

    if (0 <= lhs.lower && lhs.upper < rhs.lower) {
      return getAffineConstantExpr(0, mlir_context);
    }

    // The logic below assumes we have a constant RHS.
    if (rhs.lower != rhs.upper) return div;
    int64_t d = rhs.lower;

    int64_t a = FloorDiv(lhs.lower, d);
    int64_t b = FloorDiv(lhs.upper, d);
    if (a == b) {
      return getAffineConstantExpr(a, mlir_context);
    }

    AffineExpr extracted = getAffineConstantExpr(0, mlir_context);
    auto new_dividend = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
      if (auto multiplier = GetConstantRhsMultiplier(expr)) {
        // (x * 7 + ...) / 3 -> can't extract. We could extract x * 2 and keep
        // one x, but we currently have no reason to do that.
        if (*multiplier % d != 0) return true;
        int64_t factor = *multiplier / d;
        extracted = getAffineBinaryOpExpr(
            AffineExprKind::Add, extracted,
            getAffineBinaryOpExpr(AffineExprKind::Mul,
                                  mlir::cast<AffineBinaryOpExpr>(expr).getLHS(),
                                  getAffineConstantExpr(factor, mlir_context)));
        // Remove from dividend.
        return false;
      }

      // Not a constant multiplier, keep in dividend.
      return true;
    });

    // If we removed everything, skip the div.
    if (!new_dividend) return extracted;
    // If we removed nothing, return the original division.
    if (extracted == getAffineConstantExpr(0, mlir_context) &&
        new_dividend == div.getLHS()) {
      return div;
    }

    return getAffineBinaryOpExpr(
        AffineExprKind::Add, extracted,
        getAffineBinaryOpExpr(AffineExprKind::FloorDiv, new_dividend,
                              div.getRHS()));
  }

  std::optional<int64_t> GetConstantRhsMultiplier(AffineExpr expr) {
    if (expr.getKind() != AffineExprKind::Mul) return std::nullopt;
    auto bound = BoundsInclusive(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
    if (bound.lower != bound.upper) return std::nullopt;
    return bound.lower;
  }

  AffineExpr RewriteSumIf(AffineExpr expr,
                          const std::function<bool(AffineExpr)>& pred) {
    if (expr.getKind() == AffineExprKind::Add) {
      auto add = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
      auto lhs = RewriteSumIf(add.getLHS(), pred);
      auto rhs = RewriteSumIf(add.getRHS(), pred);
      if (lhs == add.getLHS() && rhs == add.getRHS()) {
        return add;
      }
      if (lhs && rhs) {
        return getAffineBinaryOpExpr(AffineExprKind::Add, lhs, rhs);
      }
      return lhs ? lhs : (rhs ? rhs : nullptr);
    }
    return pred(expr) ? expr : nullptr;
  }

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  AffineExpr SimplifyOnce(AffineExpr expr) {
    switch (expr.getKind()) {
      case AffineExprKind::Mul:
      case AffineExprKind::Add: {
        auto binop = mlir::cast<AffineBinaryOpExpr>(expr);
        auto lhs = SimplifyOnce(binop.getLHS());
        auto rhs = SimplifyOnce(binop.getRHS());
        if (lhs == binop.getLHS() && rhs == binop.getRHS()) {
          return expr;
        }
        return getAffineBinaryOpExpr(expr.getKind(), lhs, rhs);
      }
      case AffineExprKind::Mod:
        return RewriteMod(mlir::cast<AffineBinaryOpExpr>(expr));
      case AffineExprKind::FloorDiv:
        return RewriteFloorDiv(mlir::cast<AffineBinaryOpExpr>(expr));
      default:
        return expr;
    }
  }

  // Simplifies the expression as much as possible.
  AffineExpr Simplify(AffineExpr expr) {
    while (true) {
      auto simplified = SimplifyOnce(expr);
      if (simplified == expr) return expr;
      expr = simplified;
    }
  }

  MLIRContext* mlir_context;
  absl::Span<const int64_t> dimension_sizes;
  absl::Span<const int64_t> symbol_sizes;
  llvm::DenseMap<AffineExpr, Bounds> bounds{};
};

}  // namespace

bool IndexingMap::Simplify(absl::Span<const int64_t> dimension_sizes) {
  IndexingMapSimplifier simplifier{affine_map.getContext(), dimension_sizes,
                                   input_dims_sizes};
  std::vector<AffineExpr> results;
  results.reserve(affine_map.getNumResults());
  bool any_changed = false;
  for (auto expr : affine_map.getResults()) {
    auto simplified = simplifier.Simplify(expr);
    any_changed |= simplified != expr;
    results.push_back(simplified);
  }

  if (!any_changed) {
    return false;
  }

  affine_map = mlir::simplifyAffineMap(
      AffineMap::get(affine_map.getNumDims(), affine_map.getNumSymbols(),
                     results, affine_map.getContext()));
  return true;
}

bool HloInstructionIndexing::Simplify(
    absl::Span<const int64_t> dimension_sizes) {
  bool any_simplified = false;
  for (auto& operand_indexing : indexing_maps) {
    std::vector<IndexingMap> to_remove;
    std::vector<IndexingMap> to_add;
    absl::flat_hash_set<IndexingMap>& indexing_maps = operand_indexing.second;
    for (IndexingMap map : indexing_maps) {
      to_remove.push_back(map);
      if (map.Simplify(dimension_sizes)) {
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

std::string ToString(const AffineMap& affine_map) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  affine_map.print(ss);
  return s;
}

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.affine_map == rhs.affine_map &&
         lhs.input_dims_sizes == rhs.input_dims_sizes;
}

std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map) {
  out << ToString(indexing_map.affine_map) << " with sizes "
      << absl::StrJoin(indexing_map.input_dims_sizes, ", ") << "\n";
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

Status FuseProducerConsumerOutputToInputIndexing(
    const HloInstruction* producer_instr,
    absl::flat_hash_map<const HloInstruction*,
                        absl::flat_hash_set<IndexingMap>>* consumer_indexing,
    MLIRContext* mlir_context) {
  TF_ASSIGN_OR_RETURN(auto producer_indexing,
                      ComputeOutputToInputIndexing(
                          producer_instr, /*output_id=*/0, mlir_context));

  auto consumer_indexing_maps = (*consumer_indexing)[producer_instr];
  for (const auto& [producer_operand_id, producer_operand_indexing] :
       producer_indexing.indexing_maps) {
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
  return OkStatus();
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
    const HloInstruction* instr, int input_id, MLIRContext* mlir_context) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeInputToOutputCwiseOpIndexing(instr, mlir_context);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeInputToOutputBitcastOpIndexing(instr, mlir_context);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeInputToOutputBroadcastOpIndexing(broadcast, mlir_context);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeInputToOutputReduceOpIndexing(reduce, input_id, mlir_context);
  }
  if (auto reshape = DynCast<HloReshapeInstruction>(instr)) {
    return ComputeInputToOutputReshapeOpIndexing(reshape, mlir_context);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, mlir_context);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeInputToOutputTransposeOpIndexing(transpose, mlir_context);
  }
  return InvalidArgument("Unsupported instruction type");
}

}  // namespace gpu
}  // namespace xla
