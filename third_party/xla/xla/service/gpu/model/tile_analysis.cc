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

#include <cstdint>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

StatusOr<HloInstructionIndexing> ComputeCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  auto dims = instr->shape().dimensions();
  IndexingMap identity_map{.affine_map = AffineMap::getMultiDimIdentityMap(
                               dims.size(), mlir_context),
                           .input_dims_sizes = {}};

  std::vector<HloOperandIndexing> operand_indexing_maps;
  int64_t operand_count = instr->operand_count();
  operand_indexing_maps.reserve(operand_count);
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    operand_indexing_maps.push_back({{identity_map}, operand_id});
  }
  return HloInstructionIndexing{std::move(operand_indexing_maps)};
}

StatusOr<HloInstructionIndexing> ComputeBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  auto output_dims = bcast->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (int64_t bcast_dim : bcast->dimensions()) {
    exprs.push_back(getAffineDimExpr(bcast_dim, mlir_context));
  }
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .input_dims_sizes = {}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
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
  int64_t symbol_id = 0;
  for (int64_t dim : llvm::concat<const int64_t>(
           producer_map.input_dims_sizes, consumer_map.input_dims_sizes)) {
    if (unused_symbols_bit_vector[symbol_id++]) continue;
    combined_sizes.push_back(dim);
  }
  return IndexingMap{.affine_map = std::move(composed_map),
                     .input_dims_sizes = std::move(combined_sizes)};
}

// Computes HloInstructionIndexing that maps the iteration space of the
// consumer's output tensor to the iteration space of the producer's inputs and
// the remaining outputs of the consumer as if the producer was fused.
//
// Example:
//
//  operand1 operand2
//     |        |       # producer_instr_indexing edges
//  producer_instr
//      |               # consumer_operand_indexing edge
//  consumer
//
// The function has two inputs:
//
// 1. `producer_instr_indexing` is the producer's HloInstructionIndexing
//    that maps the iteration space of its output tensor to the inputs of
//    producers.
// 2. `consumer_operand_indexing` is the consumer's HloOperandIndexing for the
//    operand that corresponds to the provided producer.
HloInstructionIndexing ComputeFusedProducerConsumerIndexing(
    const HloInstructionIndexing& producer_instr_indexing,
    const HloOperandIndexing& consumer_operand_indexing) {
  HloInstructionIndexing fused_instr_indexing;

  // Every operand can be read 1 or more times by the consumer which also can
  // have 1 or more read accesses to its operands. So, to get the composed
  // indexing maps we have to compute a "cross product" here.
  for (const HloOperandIndexing& producer_operand_indexing :
       producer_instr_indexing.operand_indexing_maps) {
    auto& composed_operand_indexing =
        fused_instr_indexing.operand_indexing_maps.emplace_back();
    composed_operand_indexing.operand_id = producer_operand_indexing.operand_id;
    for (const IndexingMap& producer_map :
         producer_operand_indexing.indexing_maps) {
      for (const IndexingMap& consumer_map :
           consumer_operand_indexing.indexing_maps) {
        composed_operand_indexing.indexing_maps.insert(
            ComposeIndexingMaps(producer_map, consumer_map));
      }
    }
    fused_instr_indexing.operand_indexing_maps.push_back(
        std::move(composed_operand_indexing));
  }
  return fused_instr_indexing;
}

// Composes instruction indexing maps starting at the root instruction
// until the HloParameterInstruction is found.
StatusOr<HloInstructionIndexing> ComputeFusionOpIndexing(
    const HloFusionInstruction* fusion, int output_id,
    MLIRContext* mlir_context) {
  const HloInstruction* root =
      fusion->shape().IsTuple()
          ? fusion->fused_expression_root()->operand(output_id)
          : fusion->fused_expression_root();
  std::queue<std::pair<const HloInstruction*, HloInstructionIndexing>> bfs;
  TF_ASSIGN_OR_RETURN(auto root_indexing, ComputeInstructionIndexing(
                                              root, output_id, mlir_context));

  bfs.push(std::make_pair(root, root_indexing));
  absl::flat_hash_map<int64_t, absl::flat_hash_set<IndexingMap>>
      parameter_indexing_maps;
  while (!bfs.empty()) {
    const auto& [instr, instr_indexing] = bfs.front();
    for (const auto& operand_indexing : instr_indexing.operand_indexing_maps) {
      const HloInstruction* producer_instr =
          instr->operand(operand_indexing.operand_id);
      // If the producer is a fusion op parameter, store the result.
      if (auto parameter = DynCast<HloParameterInstruction>(producer_instr)) {
        parameter_indexing_maps[parameter->parameter_number()].insert(
            operand_indexing.indexing_maps.begin(),
            operand_indexing.indexing_maps.end());
        continue;
      }
      TF_ASSIGN_OR_RETURN(auto producer_instr_indexing,
                          ComputeInstructionIndexing(
                              producer_instr, /*output_id=*/0, mlir_context));
      bfs.push(std::make_pair(producer_instr,
                              ComputeFusedProducerConsumerIndexing(
                                  producer_instr_indexing, operand_indexing)));
    }
    bfs.pop();
  }
  HloInstructionIndexing fusion_indexing;
  for (const auto& [operand_id, maps] : parameter_indexing_maps) {
    fusion_indexing.operand_indexing_maps.push_back({maps, operand_id});
  }
  return fusion_indexing;
}

StatusOr<HloInstructionIndexing> ComputeDotOpIndexing(
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

  return HloInstructionIndexing{
      {HloOperandIndexing{.indexing_maps = {std::move(lhs_indexing_map)},
                          .operand_id = 0},
       HloOperandIndexing{.indexing_maps = {std::move(rhs_indexing_map)},
                          .operand_id = 1}}};
}

StatusOr<HloInstructionIndexing> ComputeReduceOpIndexing(
    const HloReduceInstruction* reduce, int output_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());

  const Shape& input_shape = reduce->operand(output_id)->shape();
  const Shape& output_shape = reduce->shape().IsTuple()
                                  ? ShapeUtil::GetSubshape(reduce->shape(), {0})
                                  : reduce->shape();

  std::vector<int64_t> input_dims_sizes;
  int64_t reduced_dim_id = 0;
  int64_t output_dim_id = 0;
  std::vector<AffineExpr> exprs;
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (reduce_dims_ids.contains(input_dim_id)) {
      exprs.push_back(getAffineSymbolExpr(reduced_dim_id++, mlir_context));
      input_dims_sizes.push_back(input_dim);
      continue;
    }
    exprs.push_back(getAffineDimExpr(output_dim_id++, mlir_context));
  }
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_shape.rank(), reduce_dims_ids.size(),
                                   exprs, mlir_context),
      .input_dims_sizes = std::move(input_dims_sizes)};

  std::vector<HloOperandIndexing> operand_indexing_maps;
  int64_t input_count = reduce->input_count();
  operand_indexing_maps.reserve(input_count);
  for (int64_t input_id = 0; input_id < input_count; ++input_id) {
    operand_indexing_maps.push_back({{indexing_map}, input_id});
  }
  return HloInstructionIndexing{std::move(operand_indexing_maps)};
}

StatusOr<HloInstructionIndexing> ComputeReverseOpIndexing(
    const HloReverseInstruction* reverse, MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reverse_dims(reverse->dimensions().begin(),
                                            reverse->dimensions().end());
  auto output_dims = reverse->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (auto [output_dim_id, output_dim] : llvm::enumerate(output_dims)) {
    auto dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
    if (!reverse_dims.contains(output_dim_id)) {
      exprs.push_back(dim_expr);
      continue;
    }
    auto dim_size = getAffineConstantExpr(output_dim, mlir_context);
    auto neg_dim_expr = getAffineBinaryOpExpr(
        AffineExprKind::Mul, getAffineConstantExpr(-1, mlir_context), dim_expr);
    exprs.push_back(
        getAffineBinaryOpExpr(AffineExprKind::Add, neg_dim_expr, dim_size));
  }

  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .input_dims_sizes = {}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
}

StatusOr<HloInstructionIndexing> ComputeSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  auto output_dims = slice->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (int64_t dim = 0; dim < output_dims.size(); ++dim) {
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
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .input_dims_sizes = {}};
  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
}

StatusOr<HloInstructionIndexing> ComputeTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  std::vector<unsigned> permutation(transpose->dimensions().begin(),
                                    transpose->dimensions().end());
  IndexingMap permutation_map{
      .affine_map = mlir::inversePermutation(
          AffineMap::getPermutationMap(permutation, mlir_context)),
      .input_dims_sizes = {}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(permutation_map)}, .operand_id = 0}}};
}

template <typename T>
std::string ToStringImpl(const T& value) {
  std::string s;
  std::stringstream ss(s);
  ss << value;
  return ss.str();
}

}  // namespace

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
                         const HloOperandIndexing& operand_indexing) {
  out << "operand id = " << operand_indexing.operand_id << ' ';
  for (const auto& map : operand_indexing.indexing_maps) {
    out << map;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  for (const auto& operand_map : instr_indexing.operand_indexing_maps) {
    out << operand_map;
  }
  return out;
}

std::string IndexingMap::ToString() const { return ToStringImpl(*this); }

std::string HloOperandIndexing::ToString() const { return ToStringImpl(*this); }

std::string HloInstructionIndexing::ToString() const {
  return ToStringImpl(*this);
}

StatusOr<HloInstructionIndexing> ComputeInstructionIndexing(
    const HloInstruction* instr, int output_id, MLIRContext* mlir_context) {
  if (HloInstruction::IsOpElementwise(instr->opcode())) {
    return ComputeCwiseOpIndexing(instr, mlir_context);
  }
  if (auto bcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeBroadcastOpIndexing(bcast, mlir_context);
  }
  if (auto dot = DynCast<HloDotInstruction>(instr)) {
    return ComputeDotOpIndexing(dot, mlir_context);
  }
  if (auto fusion = DynCast<HloFusionInstruction>(instr)) {
    return ComputeFusionOpIndexing(fusion, output_id, mlir_context);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeReduceOpIndexing(reduce, output_id, mlir_context);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, mlir_context);
  }
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeSliceOpIndexing(slice, mlir_context);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeTransposeOpIndexing(transpose, mlir_context);
  }
  return InvalidArgument("Unsupported instruction type");
}

}  // namespace gpu
}  // namespace xla
