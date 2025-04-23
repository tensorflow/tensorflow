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

#include "xla/hlo/analysis/indexing_analysis.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
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

struct HLORTVar {
  Interval feasible_values;
  const HloInstruction* hlo;
  // This is a map from the iteration space of the corresponding indexing map to
  // the iteration space of `hlo`. It shows what element of `hlo` we need to
  // extract to get the runtime value for the RTVar.
  mlir::AffineMap map;
};

bool operator==(const HLORTVar& lhs, const HLORTVar& rhs) {
  return lhs.feasible_values == rhs.feasible_values && lhs.hlo == rhs.hlo &&
         lhs.map == rhs.map;
}

inline bool operator!=(const HLORTVar& lhs, const HLORTVar& rhs) {
  return !(lhs == rhs);
}

// The return type of `OptimizeRTVar` below
struct RTVarOptimizationResult {
  // An affine expr which maps the old RTVar to the new, optimized RTVar:
  // `()[sk] -> s'k` (with k being `symbol_index` in the `OptimizeRTVar` call).
  // If `expr` doesn't depend on `sk` it means the RTVar could be optimized
  // away completely and the value of `rt_var` can be ignored.
  AffineExpr remapped_symbol;

  // The new, optimized RTVar
  HLORTVar rt_var;
};

// Tries to optimize the given RTVar by removing some parts (or entirety) of
// the dependent HLO graph:
//
// 1. If no optimization is possible it returns `{sk, rt_var}` - the
// identity expr and the unchanged rt_var.
//
// 2. If full optimization is possible, it returns
// `{const, rt_var}` - an affine expr that does not anymore depend
// on `sk` and an arbitrary rt_var.
//
// 3. if partial optimization is possible, it returns
// `{()[sk] -> f(sk), rt_var_new }` - an affine expression that maps from the
// old RTVar to the new RTVar, and the new RTVar itself. The new RTVar now
// references some HLO subgraph of the old RTVar's HLO.
RTVarOptimizationResult OptimizeRTVar(HLORTVar rt_var, int64_t symbol_index,
                                      MLIRContext* mlir_context) {
  const auto symbol = getAffineSymbolExpr(symbol_index, mlir_context);
  auto result_expr = symbol;

  while (true) {
    if (auto constant_expr = DynCast<HloConstantInstruction>(rt_var.hlo)) {
      if (rt_var.map.isConstant()) {
        const auto idx = rt_var.map.getConstantResults();
        result_expr = result_expr.replace(
            symbol, getAffineConstantExpr(
                        constant_expr->literal().GetIntegralAsS64(idx).value(),
                        mlir_context));
      }
      return {result_expr, rt_var};
    }

    if (auto iota_expr = DynCast<HloIotaInstruction>(rt_var.hlo)) {
      auto iota_dimension = iota_expr->iota_dimension();
      CHECK(iota_dimension < rt_var.map.getNumResults());
      return {
          result_expr.replace(symbol, rt_var.map.getResults()[iota_dimension]),
          rt_var};
    }

    auto is_indexing_transformation = [](const HloInstruction* instr) {
      return instr->opcode() == HloOpcode::kBitcast ||
             instr->opcode() == HloOpcode::kBroadcast ||
             instr->opcode() == HloOpcode::kReshape ||
             instr->opcode() == HloOpcode::kReverse ||
             instr->opcode() == HloOpcode::kSlice ||
             instr->opcode() == HloOpcode::kTranspose;
    };

    if (is_indexing_transformation(rt_var.hlo)) {
      auto instr_indexing_map =
          *ComputeOutputToInputIndexing(rt_var.hlo, 0, mlir_context)
               .indexing_maps[0]
               .begin();

      rt_var.hlo = rt_var.hlo->operand(0);
      rt_var.map = instr_indexing_map.GetAffineMap().compose(rt_var.map);
      continue;
    }

    if (rt_var.hlo->opcode() == HloOpcode::kNegate) {
      rt_var.hlo = rt_var.hlo->operand(0);
      result_expr = result_expr.replace(symbol, -symbol);
      continue;
    }

    if (rt_var.hlo->opcode() == HloOpcode::kAdd ||
        rt_var.hlo->opcode() == HloOpcode::kSubtract ||
        rt_var.hlo->opcode() == HloOpcode::kMultiply ||
        rt_var.hlo->opcode() == HloOpcode::kDivide) {
      const auto apply_op = [&](const AffineExpr& lhs,
                                const AffineExpr& rhs) -> AffineExpr {
        switch (rt_var.hlo->opcode()) {
          case HloOpcode::kAdd:
            return lhs + rhs;
          case HloOpcode::kSubtract:
            return lhs - rhs;
          case HloOpcode::kMultiply:
            return lhs * rhs;
          case HloOpcode::kDivide:
            return lhs.floorDiv(rhs);
          default:
            ABSL_UNREACHABLE();
        }
      };

      auto lhs = OptimizeRTVar(
          HLORTVar{rt_var.feasible_values, rt_var.hlo->operand(0), rt_var.map},
          symbol_index, mlir_context);

      if (!lhs.remapped_symbol.isFunctionOfSymbol(symbol_index)) {
        // This means that lhs is constant-like and we can eliminate the
        // operand.
        result_expr =
            result_expr.replace(symbol, apply_op(lhs.remapped_symbol, symbol));

        // We continue optimizing the `rhs` operand
        rt_var.hlo = rt_var.hlo->operand(1);
        continue;
      }

      auto rhs = OptimizeRTVar(
          HLORTVar{rt_var.feasible_values, rt_var.hlo->operand(1), rt_var.map},
          symbol_index, mlir_context);

      if (!rhs.remapped_symbol.isFunctionOfSymbol(symbol_index)) {
        // This means that rhs is constant-like and we can eliminate the
        // operand.
        result_expr =
            result_expr.replace(symbol, apply_op(symbol, rhs.remapped_symbol));

        // We can also take advantage of the optimization already done for lhs:
        result_expr = result_expr.replace(symbol, lhs.remapped_symbol);
        rt_var = lhs.rt_var;
        continue;
      }
    }

    return {result_expr, rt_var};
  }
}

std::vector<IndexingMap::Variable> ConvertHLORTVarsToRTVars(
    const std::vector<HLORTVar>& hlo_rt_vars) {
  std::vector<IndexingMap::Variable> rt_vars;
  rt_vars.reserve(hlo_rt_vars.size());
  for (const HLORTVar& hlo_rt_var : hlo_rt_vars) {
    rt_vars.push_back(IndexingMap::Variable{hlo_rt_var.feasible_values});
  }
  return rt_vars;
}

IndexingMap FoldRTVarsAndConstructIndexingMap(
    AffineMap affine_map, std::vector<IndexingMap::Variable> dim_vars,
    std::vector<HLORTVar> hlo_rt_vars) {
  if (hlo_rt_vars.empty()) {
    return IndexingMap(affine_map, std::move(dim_vars), /*range_vars=*/{},
                       ConvertHLORTVarsToRTVars(hlo_rt_vars));
  }

  auto* ctx = affine_map.getContext();

  for (auto symbol_index = 0; symbol_index < hlo_rt_vars.size();
       ++symbol_index) {
    auto& rt_var = hlo_rt_vars[symbol_index];

    // range_vars and rt_vars share the symbol space, with the rt_vars coming
    // after the range_vars.
    auto rt_var_symbol = getAffineSymbolExpr(symbol_index, ctx);

    RTVarOptimizationResult result = OptimizeRTVar(rt_var, symbol_index, ctx);

    if (result.remapped_symbol != rt_var_symbol) {
      affine_map = affine_map.replace({{rt_var_symbol, result.remapped_symbol}},
                                      affine_map.getNumDims(),
                                      affine_map.getNumSymbols());

      llvm::DenseMap<AffineExpr, AffineExpr> replacements;
    }

    if (result.remapped_symbol.isFunctionOfSymbol(symbol_index)) {
      // If we still depend on the rt_var, then we update it.
      if (rt_var != result.rt_var) {
        rt_var = std::move(result.rt_var);
      }
    }
  }
  return IndexingMap(affine_map, std::move(dim_vars), /*range_vars=*/{},
                     ConvertHLORTVarsToRTVars(hlo_rt_vars));
}

HloInstructionIndexing ComputeOutputToInputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);
  IndexingMap unit_map(
      mlir::AffineMap::get(identity_map.GetAffineMap().getNumDims(),
                           /*symbolCount=*/0, mlir_context),
      identity_map.GetDimVars(), /*range_vars=*/{}, /*rt_vars=*/{});

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(instr->operand_count());
  int64_t operand_count = instr->operand_count();
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    // Select allows implicit broadcasting in the predicate. We just handle it
    // generically here.
    auto* operand = instr->operand(operand_id);
    if (operand->shape().dimensions().size() == 0 &&
        instr->shape().dimensions().size() > 0) {
      instr_indexing.indexing_maps[operand_id].insert(unit_map);
    } else {
      instr_indexing.indexing_maps[operand_id].insert(identity_map);
    }
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
  exprs.reserve(output_shape.dimensions().size());
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
      AffineMap::get(input_shape.dimensions().size(), added_dims_sizes.size(),
                     exprs, mlir_context),
      input_shape.dimensions(), added_dims_sizes);

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeOutputToInputConcatenateOpIndexing(
    const HloConcatenateInstruction* concat, MLIRContext* mlir_context) {
  const auto& operand_0_dims = concat->operand(0)->shape().dimensions();

  // Initialize affine map and domain. Only concat_dim elements of both have to
  // be adjusted for a particular operand_id.
  mlir::MutableAffineMap affine_map =
      AffineMap::getMultiDimIdentityMap(operand_0_dims.size(), mlir_context);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromTensorSizes(operand_0_dims);

  HloInstructionIndexing concat_indexing;
  concat_indexing.indexing_maps.resize(concat->operand_count());
  int64_t concat_dim = concat->concatenate_dimension();
  AffineExpr concat_dim_expr = getAffineDimExpr(concat_dim, mlir_context);
  int64_t offset = 0;
  for (const auto [operand_id, operand] : llvm::enumerate(concat->operands())) {
    affine_map.setResult(concat_dim, concat_dim_expr - offset);
    int64_t operand_concat_dim = operand->shape().dimensions()[concat_dim];
    dim_vars[concat_dim] =
        IndexingMap::Variable{{offset, offset + operand_concat_dim - 1}};
    concat_indexing.indexing_maps[operand_id].insert(
        IndexingMap(affine_map.getAffineMap(), dim_vars,
                    /*range_vars=*/{}, /*rt_vars=*/{}));
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
  SmallVector<AffineExpr> lhs_exprs(lhs_shape.dimensions().size());
  SmallVector<AffineExpr> rhs_exprs(rhs_shape.dimensions().size());
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
  auto lhs_non_contracting_dims = gpu::GetNonContractingDims(
      lhs_shape, lhs_batch_dims, lhs_contracting_dims);
  assert(lhs_non_contracting_dims.ok());

  for (int64_t lhs_non_contracting_dim : lhs_non_contracting_dims.value()) {
    lhs_exprs[lhs_non_contracting_dim] =
        getAffineDimExpr(output_dim_id++, mlir_context);
  }

  // rhs_non_contracting_dims
  auto rhs_non_contracting_dims = gpu::GetNonContractingDims(
      rhs_shape, rhs_batch_dims, rhs_contracting_dims);
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
      AffineMap::get(dot->shape().dimensions().size(), input_dim_sizes.size(),
                     lhs_exprs, mlir_context),
      dot->shape().dimensions(), input_dim_sizes);

  IndexingMap rhs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(dot->shape().dimensions().size(), input_dim_sizes.size(),
                     rhs_exprs, mlir_context),
      dot->shape().dimensions(), input_dim_sizes);
  return HloInstructionIndexing::FromIndexingMaps(
      {lhs_indexing_map, rhs_indexing_map});
}

HloInstructionIndexing ComputeOutputToInputDynamicSliceOpIndexing(
    const HloDynamicSliceInstruction* dynamic_slice,
    MLIRContext* mlir_context) {
  const Shape& input_shape = dynamic_slice->operand(0)->shape();
  const Shape& output_shape = dynamic_slice->shape();
  int64_t rank = output_shape.dimensions().size();
  const int64_t first_index_num = dynamic_slice->first_index_operand_number();

  CHECK(dynamic_slice->operand(first_index_num)->shape().dimensions().size() ==
        0)
      << "b/118437727: Old form, not supported.";
  // A map from tensor iteration space to (), because index operands are 0d
  // tensors.
  AffineMap empty_results_affine_map = AffineMap::get(
      /*dimCount=*/rank, /*symbolCount=*/0, /*results=*/{}, mlir_context);
  IndexingMap start_indices_map = IndexingMap::FromTensorSizes(
      empty_results_affine_map, output_shape.dimensions(), {});

  std::vector<HLORTVar> offsets_rt_vars;
  offsets_rt_vars.reserve(rank);
  std::vector<AffineExpr> exprs;
  exprs.reserve(rank);
  for (auto [dim, slice_size] :
       llvm::enumerate(dynamic_slice->dynamic_slice_sizes())) {
    exprs.push_back(getAffineDimExpr(dim, mlir_context) +
                    getAffineSymbolExpr(dim, mlir_context));
    offsets_rt_vars.push_back(
        HLORTVar{Interval{0, input_shape.dimensions(dim) - slice_size},
                 dynamic_slice->operand(dim + first_index_num),
                 empty_results_affine_map});
  }
  std::vector<IndexingMap> indexing_maps(dynamic_slice->operand_count(),
                                         start_indices_map);
  indexing_maps.front() = FoldRTVarsAndConstructIndexingMap(
      AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/rank, exprs,
                     mlir_context),
      start_indices_map.GetDimVars(), std::move(offsets_rt_vars));
  return HloInstructionIndexing::FromIndexingMaps(indexing_maps);
}

HloInstructionIndexing ComputeOutputToInputDynamicUpdateSliceOpIndexing(
    const HloDynamicUpdateSliceInstruction* dus, MLIRContext* mlir_context) {
  const Shape& update_shape = dus->update()->shape();
  const Shape& output_shape = dus->shape();
  int64_t rank = output_shape.dimensions().size();

  // operand: (d0, ... d_{N-1}) -> (d0, ... d_{N-1})
  std::vector<AffineExpr> identity;
  identity.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    identity.push_back(getAffineDimExpr(dim, mlir_context));
  }
  IndexingMap operand_map = IndexingMap::FromTensorSizes(
      AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, /*results=*/identity,
                     mlir_context),
      output_shape.dimensions(), {});

  // start_indices: (d0, ... d_{N-1}) -> ()
  AffineMap empty_results_affine_map = AffineMap::get(
      /*dimCount=*/rank, /*symbolCount=*/0, /*results=*/{}, mlir_context);
  IndexingMap start_indices_map = IndexingMap::FromTensorSizes(
      empty_results_affine_map, output_shape.dimensions(), {});

  // update: (d_0 - s_0, ..., d_{N-1} - s_{N-1})
  std::vector<AffineExpr> exprs;
  exprs.reserve(rank);
  std::vector<HLORTVar> rt_vars;
  rt_vars.reserve(rank);
  for (auto [dim, slice_size] : llvm::enumerate(update_shape.dimensions())) {
    exprs.push_back(getAffineDimExpr(dim, mlir_context) -
                    getAffineSymbolExpr(dim, mlir_context));
    Interval feasible_values{0, output_shape.dimensions(dim) - slice_size};
    rt_vars.push_back(HLORTVar{feasible_values, dus->operand(2 + dim),
                               empty_results_affine_map});
  }
  IndexingMap update_map = FoldRTVarsAndConstructIndexingMap(
      AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/rank,
                     /*results=*/exprs, mlir_context),
      operand_map.GetDimVars(), std::move(rt_vars));

  std::vector<IndexingMap> indexing_maps(dus->operand_count(),
                                         start_indices_map);
  indexing_maps[0] = std::move(operand_map);
  indexing_maps[1] = std::move(update_map);
  return HloInstructionIndexing::FromIndexingMaps(indexing_maps);
}

HloInstructionIndexing ComputeOutputToInputGatherOpIndexing(
    const HloGatherInstruction* gather, MLIRContext* mlir_context) {
  CHECK(GatherSimplifier::IsSimplifiedGather(gather))
      << "Non-simplified HLO Gather is not supported.";
  const Shape& operand_shape = gather->operand(0)->shape();
  const Shape& indices_shape = gather->operand(1)->shape();

  const GatherDimensionNumbers& dimension_numbers =
      gather->gather_dimension_numbers();
  int64_t index_vector_length =
      indices_shape.dimensions(dimension_numbers.index_vector_dim());

  const Shape& output_shape = gather->shape();
  int64_t output_rank = output_shape.dimensions().size();

  // A map for the `indices` operand of gather. It is always
  // (d_0, ... d_{rank - 1}) -> (d_0, s_0),
  // where 0 <= s_0 <= indices_shape[1] - 1.
  AffineExpr indices_id_dim = getAffineDimExpr(0, mlir_context);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromTensorSizes(output_shape.dimensions());
  IndexingMap indices_map{
      AffineMap::get(output_rank, 1,
                     {indices_id_dim, getAffineSymbolExpr(0, mlir_context)},
                     mlir_context),
      dim_vars,
      {IndexingMap::Variable{{0, index_vector_length - 1}}},
      /*rt_vars=*/{}};

  // A map for the `operand` operand of gather, from which we extract slices.
  // (d_0, ... d_{rank - 1}) -> (d_1 + s0, d_2 + s_1, ...),
  // where s_i are RTVars that extract indices from the `indices` operand.
  std::vector<HLORTVar> rt_vars;
  std::vector<AffineExpr> exprs;
  exprs.reserve(operand_shape.dimensions().size());
  for (auto [operand_dim_id, slice_size] :
       llvm::enumerate(gather->gather_slice_sizes())) {
    int64_t output_dim_id = dimension_numbers.offset_dims(operand_dim_id);
    exprs.push_back(getAffineDimExpr(output_dim_id, mlir_context));

    if (operand_dim_id >= index_vector_length) continue;

    rt_vars.push_back(HLORTVar{
        Interval{0, operand_shape.dimensions(operand_dim_id) - slice_size},
        gather->operand(1),
        AffineMap::get(output_rank, /*symbolCount=*/0,
                       {indices_id_dim,
                        getAffineConstantExpr(operand_dim_id, mlir_context)},
                       mlir_context)});
    exprs.back() =
        exprs.back() + getAffineSymbolExpr(operand_dim_id, mlir_context);
  }
  IndexingMap operand_map = FoldRTVarsAndConstructIndexingMap(
      AffineMap::get(/*dimCount=*/output_rank,
                     /*symbolCount=*/index_vector_length, exprs, mlir_context),
      std::move(dim_vars), std::move(rt_vars));
  return HloInstructionIndexing::FromIndexingMaps({operand_map, indices_map});
}

IndexingMap ComputeOutputToInputPadOpIndexingImpl(
    absl::Span<const int64_t> output_dims,
    absl::Span<const int64_t> padding_low,
    absl::Span<const int64_t> padding_high,
    absl::Span<const int64_t> padding_interior, MLIRContext* mlir_context) {
  int64_t output_rank = output_dims.size();

  std::vector<AffineExpr> exprs;
  std::vector<std::pair<AffineExpr, Interval>> constraints;
  std::vector<IndexingMap::Variable> dim_vars;
  exprs.reserve(output_rank);
  constraints.reserve(output_rank);
  int64_t output_dim_id = 0;
  for (const auto [output_dim, pad_low, pad_high, pad_interior] :
       llvm::zip(output_dims, padding_low, padding_high, padding_interior)) {
    AffineExpr dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
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
  return IndexingMap{
      AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      std::move(dim_vars),
      /*range_vars = */ {},
      /*rt_vars = */ {}, absl::MakeSpan(constraints)};
}

HloInstructionIndexing ComputeOutputToInputPadOpIndexing(
    const HloPadInstruction* pad, MLIRContext* mlir_context) {
  const Shape& output_shape = pad->shape();
  int64_t rank = output_shape.dimensions().size();
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
      AffineMap::get(output_shape.dimensions().size(), /*symbolCount=*/0, {},
                     mlir_context),
      output_shape.dimensions(), /*symbol_upper_bounds=*/{});
  return HloInstructionIndexing::FromIndexingMaps(
      {input_indexing_map, padding_value_indexing_map});
}

HloInstructionIndexing ComputeOutputToInputReduceOpIndexing(
    const HloReduceInstruction* reduce, MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());

  const Shape& input_shape = reduce->operand(0)->shape();
  const Shape& output_shape = GetOutputShape(reduce, 0);

  std::vector<int64_t> parallel_dims_sizes;
  int64_t output_dim_id = 0;
  std::vector<AffineExpr> exprs;
  exprs.reserve(input_shape.dimensions().size());
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
      AffineMap::get(output_shape.dimensions().size(), reduce_dims_ids.size(),
                     exprs, mlir_context),
      output_shape.dimensions(), parallel_dims_sizes);
  IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.dimensions().size(), /*symbolCount=*/0, {},
                     mlir_context),
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
  const Shape& output_shape = GetOutputShape(reduce, 0);
  int64_t output_rank = output_shape.dimensions().size();

  HloInstructionIndexing instr_indexing;
  int arity = reduce->input_count();
  instr_indexing.indexing_maps.resize(arity);
  if (input_id >= arity) {
    // This is an init value: it contributes to every output element.
    std::vector<AffineExpr> inits_exprs;
    inits_exprs.reserve(output_rank);
    for (int sym = 0; sym < output_rank; ++sym) {
      inits_exprs.push_back(getAffineSymbolExpr(sym, mlir_context));
    }
    IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
        AffineMap::get(0, /*symbolCount=*/output_rank, inits_exprs,
                       mlir_context),
        {}, output_shape.dimensions());
    for (int64_t id = 0; id < arity; ++id) {
      instr_indexing.indexing_maps[id].insert(inits_indexing_map);
    }
    return instr_indexing;
  }

  // This is a reduced value: it contributes to all output elements at the
  // input element's indices with the reduced dimensions removed.
  const Shape& input_shape = reduce->operand(input_id)->shape();
  std::vector<AffineExpr> inputs_exprs;
  inputs_exprs.reserve(output_rank);
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (!absl::c_linear_search(reduce->dimensions(), input_dim_id)) {
      inputs_exprs.push_back(getAffineDimExpr(input_dim_id, mlir_context));
    }
  }
  IndexingMap inputs_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(input_shape.dimensions().size(), /*symbolCount=*/0,
                     inputs_exprs, mlir_context),
      input_shape.dimensions(), {});
  for (int64_t id = 0; id < arity; ++id) {
    instr_indexing.indexing_maps[id].insert(inputs_indexing_map);
  }
  return instr_indexing;
}

IndexingMap ComposeIndexingMapsForWindow(
    absl::Span<const int64_t> input_dimensions,
    absl::Span<const int64_t> output_dimensions, const Window& window,
    MLIRContext* mlir_context) {
  size_t rank = input_dimensions.size();

  // Compute shape of the padded input and the indexing map of pad op required
  // to pad the input.
  SmallVector<int64_t> padding_low, padding_high, padding_interior,
      padded_input_dimensions;
  padding_low.reserve(rank);
  padding_high.reserve(rank);
  padding_interior.reserve(rank);
  padded_input_dimensions.reserve(rank);
  SmallVector<AffineExpr, 4> exprs;
  std::vector<IndexingMap::Variable> dim_vars;
  std::vector<IndexingMap::Variable> range_vars;
  exprs.reserve(rank);
  dim_vars.reserve(rank);
  range_vars.reserve(rank);
  for (const auto& [dim_id, window_config] :
       llvm::enumerate(window.dimensions())) {
    padding_low.push_back(window_config.padding_low());
    padding_high.push_back(window_config.padding_high());
    // For some reason interior_padding in HLO pad is offset from base_dilations
    // in HLO reduce-window by 1.
    padding_interior.push_back(window_config.base_dilation() - 1);
    padded_input_dimensions.push_back(
        input_dimensions[dim_id] + window_config.padding_low() +
        window_config.padding_high() +
        (input_dimensions[dim_id] - 1) * (window_config.base_dilation() - 1));
    AffineExpr dim_expr = getAffineDimExpr(dim_id, mlir_context);
    AffineExpr symbol_expr = getAffineSymbolExpr(dim_id, mlir_context);

    exprs.push_back(symbol_expr * window_config.window_dilation() +
                    window_config.stride() * dim_expr);
    dim_vars.push_back(
        {IndexingMap::Variable{0, output_dimensions[dim_id] - 1}});
    range_vars.push_back({IndexingMap::Variable{0, window_config.size() - 1}});
  }
  // Indexing map for pad op that pads the input.
  IndexingMap padded_input_indexing = ComputeOutputToInputPadOpIndexingImpl(
      padded_input_dimensions, padding_low, padding_high, padding_interior,
      mlir_context);
  // Indexing map for reduce-window, that does not do any padding.
  IndexingMap input_indexing_no_padding(
      AffineMap::get(rank, rank, exprs, mlir_context), dim_vars, range_vars,
      /*rt_vars=*/{});

  // Composed indexing.
  IndexingMap result =
      ComposeIndexingMaps(input_indexing_no_padding, padded_input_indexing);
  result.Simplify();
  result.RemoveUnusedSymbols();
  return result;
}

// Indexing for reduce-window with dilations and non-trivial padding can be
// represented as a composition of pad op and reduce-window that never goes out
// of bounds.
HloInstructionIndexing ComputeOutputToInputReduceWindowOpIndexing(
    const HloReduceWindowInstruction* reduce_window,
    MLIRContext* mlir_context) {
  const Shape& input_shape = reduce_window->operand(0)->shape();
  const Shape& output_shape = GetOutputShape(reduce_window, 0);

  // Indexing map for the input value.
  IndexingMap inputs_indexing = ComposeIndexingMapsForWindow(
      input_shape.dimensions(), output_shape.dimensions(),
      reduce_window->window(), mlir_context);

  // Indexing map for the init value.
  IndexingMap inits_indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_shape.dimensions().size(), /*symbolCount=*/0, {},
                     mlir_context),
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

HloInstructionIndexing ComputeOutputToInputConvolutionOpIndexing(
    const HloConvolutionInstruction* convolution, MLIRContext* mlir_context) {
  const Shape& input_shape = convolution->operand(0)->shape();
  const Shape& kernel_shape = convolution->operand(1)->shape();
  const Shape& output_shape = convolution->shape();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();
  size_t rank = output_shape.dimensions().size();

  // Collect sizes for input/output spatial dimensions.
  size_t spatial_rank = rank - 2;
  std::vector<int64_t> input_spatial_sizes(spatial_rank);
  std::vector<int64_t> kernel_spatial_sizes(spatial_rank);
  std::vector<int64_t> output_spatial_sizes(spatial_rank);
  for (int i = 0; i < spatial_rank; ++i) {
    input_spatial_sizes[i] =
        input_shape.dimensions(dnums.input_spatial_dimensions(i));
    kernel_spatial_sizes[i] =
        kernel_shape.dimensions(dnums.kernel_spatial_dimensions(i));
    output_spatial_sizes[i] =
        output_shape.dimensions(dnums.output_spatial_dimensions(i));
  }

  // Indexing map for the input value (spatial dimensions only).
  // The dimension numbers in the resulting affine expressions have to be
  // remapped to correspond to the correct output dimensions.
  IndexingMap input_spatial_indexing =
      ComposeIndexingMapsForWindow(input_spatial_sizes, output_spatial_sizes,
                                   convolution->window(), mlir_context);
  std::vector<AffineExpr> replacement_dims(spatial_rank);
  for (int i = 0; i < spatial_rank; ++i) {
    replacement_dims[i] =
        getAffineDimExpr(dnums.output_spatial_dimensions(i), mlir_context);
  }

  // Build affine expressions and constraints for input spatial dimensions.
  std::vector<AffineExpr> input_exprs(rank);
  for (int i = 0; i < spatial_rank; ++i) {
    input_exprs[dnums.input_spatial_dimensions(i)] =
        input_spatial_indexing.GetAffineMap().getResult(i).replaceDims(
            replacement_dims);
  }
  llvm::DenseMap<AffineExpr, Interval> input_constraints;
  for (const auto& [key, val] : input_spatial_indexing.GetConstraints()) {
    input_constraints[key.replaceDims(replacement_dims)] = val;
  }

  // Build affine expressions for kernel spatial and output dimensions.
  std::vector<AffineExpr> kernel_exprs(rank);
  for (int i = 0; i < spatial_rank; ++i) {
    kernel_exprs[dnums.kernel_spatial_dimensions(i)] =
        getAffineSymbolExpr(i, mlir_context);
  }
  AffineExpr dim_expr =
      getAffineDimExpr(dnums.output_feature_dimension(), mlir_context);
  kernel_exprs[dnums.kernel_output_feature_dimension()] = dim_expr;

  // Build initial symbol ranges.
  std::vector<IndexingMap::Variable> input_symbols =
      input_spatial_indexing.GetRangeVars();
  std::vector<IndexingMap::Variable> kernel_symbols =
      RangeVarsFromTensorSizes(kernel_spatial_sizes);

  // Add symbol for input feature dimension.
  input_exprs[dnums.input_feature_dimension()] =
      getAffineSymbolExpr(input_symbols.size(), mlir_context);
  kernel_exprs[dnums.kernel_input_feature_dimension()] =
      getAffineSymbolExpr(kernel_symbols.size(), mlir_context);

  int64_t input_group_size =
      kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  Interval input_feature_range{0, input_group_size - 1};
  input_symbols.push_back(IndexingMap::Variable{input_feature_range});
  kernel_symbols.push_back(IndexingMap::Variable{input_feature_range});

  // With multiple feature groups, the input feature dimension is equally split.
  if (convolution->feature_group_count() > 1) {
    AffineExpr& input_feature = input_exprs[dnums.input_feature_dimension()];
    int64_t output_group_size =
        output_shape.dimensions(dnums.output_feature_dimension());
    int64_t feature_group_size =
        output_group_size / convolution->feature_group_count();
    input_feature = dim_expr.floorDiv(feature_group_size) * input_group_size +
                    input_feature;
  }

  // With multiple batch groups, the input batch dimension is equally split.
  AffineExpr batch_dim_expr =
      getAffineDimExpr(dnums.output_batch_dimension(), mlir_context);
  if (convolution->batch_group_count() > 1) {
    int64_t batch_group_size =
        output_shape.dimensions(dnums.output_batch_dimension());
    AffineExpr batch_group_expr =
        getAffineSymbolExpr(input_symbols.size(), mlir_context);
    input_symbols.push_back(
        IndexingMap::Variable{{0, convolution->batch_group_count() - 1}});
    input_exprs[dnums.input_batch_dimension()] =
        batch_group_expr * batch_group_size + batch_dim_expr;
  } else {
    input_exprs[dnums.input_batch_dimension()] = batch_dim_expr;
  }

  // Indexing map for the input value.
  IndexingMap inputs_indexing(
      AffineMap::get(rank, input_symbols.size(), input_exprs, mlir_context),
      DimVarsFromTensorSizes(output_shape.dimensions()), input_symbols,
      /*rt_vars=*/{}, input_constraints);

  // Indexing map for the kernel value.
  IndexingMap kernel_indexing(
      AffineMap::get(rank, kernel_symbols.size(), kernel_exprs, mlir_context),
      DimVarsFromTensorSizes(output_shape.dimensions()), kernel_symbols,
      /*rt_vars=*/{});

  return HloInstructionIndexing::FromIndexingMaps(
      {inputs_indexing, kernel_indexing});
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

}  // namespace

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

namespace {

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
  exprs.reserve(input.dimensions().size());

  // If the input shape has no elements (e.g. 1000x10x0 -> 100x100x0), just set
  // everything to 0.
  if (ShapeUtil::ElementsIn(input) == 0) {
    for (int i = 0; i < input.dimensions().size(); ++i) {
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
  while (input_dim_id < input.dimensions().size() ||
         output_dim_id < output.dimensions().size() ||
         !input_subshape.empty()) {
    if (input_dim_id < input.dimensions().size() &&
        (input_subshape.empty() || input_num_elements < output_num_elements ||
         input_dims[input_dim_id] == 1)) {
      input_num_elements *= input_dims[input_dim_id];
      input_subshape.push_back(input_dims[input_dim_id]);
      ++input_dim_id;
      continue;
    }
    if (output_dim_id < output.dimensions().size() &&
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
  auto output_rank = slice->shape().dimensions().size();

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

HloInstructionIndexing ComputeInputToOutputSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  auto output_rank = slice->shape().dimensions().size();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);
  for (int64_t dim = 0; dim < output_rank; ++dim) {
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);
    exprs.push_back((dim_expr - slice->slice_starts()[dim])
                        .floorDiv(slice->slice_strides()[dim]));
  }
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      slice->operand(0)->shape().dimensions(), {});

  for (int64_t dim = 0; dim < output_rank; ++dim) {
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);
    int64_t lb = slice->slice_starts()[dim];
    int64_t ub =
        (slice->shape().dimensions(dim) - 1) * slice->slice_strides()[dim] +
        slice->slice_starts()[dim];
    indexing_map.AddConstraint(dim_expr, {lb, ub});
    indexing_map.AddConstraint((dim_expr - lb) % slice->slice_strides()[dim],
                               {0, 0});
  }

  return HloInstructionIndexing::FromIndexingMaps({std::move(indexing_map)});
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

IndexingMap GetBitcastMap(absl::Span<const int64_t> input_shape,
                          const Shape& output_shape,
                          mlir::MLIRContext* mlir_context) {
  return GetBitcastMap(ShapeUtil::MakeShapeWithDescendingLayout(
                           output_shape.element_type(), input_shape),
                       output_shape, mlir_context);
}
IndexingMap GetBitcastMap(absl::Span<const int64_t> input_shape,
                          absl::Span<const int64_t> output_shape,
                          mlir::MLIRContext* mlir_context) {
  return GetBitcastMap(
      ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::S8, input_shape),
      ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::S8, output_shape),
      mlir_context);
}
IndexingMap GetBitcastMap(const Shape& input_shape, const Shape& output_shape,
                          MLIRContext* mlir_context) {
  ShapeUtil::BitcastDecomposition decomposed_bitcast =
      ShapeUtil::DecomposeBitcast(input_shape, output_shape);

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          decomposed_bitcast)) {
    auto permutation = ShapeUtil::DeduceTransposeDimensionsForBitcast(
        input_shape, output_shape);
    CHECK(permutation.has_value())
        << "Failed to deduce permutation for a bitcast.";

    return IndexingMap::FromTensorSizes(
        ComputeTransposeIndexingMap(permutation.value(), mlir_context),
        input_shape.dimensions(), {});
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposed_bitcast)) {
    // Note: ComputeReshapeIndexingMap assumes it's computing an output->input
    // indexing, so input and output are reversed.
    return IndexingMap::FromTensorSizes(
        ComputeReshapeIndexingMap(output_shape, input_shape, mlir_context),
        input_shape.dimensions(), {});
  }
  // `trt` stands for transpose-reshape-transpose decomposition of bitcast.
  auto trt = std::get<ShapeUtil::BitcastDecompositionTrt>(decomposed_bitcast);
  auto transpose_map_1 =
      ComputeTransposeIndexingMap(trt.transpose1_dims, mlir_context);
  auto reshape_map = ComputeReshapeIndexingMap(
      trt.reshape_shape, trt.transpose1_shape, mlir_context);
  auto transpose_map_2 =
      ComputeTransposeIndexingMap(trt.transpose2_dims, mlir_context);
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

}  // namespace

IndexingMap CreateIdentityMap(absl::Span<const int64_t> dimensions,
                              mlir::MLIRContext* mlir_context) {
  return IndexingMap::FromTensorSizes(
      AffineMap::getMultiDimIdentityMap(dimensions.size(), mlir_context),
      /*dim_upper_bounds=*/dimensions, /*symbol_upper_bounds=*/{});
}

IndexingMap CreateIdentityMap(const Shape& shape, MLIRContext* mlir_context) {
  if (shape.IsTuple()) {
    // Should happen only for variadic reduce. In that case all tuple shapes are
    // equal.
    return CreateIdentityMap(shape.tuple_shapes(0), mlir_context);
  }
  return CreateIdentityMap(shape.dimensions(), mlir_context);
}

llvm::SmallVector<AffineExpr, 4> DelinearizeInBoundsIndex(
    AffineExpr linear, absl::Span<const int64_t> sizes) {
  llvm::SmallVector<AffineExpr, 4> result;
  result.reserve(sizes.size());
  if (absl::c_linear_search(sizes, 0)) {
    for (int dim = 0; dim < sizes.size(); ++dim) {
      result.push_back(mlir::getAffineConstantExpr(0, linear.getContext()));
    }
    return result;
  }

  auto strides = ComputeStrides(sizes);
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

IndexingMap GetIndexingMapFromPhysicalLayoutToLogical(
    const Shape& shape, MLIRContext* mlir_context) {
  if (shape.dimensions().size() == 0) {
    return IndexingMap(AffineMap::get(mlir_context),
                       /*dimensions=*/{}, /*range vars=*/{}, /*rt_vars=*/{});
  }
  return IndexingMap::FromTensorSizes(
      ComputeTransposeIndexingMap(
          InversePermutation(ToTransposeDimensions(shape.layout())),
          mlir_context),
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape)
          .dimensions(),
      {});
}

IndexingMap GetIndexingMapFromLogicalToPhysicalLayout(
    const Shape& shape, MLIRContext* mlir_context) {
  if (shape.dimensions().size() == 0) {
    return IndexingMap(AffineMap::get(mlir_context),
                       /*dimensions=*/{}, /*range vars=*/{}, /*rt_vars=*/{});
  }
  return IndexingMap::FromTensorSizes(
      ComputeTransposeIndexingMap(ToTransposeDimensions(shape.layout()),
                                  mlir_context),
      shape.dimensions(), {});
}

bool HloInstructionIndexing::Simplify() {
  bool any_simplified = false;
  for (auto& operand_indexing : indexing_maps) {
    std::vector<IndexingMap> to_remove, to_add;
    for (IndexingMap map : operand_indexing) {
      if (map.Simplify()) {
        map.RemoveUnusedSymbols();
        to_add.push_back(map);
      }
      if (map.IsUndefined()) {
        to_remove.push_back(map);
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

std::string HloInstructionIndexing::ToString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  for (const auto& [operand_id, indexing_maps] :
       llvm::enumerate(instr_indexing.indexing_maps)) {
    out << "operand id = " << operand_id << ' ';
    for (const auto& indexing_map : indexing_maps) {
      if (indexing_map.IsUndefined()) {
        out << "unknown indexing";
        continue;
      }
      out << indexing_map;
    }
  }
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
      auto fusion_instr = parameter_instr->parent()->FusionInstruction();
      auto fusion_operand =
          fusion_instr->operand(parameter_instr->parameter_number());
      grouped_indexing_maps[fusion_operand] = {initial_map};
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
  if (HloInstruction::IsOpElementwise(instr->opcode()) ||
      instr->opcode() == HloOpcode::kMap) {
    // Note: map has a `dimensions` attribute, but it does nothing. See
    // b/65689298.
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
  if (auto dynamic_slice = DynCast<HloDynamicSliceInstruction>(instr)) {
    return ComputeOutputToInputDynamicSliceOpIndexing(dynamic_slice, ctx);
  }
  if (auto dus = DynCast<HloDynamicUpdateSliceInstruction>(instr)) {
    return ComputeOutputToInputDynamicUpdateSliceOpIndexing(dus, ctx);
  }
  if (auto fusion = DynCast<HloFusionInstruction>(instr)) {
    return ComputeOutputToInputFusionOpIndexing(fusion, output_id, ctx);
  }
  if (auto gather = DynCast<HloGatherInstruction>(instr)) {
    return ComputeOutputToInputGatherOpIndexing(gather, ctx);
  }
  if (auto iota = DynCast<HloIotaInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto pad = DynCast<HloPadInstruction>(instr)) {
    return ComputeOutputToInputPadOpIndexing(pad, ctx);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeOutputToInputReduceOpIndexing(reduce, ctx);
  }
  if (auto reduce_window = DynCast<HloReduceWindowInstruction>(instr)) {
    return ComputeOutputToInputReduceWindowOpIndexing(reduce_window, ctx);
  }
  if (auto convolution = DynCast<HloConvolutionInstruction>(instr)) {
    return ComputeOutputToInputConvolutionOpIndexing(convolution, ctx);
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
  LOG(ERROR) << "ComputeOutputToInputIndexing is not implemented for opcode "
             << instr->opcode();
  // If we cannot compute output-to-input indexing, we return std::nullopt for
  // every op parameter.
  return CreateUnknownIndexing(instr->operand_count());
}

HloInstructionIndexing ComputeInputToOutputIndexing(const HloInstruction* instr,
                                                    int input_id,
                                                    MLIRContext* ctx) {
  if (HloInstruction::IsOpElementwise(instr->opcode()) ||
      instr->opcode() == HloOpcode::kMap) {
    // Note: map has a `dimensions` attribute, but it does nothing. See
    // b/65689298.
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
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeInputToOutputSliceOpIndexing(slice, ctx);
  }
  if (instr->opcode() == HloOpcode::kTuple) {
    return HloInstructionIndexing::FromIndexingMaps(
        {CreateIdentityMap(instr->shape().tuple_shapes(input_id), ctx)});
  }
  // If we cannot compute input-to-output indexing, we return std::nullopt for
  // every op result.
  int64_t num_results =
      instr->shape().IsTuple() ? instr->shape().tuple_shapes().size() : 1;
  return CreateUnknownIndexing(num_results);
}

IndexingMap ComputeEpilogueInputToOutputIndexing(
    HloInstructionAdaptor epilogue_parent, HloInstructionAdaptor epilogue_root,
    MLIRContext* mlir_context) {
  auto chain = HloFindUseChain(epilogue_parent, epilogue_root);
  CHECK(!chain.empty()) << "There is no use chain from parent to root";
  auto root_indexing = CreateIdentityMap(epilogue_parent.shape(), mlir_context);
  for (int i = 1; i < chain.size(); ++i) {
    const auto& producer = chain[i - 1].instruction();
    const auto& user = chain[i].instruction();
    auto user_indexing = ComputeInputToOutputIndexing(
        &user, user.operand_index(&producer), mlir_context);
    root_indexing = root_indexing * *user_indexing.indexing_maps[0].begin();
    root_indexing.Simplify();
    root_indexing.RemoveUnusedSymbols();
  }
  return root_indexing;
}

}  // namespace xla
