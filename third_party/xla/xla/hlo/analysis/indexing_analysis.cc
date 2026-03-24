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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/analysis/indexing_analysis_utils.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/matmul_indexing_utils.h"
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
  return HloInstructionIndexing::FromIndexingMaps(
      std::vector<IndexingMap>(count, IndexingMap::GetUndefined()));
}

// HLORTVar represents the origin operation and possible values for a runtime
// variable in an indexing map.
//
// For example, in HLO
//
// data = f32[100] ...
// idx = s32[10,1] ...
// ROOT result = gather(data, idx), <...>
//
// in the indexing map of `data`: `(d0, d1){rt0} -> (d1 + rt0)` we have `rt0`
// with HLORTVar:
// - `feasible_values` in [0, 99],
// - `hlo` pointing to `idx`,
// - `map` of `(d0, d1) -> (d0, 0)` from the output of `gather` (not `data`)
//   into `idx`.
struct HLORTVar {
  Interval feasible_values;
  InstructionRef hlo;
  mlir::AffineMap map;
  DimensionVector dim_upper_bounds;
};

bool operator==(const HLORTVar& lhs, const HLORTVar& rhs) {
  return lhs.feasible_values == rhs.feasible_values && lhs.hlo == rhs.hlo &&
         lhs.map == rhs.map;
}

inline bool operator!=(const HLORTVar& lhs, const HLORTVar& rhs) {
  return !(lhs == rhs);
}

std::optional<int64_t> GetIntOrSplatIntValue(mlir::Attribute attr) {
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    return int_attr.getInt();
  }
  if (auto splat = mlir::dyn_cast<mlir::SplatElementsAttr>(attr)) {
    if (auto element_attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
            splat.getSplatValue<mlir::Attribute>())) {
      return element_attr.getInt();
    }
  }
  return std::nullopt;
}

}  // namespace

std::optional<AffineExpr> OptimizeHloRTVar(const HloInstruction* hlo,
                                           const RuntimeVarIndexing& rt_var,
                                           const Interval& feasible_values,
                                           MLIRContext* mlir_context) {
  if (auto constant_expr = DynCast<HloConstantInstruction>(hlo)) {
    if (rt_var.map.GetAffineMap().isConstant()) {
      const auto idx = rt_var.map.GetAffineMap().getConstantResults();
      auto const_value = constant_expr->literal().GetIntegralAsS64(idx).value();
      if (!feasible_values.Contains(const_value)) {
        return std::nullopt;
      }
      return getAffineConstantExpr(const_value, mlir_context);
    }
  }
  if (auto iota_expr = DynCast<HloIotaInstruction>(hlo)) {
    auto iota_dimension = iota_expr->iota_dimension();
    CHECK(iota_dimension < rt_var.map.GetAffineMap().getNumResults());
    return rt_var.map.GetAffineMap().getResults()[iota_dimension];
  }
  return std::nullopt;
}

std::optional<AffineExpr> OptimizeMlirRTVar(mlir::Operation* op,
                                            const RuntimeVarIndexing& rt_var,
                                            const Interval& feasible_values,
                                            MLIRContext* mlir_context) {
  mlir::Attribute attr;
  if (mlir::matchPattern(op, mlir::m_Constant(&attr))) {
    auto int_val = GetIntOrSplatIntValue(attr);
    if (int_val.has_value()) {
      if (!feasible_values.Contains(*int_val)) {
        return std::nullopt;
      }
      return getAffineConstantExpr(*int_val, mlir_context);
    }
  }
  if (auto iota_op = llvm::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
    int64_t iota_dim = iota_op.getIotaDimension();
    if (iota_dim < rt_var.map.GetAffineMap().getNumResults()) {
      return rt_var.map.GetAffineMap().getResults()[iota_dim];
    }
  }
  return std::nullopt;
}

std::optional<AffineExpr> OptimizeRTVar(const RuntimeVarIndexing& rt_var,
                                        const Interval& feasible_values,
                                        MLIRContext* mlir_context) {
  if (const HloInstruction* hlo = rt_var.hlo()) {
    return OptimizeHloRTVar(hlo, rt_var, feasible_values, mlir_context);
  }
  if (auto* op = rt_var.mlir_op()) {
    return OptimizeMlirRTVar(op, rt_var, feasible_values, mlir_context);
  }
  return std::nullopt;
}

namespace {

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
  auto* mlir_context = affine_map.getContext();
  // TODO (b/446856820): Get context from SymbolicMap after refactoring.
  // Range and runtime variables share the symbol space in the affine map but
  // currently we never have range variables here.
  CHECK_EQ(affine_map.getNumSymbols(), hlo_rt_vars.size());
  for (auto idx = 0; idx < affine_map.getNumSymbols(); ++idx) {
    auto& rt_var = hlo_rt_vars[idx];
    std::optional<AffineExpr> result = OptimizeRTVar(
        RuntimeVarIndexing{rt_var.hlo, IndexingMap::FromTensorSizes(
                                           rt_var.map, rt_var.dim_upper_bounds,
                                           /*symbol_upper_bounds=*/{})},
        rt_var.feasible_values, mlir_context);
    if (!result) {
      continue;
    }
    affine_map =
        affine_map.replace({{getAffineSymbolExpr(idx, mlir_context), *result}},
                           affine_map.getNumDims(), affine_map.getNumSymbols());
  }
  return IndexingMap(affine_map, std::move(dim_vars), /*range_vars=*/{},
                     ConvertHLORTVarsToRTVars(hlo_rt_vars));
}

// Creates an OperandIndexing from an affine map with dimensions, and runtime
// variables.
OperandIndexing CreateOperandIndexingWithRTVars(
    AffineMap operand_map, const std::vector<IndexingMap::Variable>& dim_vars,
    std::vector<HLORTVar> rt_vars) {
  std::vector<RuntimeVarIndexing> rt_indexing;
  rt_indexing.reserve(rt_vars.size());
  for (const HLORTVar& rt : rt_vars) {
    IndexingMap map = IndexingMap::FromTensorSizes(rt.map, rt.dim_upper_bounds,
                                                   /*symbol_upper_bounds=*/{});
    RuntimeVarIndexing operand_indexing{rt.hlo, map};
    rt_indexing.push_back(operand_indexing);
  }

  IndexingMap update_map_ops = FoldRTVarsAndConstructIndexingMap(
      operand_map, dim_vars, std::move(rt_vars));

  OperandIndexing indexing(update_map_ops, rt_indexing);
  indexing.RemoveUnusedSymbols();
  return indexing;
}

HloInstructionIndexing ComputeOutputToInputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  HloInstructionIndexing instr_indexing = CreateElementwiseIndexing(
      instr->operand_count(), instr->shape(), mlir_context);
  for (int64_t operand_id = 0; operand_id < instr->operand_count();
       ++operand_id) {
    // Select allows implicit broadcasting in the predicate. We just handle it
    // generically here.
    if (instr->operand(operand_id)->shape().dimensions().empty() &&
        !instr->shape().dimensions().empty()) {
      instr_indexing.indexing_maps[operand_id].clear();
      instr_indexing.indexing_maps[operand_id].emplace(
          CreateScalarIndexingMap(instr->shape(), mlir_context));
    }
  }
  return instr_indexing;
}

HloInstructionIndexing ComputeInputToOutputCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  IndexingMap identity_map = CreateIdentityMap(instr->shape(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({identity_map});
}

}  // namespace

namespace {

HloInstructionIndexing ComputeOutputToInputBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  IndexingMap indexing_map = ComputeBroadcastIndexingMap(
      bcast->shape().dimensions(), bcast->dimensions(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeInputToOutputBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  const Shape& input_shape = bcast->operand(0)->shape();
  const Shape& output_shape = bcast->shape();

  std::vector<int64_t> added_dims_sizes;
  std::vector<AffineExpr> exprs;
  exprs.reserve(output_shape.dimensions().size());
  for (auto [output_dim_id, output_dim] :
       llvm::enumerate(output_shape.dimensions())) {
    auto operand_dim = bcast->MapUnaryOutputDimToOperandDim(output_dim_id);
    if (!operand_dim.has_value()) {
      exprs.push_back(
          getAffineSymbolExpr(added_dims_sizes.size(), mlir_context));
      added_dims_sizes.push_back(output_dim);
      continue;
    }
    exprs.push_back(getAffineDimExpr(*operand_dim, mlir_context));
  }
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(input_shape.dimensions().size(), added_dims_sizes.size(),
                     exprs, mlir_context),
      input_shape.dimensions(), added_dims_sizes);

  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeOutputToInputConcatenateOpIndexing(
    const HloConcatenateInstruction* concat, MLIRContext* mlir_context) {
  int64_t concat_dim = concat->concatenate_dimension();
  std::vector<int64_t> operand_concat_dim_sizes;
  operand_concat_dim_sizes.reserve(concat->operand_count());
  for (const auto* operand : concat->operands()) {
    operand_concat_dim_sizes.push_back(operand->shape().dimensions(concat_dim));
  }
  return ComputeConcatenateIndexing(concat->shape().dimensions().size(),
                                    concat_dim, concat->shape().dimensions(),
                                    operand_concat_dim_sizes, mlir_context);
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

std::pair<IndexingMap, IndexingMap> ComputeDotOperandsIndexingImpl(
    const Shape& lhs_shape, const Shape& rhs_shape, const Shape& output_shape,
    const DotDimensionNumbers& dim_numbers, MLIRContext* mlir_context) {
  return ComputeDotOperandsIndexing(
      lhs_shape.dimensions(), rhs_shape.dimensions(), output_shape.dimensions(),
      dim_numbers.lhs_batch_dimensions(), dim_numbers.rhs_batch_dimensions(),
      dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.rhs_contracting_dimensions(), mlir_context);
}

// Returns the new map with the results scaled by (operand_shape / scale_shape).
IndexingMap RescaleIndexingMap(const IndexingMap& operand_map,
                               const Shape& operand_shape,
                               const Shape& scale_shape) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(operand_shape.dimensions().size());
  AffineMap affine_map = operand_map.GetAffineMap();
  for (const auto& [scale_dim, operand_dim, expr] :
       llvm::zip(scale_shape.dimensions(), operand_shape.dimensions(),
                 affine_map.getResults())) {
    CHECK_EQ(operand_dim % scale_dim, 0)
        << "Scale dimension must divide the operand dimension.";
    exprs.push_back(scale_dim == operand_dim
                        ? expr
                        : expr.floorDiv(operand_dim / scale_dim));
  }
  return IndexingMap{
      AffineMap::get(affine_map.getNumDims(), affine_map.getNumSymbols(), exprs,
                     affine_map.getContext()),
      operand_map.GetDimVars(), operand_map.GetRangeVars(),
      operand_map.GetRTVars()};
}

HloInstructionIndexing ComputeOutputToInputDotOpIndexing(
    const HloDotInstruction* dot, MLIRContext* mlir_context) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();

  auto [lhs_map, rhs_map] = ComputeDotOperandsIndexingImpl(
      lhs_shape, rhs_shape, dot->shape(), dot->dot_dimension_numbers(),
      mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({lhs_map, rhs_map});
}

HloInstructionIndexing ComputeOutputToInputScaledDotOpIndexing(
    const HloScaledDotInstruction* scaled_dot, MLIRContext* mlir_context) {
  const Shape& lhs_shape = scaled_dot->operand(0)->shape();
  const Shape& rhs_shape = scaled_dot->operand(1)->shape();
  const Shape& lhs_scale_shape = scaled_dot->operand(2)->shape();
  const Shape& rhs_scale_shape = scaled_dot->operand(3)->shape();

  auto [lhs_map, rhs_map] = ComputeDotOperandsIndexingImpl(
      lhs_shape, rhs_shape, scaled_dot->shape(),
      scaled_dot->dot_dimension_numbers(), mlir_context);

  IndexingMap lhs_scale_map =
      RescaleIndexingMap(lhs_map, lhs_shape, lhs_scale_shape);
  IndexingMap rhs_scale_map =
      RescaleIndexingMap(rhs_map, rhs_shape, rhs_scale_shape);

  return HloInstructionIndexing::FromIndexingMaps(
      {lhs_map, rhs_map, lhs_scale_map, rhs_scale_map});
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
  IndexingMap start_indices_map =
      CreateScalarIndexingMap(output_shape, mlir_context);

  AffineMap empty_results_affine_map = AffineMap::get(
      /*dimCount=*/rank, /*symbolCount=*/0, /*results=*/{}, mlir_context);
  std::vector<HLORTVar> offsets_rt_vars;
  offsets_rt_vars.reserve(rank);
  std::vector<AffineExpr> exprs;
  exprs.reserve(rank);

  for (auto [dim, slice_size] :
       llvm::enumerate(dynamic_slice->dynamic_slice_sizes())) {
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);
    const HloInstruction* offset_op =
        dynamic_slice->operand(dim + first_index_num);
    int64_t max_index = input_shape.dimensions(dim) - slice_size;

    // Construct temp objects for optimization
    RuntimeVarIndexing rt_indexing{offset_op, start_indices_map};
    Interval feasible_values{0, max_index};

    auto simplified_expr =
        OptimizeRTVar(rt_indexing, feasible_values, mlir_context);
    if (simplified_expr) {
      exprs.push_back(dim_expr + *simplified_expr);
    } else {
      exprs.push_back(
          dim_expr + getAffineSymbolExpr(offsets_rt_vars.size(), mlir_context));
      offsets_rt_vars.push_back(
          HLORTVar{feasible_values, offset_op, empty_results_affine_map,
                   ShapeUtil::CreateDimensionVectorFromShape(output_shape)});
    }
  }
  std::vector<OperandIndexing> indexing_maps(
      dynamic_slice->operand_count(), OperandIndexing(start_indices_map));

  int symbol_count = offsets_rt_vars.size();
  indexing_maps[0] = CreateOperandIndexingWithRTVars(
      AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/symbol_count, exprs,
                     mlir_context),
      start_indices_map.GetDimVars(), std::move(offsets_rt_vars));
  HloInstructionIndexing result =
      HloInstructionIndexing::FromOperandIndexing(indexing_maps);
  return result;
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

  // start_indices: (d0, ... d{N-1}) -> ()
  IndexingMap start_indices_map =
      CreateScalarIndexingMap(output_shape, mlir_context);

  AffineMap empty_results_affine_map = AffineMap::get(
      /*dimCount=*/rank, /*symbolCount=*/0, /*results=*/{}, mlir_context);
  // update: (d0 - rt0, ..., d{N-1} - rt{N-1})
  std::vector<AffineExpr> exprs;
  exprs.reserve(rank);
  std::vector<HLORTVar> rt_vars;
  rt_vars.reserve(rank);
  for (auto [dim, slice_size] : llvm::enumerate(update_shape.dimensions())) {
    exprs.push_back(getAffineDimExpr(dim, mlir_context) -
                    getAffineSymbolExpr(dim, mlir_context));
    Interval feasible_values{0, output_shape.dimensions(dim) - slice_size};
    rt_vars.push_back(HLORTVar{
        feasible_values, dus->operand(2 + dim), empty_results_affine_map,
        ShapeUtil::CreateDimensionVectorFromShape(output_shape)});
  }
  OperandIndexing update_indexing = CreateOperandIndexingWithRTVars(
      AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/rank,
                     /*results=*/exprs, mlir_context),
      operand_map.GetDimVars(), std::move(rt_vars));
  std::vector<OperandIndexing> indexing_maps(
      dus->operand_count(), OperandIndexing(start_indices_map));
  indexing_maps[0] = OperandIndexing(operand_map);
  indexing_maps[1] = update_indexing;
  return HloInstructionIndexing::FromOperandIndexing(indexing_maps);
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
  // (d0, ... d{rank - 1}) -> (d0, s0),
  // where 0 <= s0 <= indices_shape[1] - 1.
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
  // If operand dimension `i` corresponds to `start_index_map[j]`, then i-th
  // dimension of operand is indexed as d_{offset_dims[i]} + start_indices[d0,
  // j], otherwise it's d_{offset_dims[i]}.
  std::vector<HLORTVar> rt_vars;
  std::vector<AffineExpr> exprs;
  exprs.reserve(operand_shape.dimensions().size());
  const auto& start_index_map = dimension_numbers.start_index_map();
  for (auto [operand_dim_id, slice_size] :
       llvm::enumerate(gather->gather_slice_sizes())) {
    int64_t output_dim_id = dimension_numbers.offset_dims(operand_dim_id);
    exprs.push_back(getAffineDimExpr(output_dim_id, mlir_context));

    int64_t start_index_map_idx =
        absl::c_find(start_index_map, operand_dim_id) - start_index_map.begin();
    if (start_index_map_idx == start_index_map.size()) {
      continue;
    }
    AffineMap rt_var_map = AffineMap::get(
        output_rank, /*symbolCount=*/0,
        {indices_id_dim,
         getAffineConstantExpr(start_index_map_idx, mlir_context)},
        mlir_context);
    rt_vars.push_back(HLORTVar{
        Interval{0, operand_shape.dimensions(operand_dim_id) - slice_size},
        gather->operand(1), rt_var_map,
        ShapeUtil::CreateDimensionVectorFromShape(output_shape)});
    exprs.back() =
        exprs.back() + getAffineSymbolExpr(rt_vars.size() - 1, mlir_context);
  }
  OperandIndexing operand_indexing = CreateOperandIndexingWithRTVars(
      AffineMap::get(/*dimCount=*/output_rank,
                     /*symbolCount=*/start_index_map.size(), exprs,
                     mlir_context),
      dim_vars, std::move(rt_vars));

  return HloInstructionIndexing::FromOperandIndexing(
      {operand_indexing, OperandIndexing(indices_map)});
}

}  // namespace

namespace {

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
  IndexingMap input_indexing_map =
      ComputePadIndexingMap(output_shape.dimensions(), padding_low,
                            padding_high, padding_interior, mlir_context);
  IndexingMap padding_value_indexing_map =
      CreateScalarIndexingMap(output_shape, mlir_context);
  return HloInstructionIndexing::FromIndexingMaps(
      {input_indexing_map, padding_value_indexing_map});
}

HloInstructionIndexing ComputeOutputToInputReduceOpIndexing(
    const HloReduceInstruction* reduce, MLIRContext* mlir_context) {
  const Shape& input_shape = reduce->operand(0)->shape();
  const Shape& output_shape = GetOutputShape(reduce, 0);

  IndexingMap inputs_indexing_map = ComputeReduceInputIndexingMap(
      input_shape.dimensions(), output_shape.dimensions(), reduce->dimensions(),
      mlir_context);
  IndexingMap inits_indexing_map =
      CreateScalarIndexingMap(output_shape, mlir_context);

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(reduce->operand_count());
  for (int64_t id = 0; id < reduce->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(
        OperandIndexing(inputs_indexing_map));
  }
  for (int64_t id = reduce->input_count(); id < reduce->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(
        OperandIndexing(inits_indexing_map));
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
      instr_indexing.indexing_maps[id].insert(
          OperandIndexing(inits_indexing_map));
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
    instr_indexing.indexing_maps[id].insert(
        OperandIndexing(inputs_indexing_map));
  }
  return instr_indexing;
}

IndexingMap ComposeIndexingMapsForWindow(
    absl::Span<const int64_t> input_dimensions,
    absl::Span<const int64_t> output_dimensions, const Window& window,
    MLIRContext* mlir_context) {
  size_t rank = input_dimensions.size();
  SmallVector<int64_t> window_dims, window_strides, window_dilations,
      base_dilations, padding;
  window_dims.reserve(rank);
  window_strides.reserve(rank);
  window_dilations.reserve(rank);
  base_dilations.reserve(rank);
  padding.reserve(rank * 2);
  for (const auto& dim : window.dimensions()) {
    window_dims.push_back(dim.size());
    window_strides.push_back(dim.stride());
    window_dilations.push_back(dim.window_dilation());
    base_dilations.push_back(dim.base_dilation());
    padding.push_back(dim.padding_low());
    padding.push_back(dim.padding_high());
  }
  return ComposeWindowIndexingMap(input_dimensions, output_dimensions,
                                  window_dims, window_strides, window_dilations,
                                  base_dilations, padding, mlir_context);
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
  IndexingMap inits_indexing_map =
      CreateScalarIndexingMap(output_shape, mlir_context);

  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(reduce_window->operand_count());
  for (int64_t id = 0; id < reduce_window->input_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(OperandIndexing(inputs_indexing));
  }
  for (int64_t id = reduce_window->input_count();
       id < reduce_window->operand_count(); ++id) {
    instr_indexing.indexing_maps[id].insert(
        OperandIndexing(inits_indexing_map));
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
  llvm::MapVector<AffineExpr, Interval> input_constraints;
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
  // We may need to simplify and remove unused symbols again, as the input
  // feature dimension size may be trivial.
  inputs_indexing.Simplify();
  inputs_indexing.RemoveUnusedSymbols();

  // Indexing map for the kernel value.
  IndexingMap kernel_indexing(
      AffineMap::get(rank, kernel_symbols.size(), kernel_exprs, mlir_context),
      DimVarsFromTensorSizes(output_shape.dimensions()), kernel_symbols,
      /*rt_vars=*/{});
  kernel_indexing.Simplify();
  kernel_indexing.RemoveUnusedSymbols();

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
        (input_subshape.empty() || input_num_elements < output_num_elements)) {
      input_num_elements *= input_dims[input_dim_id];
      input_subshape.push_back(input_dims[input_dim_id]);
      ++input_dim_id;
      continue;
    }
    if (output_dim_id < output.dimensions().size() &&
        (output_subshape.empty() || output_num_elements < input_num_elements)) {
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
  reshape_indexing_map.Simplify(
      IndexingMap::SimplifyPointDimensions::kPreserve);
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}
HloInstructionIndexing ComputeInputToOutputReshapeOpIndexing(
    const HloReshapeInstruction* reshape, MLIRContext* mlir_context) {
  const auto& input = reshape->operand(0)->shape();
  const auto& output = reshape->shape();

  IndexingMap reshape_indexing_map = IndexingMap::FromTensorSizes(
      ComputeReshapeIndexingMap(output, input, mlir_context),
      input.dimensions(), {});
  reshape_indexing_map.Simplify(
      IndexingMap::SimplifyPointDimensions::kPreserve);
  return HloInstructionIndexing::FromIndexingMaps({reshape_indexing_map});
}

HloInstructionIndexing ComputeReverseOpIndexing(
    const HloReverseInstruction* reverse, MLIRContext* mlir_context) {
  IndexingMap indexing_map = ComputeReverseIndexingMap(
      reverse->shape().dimensions(), reverse->dimensions(), mlir_context);
  return HloInstructionIndexing::FromIndexingMaps({indexing_map});
}

HloInstructionIndexing ComputeOutputToInputSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  IndexingMap indexing_map = ComputeSliceIndexingMap(
      slice->shape().dimensions(), slice->slice_starts(),
      slice->slice_strides(), mlir_context);
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
                          MLIRContext* mlir_context) {
  return GetBitcastMap(ShapeUtil::MakeShapeWithDescendingLayout(
                           output_shape.element_type(), input_shape),
                       output_shape, mlir_context);
}
IndexingMap GetBitcastMap(absl::Span<const int64_t> input_shape,
                          absl::Span<const int64_t> output_shape,
                          MLIRContext* mlir_context) {
  return GetBitcastMap(
      ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::S8, input_shape),
      ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::S8, output_shape),
      mlir_context);
}
IndexingMap GetBitcastMap(const Shape& input_shape, const Shape& output_shape,
                          MLIRContext* mlir_context) {
  ShapeUtil::BitcastDecomposition decomposed_bitcast =
      ShapeUtil::DecomposeBitcast(input_shape, output_shape);
  if (!decomposed_bitcast.has_value()) {
    return IndexingMap::GetUndefined();
  }

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          *decomposed_bitcast)) {
    auto permutation = ShapeUtil::DeduceTransposeDimensionsForBitcast(
        input_shape, output_shape);
    CHECK(permutation.has_value())
        << "Failed to deduce permutation for a bitcast.";

    return IndexingMap::FromTensorSizes(
        ComputeTransposeIndexingMap(permutation.value(), mlir_context),
        input_shape.dimensions(), {});
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          *decomposed_bitcast)) {
    // Note: ComputeReshapeIndexingMap assumes it's computing an output->input
    // indexing, so input and output are reversed.
    return IndexingMap::FromTensorSizes(
        ComputeReshapeIndexingMap(output_shape, input_shape, mlir_context),
        input_shape.dimensions(), {});
  }
  // `trt` stands for transpose-reshape-transpose decomposition of bitcast.
  auto trt = std::get<ShapeUtil::BitcastDecompositionTrt>(*decomposed_bitcast);
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
  bitcast_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);
  return HloInstructionIndexing::FromIndexingMaps({bitcast_map});
}

HloInstructionIndexing ComputeInputToOutputBitcastOpIndexing(
    const HloInstruction* bitcast, MLIRContext* mlir_context) {
  auto bitcast_map = GetBitcastMap(bitcast->operand(0)->shape(),
                                   bitcast->shape(), mlir_context);
  bitcast_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);
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
                              MLIRContext* mlir_context) {
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
    std::vector<OperandIndexing> to_remove, to_add;
    for (OperandIndexing idx : operand_indexing) {
      auto old_idx = idx;
      if (idx.Simplify()) {
        to_remove.push_back(old_idx);
        idx.RemoveUnusedSymbols();
        to_add.push_back(idx);
      }
      if (idx.map().IsUndefined()) {
        to_remove.push_back(idx);
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
    instr_indexing.indexing_maps[index].insert(OperandIndexing(map));
  }
  return instr_indexing;
}

HloInstructionIndexing HloInstructionIndexing::FromOperandIndexing(
    absl::Span<const OperandIndexing> operand_indexing) {
  HloInstructionIndexing instr_indexing;
  instr_indexing.indexing_maps.resize(operand_indexing.size());
  for (const auto& [index, rt_map] : llvm::enumerate(operand_indexing)) {
    instr_indexing.indexing_maps[index].insert(rt_map);
  }
  return instr_indexing;
}

std::string HloInstructionIndexing::ToString(
    absl::string_view separator) const {
  std::stringstream out;
  for (const auto& [operand_id, operand_indexes] :
       llvm::enumerate(indexing_maps)) {
    out << "operand id = " << operand_id << ' ';
    for (const auto& idx : operand_indexes) {
      if (idx.map().IsUndefined()) {
        out << "unknown indexing";
        continue;
      }
      out << idx;
    }
    out << separator;
  }
  return out.str();
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  out << instr_indexing.ToString();
  return out;
}

const Shape& GetOutputShape(const HloInstruction* instr, int64_t output_id) {
  return instr->shape().IsTuple()
             ? ShapeUtil::GetSubshape(instr->shape(), {output_id})
             : instr->shape();
}

GroupedByOpIndexing GroupIndexingMapsByProducers(
    const HloInstructionIndexing& indexing, const HloInstruction* instr) {
  GroupedByOpIndexing result;
  for (const auto& [operand_id, indexing_maps] :
       llvm::enumerate(indexing.indexing_maps)) {
    result[instr->operand(operand_id)].insert(indexing_maps.begin(),
                                              indexing_maps.end());
  }
  return result;
}

GroupedByOpIndexing ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, HloInstructionAdaptor target_instr,
    MLIRContext* mlir_context) {
  OperandIndexing initial_map = OperandIndexing(
      CreateIdentityMap(target_instr.instruction().shape(), mlir_context));

  GroupedByOpIndexing grouped_indexing_maps;
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
    auto producer_indexing =
        ComputeOutputToInputIndexing(&it->instruction(),
                                     /*output_id=*/0, mlir_context);
    auto consumer_indexing_maps =
        grouped_indexing_maps.find(&it->instruction());
    if (consumer_indexing_maps == grouped_indexing_maps.end()) {
      continue;
    }
    // Indexing maps have to be copied because of rehashing. Consider using a
    // different container to get better performance.
    OperandIndexingSet consumer_indexing_maps_copy =
        consumer_indexing_maps->second;
    for (const auto& [producer_operand_id, producer_operand_indexing] :
         llvm::enumerate(producer_indexing.indexing_maps)) {
      auto producer_operand_adaptor = it->GetOperand(producer_operand_id);
      for (const OperandIndexing& producer_map : producer_operand_indexing) {
        for (const OperandIndexing& consumer_map :
             consumer_indexing_maps_copy) {
          OperandIndexing composed_map =
              ComposeOperandIndexing(consumer_map, producer_map);
          composed_map.Simplify();
          composed_map.RemoveUnusedSymbols();
          composed_map.VerifyOrDie();
          grouped_indexing_maps[&producer_operand_adaptor.instruction()].insert(
              composed_map);
        }
      }
    }
  }
  return grouped_indexing_maps;
}

namespace {
// Returns a linearized shape, i.e. tensor<num_elements(input) x
// element_type>.
Shape GetLinearizedShape(const Shape& shape) {
  if (shape.dimensions().empty()) {
    return shape;
  }
  std::vector<int64_t> dims{ShapeUtil::ElementsIn(shape)};
  auto result = Shape(shape.element_type(), dims);
  *result.mutable_layout() = xla::Layout({0});
  return result;
}
}  // namespace

llvm::SmallVector<IndexingMap, 4> MapLogicalToLinearizedPhysicalShape(
    absl::Span<const HloInstruction* const> operands,
    MLIRContext* mlir_context) {
  llvm::SmallVector<IndexingMap, 4> indexing_maps;
  // For every operand compute thread ID -> physical layout of operand
  // indexing map.
  for (const HloInstruction* operand : operands) {
    const Shape& operand_shape = operand->shape();

    IndexingMap operand_logical_to_physical_map =
        GetIndexingMapFromLogicalToPhysicalLayout(operand_shape, mlir_context);
    IndexingMap operand_physical_to_linearized_shape = GetBitcastMap(
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            operand_shape),
        GetLinearizedShape(operand_shape), mlir_context);
    IndexingMap operand_logical_to_linearized_physical_shape =
        operand_logical_to_physical_map * operand_physical_to_linearized_shape;
    operand_logical_to_linearized_physical_shape.Simplify();
    indexing_maps.push_back(
        std::move(operand_logical_to_linearized_physical_shape));
  }
  return indexing_maps;
}

void GetThreadIdToInputMemoryLayoutsMaps(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const IndexingMap> hero_indexing_maps,
    const HloInstructionAdaptor& hero,
    absl::Span<const HloInstruction* const> operands,
    absl::Span<const IndexingMap> operand_logical_to_linearized_physical_maps,
    MLIRContext* mlir_context, GroupedByOpIndexingMap& result) {
  for (const auto& [hero_operand_index, hero_operand] :
       llvm::enumerate(hero.GetOperands())) {
    if (hero_operand.shape().dimensions().empty()) {
      continue;
    }
    // Compute thread ID -> hero operand indexing map.
    const IndexingMap& thread_id_to_hero_operand_map =
        hero_indexing_maps[hero_operand_index];
    // Compute indexing from output to inputs for logical layout.
    GroupedByOpIndexing instr_indexing_keyed_by_operands =
        ComputeGroupedOutputToInputIndexing(fusion_adaptor, hero_operand,
                                            mlir_context);
    // For every operand compute thread ID -> physical layout of operand
    // indexing map.
    for (auto&& [operand, operand_linearized_physical_map] :
         llvm::zip(operands, operand_logical_to_linearized_physical_maps)) {
      auto operand_indexing_maps_it =
          instr_indexing_keyed_by_operands.find(operand);
      if (operand_indexing_maps_it == instr_indexing_keyed_by_operands.end()) {
        continue;
      }

      for (const OperandIndexing& operand_indexing :
           operand_indexing_maps_it->second) {
        const IndexingMap& operand_indexing_map = operand_indexing.map();
        // If one of the indexing maps for the operand is undefined, we remove
        // all indexing maps for it and store only the undefined one.
        if (operand_indexing_map.IsUndefined()) {
          result[operand] = {operand_indexing_map};
          break;
        }
        IndexingMap logical_output_to_linearized_physical_input_map =
            operand_indexing_map * operand_linearized_physical_map;
        IndexingMap thread_id_to_linearized_physical_input_map =
            thread_id_to_hero_operand_map *
            logical_output_to_linearized_physical_input_map;
        thread_id_to_linearized_physical_input_map.Simplify();
        result[operand].insert(thread_id_to_linearized_physical_input_map);
      }
    }
  }
}

// Replaces RTVars with the midpoints of the feasible intervals.
void AssignValuesToRTVars(IndexingMap* indexing_map) {
  // If RTVars are present, replace them with constants.
  if (indexing_map->GetRTVarsCount() == 0) {
    return;
  }
  llvm::SmallVector<AffineExpr, 2> symbol_replacements;
  for (int64_t symbol_id = 0; symbol_id < indexing_map->GetRangeVarsCount();
       ++symbol_id) {
    symbol_replacements.push_back(
        mlir::getAffineSymbolExpr(symbol_id, indexing_map->GetMLIRContext()));
  }
  for (const IndexingMap::Variable& rt_var : indexing_map->GetRTVars()) {
    // Take midpoint of the feasible interval for the RT variable.
    symbol_replacements.push_back(
        getAffineConstantExpr((rt_var.bounds.lower + rt_var.bounds.upper) / 2,
                              indexing_map->GetMLIRContext()));
  }
  AffineMap thread_x_to_input_no_dim_symbols =
      indexing_map->GetAffineMap().replaceDimsAndSymbols(
          {}, symbol_replacements, indexing_map->GetDimVarsCount(),
          indexing_map->GetRangeVarsCount());
  *indexing_map = IndexingMap{thread_x_to_input_no_dim_symbols,
                              indexing_map->GetDimVars(),
                              indexing_map->GetRangeVars(),
                              {}};
  indexing_map->Simplify();
  indexing_map->RemoveUnusedSymbols();
}

HloInstructionIndexing ComputeOutputToInputAllGatherOpIndexing(
    const HloAllGatherInstruction* instr, MLIRContext* mlir_context) {
  // CHECK_EQ(instr->all_gather_dimension(), 0);
  // if (instr->all_gather_dimension() != 0) {
  //   return CreateUnknownIndexing(instr->operand_count());
  // }

  int64_t all_gather_dim = instr->all_gather_dimension();

  auto output_rank = instr->shape().dimensions().size();

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);

  int64_t all_gather_input_dim_size =
      instr->operand(0)->shape().dimensions()[instr->all_gather_dimension()];

  for (int64_t i = 0; i < output_rank; ++i) {
    auto dim = mlir::getAffineDimExpr(i, mlir_context);
    exprs.push_back(i == all_gather_dim ? dim % all_gather_input_dim_size
                                        : dim);
  }

  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, /*symbolCount=*/0, exprs, mlir_context),
      instr->shape().dimensions(), {});

  AffineExpr replica_id_expr =
      mlir::getAffineDimExpr(all_gather_dim, mlir_context)
          .floorDiv(all_gather_input_dim_size);

  IndexingMap replica_id_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, /*symbolCount=*/0, replica_id_expr,
                     mlir_context),
      instr->shape().dimensions(), {});

  OperandIndexing operand_indexing(indexing_map, {}, replica_id_map);

  return HloInstructionIndexing::FromOperandIndexing({operand_indexing});
}

HloInstructionIndexing ComputeOutputToInputIndexing(const HloInstruction* instr,
                                                    int output_id,
                                                    MLIRContext* mlir_context) {
  if (HloInstruction::IsOpElementwise(instr->opcode()) ||
      // Note: map has a `dimensions` attribute, but it does nothing. See
      // b/65689298.
      instr->opcode() == HloOpcode::kMap ||
      // For a single device, all-reduce is an elementwise op.
      instr->opcode() == HloOpcode::kAllReduceStart ||
      instr->opcode() == HloOpcode::kAllReduceDone) {
    return ComputeOutputToInputCwiseOpIndexing(instr, mlir_context);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeOutputToInputBitcastOpIndexing(instr, mlir_context);
  }
  // go/keep-sorted start
  if (auto all_gather = DynCast<HloAllGatherInstruction>(instr)) {
    return ComputeOutputToInputAllGatherOpIndexing(all_gather, mlir_context);
  }
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeOutputToInputBroadcastOpIndexing(broadcast, mlir_context);
  }
  if (auto concat = DynCast<HloConcatenateInstruction>(instr)) {
    return ComputeOutputToInputConcatenateOpIndexing(concat, mlir_context);
  }
  if (auto constant = DynCast<HloConstantInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto convolution = DynCast<HloConvolutionInstruction>(instr)) {
    return ComputeOutputToInputConvolutionOpIndexing(convolution, mlir_context);
  }
  if (auto dot = DynCast<HloDotInstruction>(instr)) {
    return ComputeOutputToInputDotOpIndexing(dot, mlir_context);
  }
  if (auto dus = DynCast<HloDynamicUpdateSliceInstruction>(instr)) {
    return ComputeOutputToInputDynamicUpdateSliceOpIndexing(dus, mlir_context);
  }
  if (auto dynamic_slice = DynCast<HloDynamicSliceInstruction>(instr)) {
    return ComputeOutputToInputDynamicSliceOpIndexing(dynamic_slice,
                                                      mlir_context);
  }
  if (auto fusion = DynCast<HloFusionInstruction>(instr)) {
    return ComputeOutputToInputFusionOpIndexing(fusion, output_id,
                                                mlir_context);
  }
  if (auto gather = DynCast<HloGatherInstruction>(instr)) {
    return ComputeOutputToInputGatherOpIndexing(gather, mlir_context);
  }
  if (auto iota = DynCast<HloIotaInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto pad = DynCast<HloPadInstruction>(instr)) {
    return ComputeOutputToInputPadOpIndexing(pad, mlir_context);
  }
  if (auto parameter = DynCast<HloParameterInstruction>(instr)) {
    return HloInstructionIndexing{};
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeOutputToInputReduceOpIndexing(reduce, mlir_context);
  }
  if (auto reduce_window = DynCast<HloReduceWindowInstruction>(instr)) {
    return ComputeOutputToInputReduceWindowOpIndexing(reduce_window,
                                                      mlir_context);
  }
  if (auto reshape = DynCast<HloReshapeInstruction>(instr)) {
    return ComputeOutputToInputReshapeOpIndexing(reshape, mlir_context);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, mlir_context);
  }
  if (auto scaled_dot = DynCast<HloScaledDotInstruction>(instr)) {
    return ComputeOutputToInputScaledDotOpIndexing(scaled_dot, mlir_context);
  }
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeOutputToInputSliceOpIndexing(slice, mlir_context);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeOutputToInputTransposeOpIndexing(transpose, mlir_context);
  }
  // go/keep-sorted end
  LOG(ERROR) << "ComputeOutputToInputIndexing is not implemented for opcode "
             << instr->opcode();
  // If we cannot compute output-to-input indexing, we return std::nullopt for
  // every op parameter.
  return CreateUnknownIndexing(instr->operand_count());
}

HloInstructionIndexing ComputeInputToOutputIndexing(const HloInstruction* instr,
                                                    int input_id,
                                                    MLIRContext* mlir_context) {
  if (HloInstruction::IsOpElementwise(instr->opcode()) ||
      // Note: map has a `dimensions` attribute, but it does nothing. See
      // b/65689298.
      instr->opcode() == HloOpcode::kMap ||
      // For a single device, all-reduce has 1:1 output to input mapping.
      instr->opcode() == HloOpcode::kAllReduceStart ||
      instr->opcode() == HloOpcode::kAllReduceDone) {
    return ComputeInputToOutputCwiseOpIndexing(instr, mlir_context);
  }
  if (instr->opcode() == HloOpcode::kBitcast) {
    return ComputeInputToOutputBitcastOpIndexing(instr, mlir_context);
  }
  // go/keep-sorted start
  if (auto broadcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeInputToOutputBroadcastOpIndexing(broadcast, mlir_context);
  }
  if (auto concat = DynCast<HloConcatenateInstruction>(instr)) {
    return ComputeInputToOutputConcatenateOpIndexing(concat, input_id,
                                                     mlir_context);
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
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeInputToOutputSliceOpIndexing(slice, mlir_context);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeInputToOutputTransposeOpIndexing(transpose, mlir_context);
  }
  // go/keep-sorted end
  if (instr->opcode() == HloOpcode::kTuple) {
    return HloInstructionIndexing::FromIndexingMaps({CreateIdentityMap(
        instr->shape().tuple_shapes(input_id), mlir_context)});
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
  std::vector<HloInstructionAdaptor> chain =
      HloFindUseChain(epilogue_parent, epilogue_root);
  CHECK(!chain.empty()) << "There is no use chain from parent to root";
  OperandIndexing root_indexing(
      CreateIdentityMap(epilogue_parent.shape(), mlir_context));
  for (int i = 1; i < chain.size(); ++i) {
    const auto& producer = chain[i - 1].instruction();
    const auto& user = chain[i].instruction();
    auto user_indexing = ComputeInputToOutputIndexing(
        &user, user.operand_index(&producer), mlir_context);
    root_indexing = ComposeOperandIndexing(
        {root_indexing}, *user_indexing.indexing_maps[0].begin());
    root_indexing.Simplify();
    root_indexing.RemoveUnusedSymbols();
  }
  return root_indexing.map();
}

std::string OperandIndexing::ToString() const {
  std::string result = absl::StrCat(xla::ToString(map_));
  if (!rt_vars_.empty()) {
    absl::StrAppend(&result, "\nruntime variables:\n");
    for (const auto& [id, rt_var] : llvm::enumerate(rt_vars_)) {
      absl::StrAppend(&result, "\nrt", id, ": ", rt_var.ToString());
    }
  }
  if (replica_id_map_.has_value()) {
    absl::StrAppend(&result, "\nreplica id:\n",
                    xla::ToString(*replica_id_map_));
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const OperandIndexing& var) {
  os << var.ToString();
  return os;
}

llvm::SmallBitVector OperandIndexing::RemoveUnusedSymbols() {
  const int64_t range_vars_count = map_.GetRangeVarsCount();
  llvm::SmallBitVector removed = map_.RemoveUnusedSymbols();
  std::vector<RuntimeVarIndexing> updated_rt_vars;
  updated_rt_vars.reserve(rt_vars_.size());
  for (const auto& [i, rt] : llvm::enumerate(rt_vars_)) {
    const int64_t idx = i + range_vars_count;
    if (idx < removed.size() && removed[idx]) {
      continue;
    }
    updated_rt_vars.push_back(rt);
  }
  rt_vars_.swap(updated_rt_vars);
  return removed;
}

bool OperandIndexing::Verify(std::ostream& out) const {
  bool ok = map_.Verify(out);
  if (map_.GetRTVars().size() != rt_vars_.size()) {
    out << "number of rt vars in indexing map " << xla::ToString(map_) << " "
        << map_.GetRTVars().size()
        << " does not match number of runtime variables " << rt_vars_.size();
    ok = false;
  }
  for (const auto& [i, rt] : llvm::enumerate(rt_vars_)) {
    int64_t rt_dim_vars_size = rt.map.GetDimVars().size();
    int64_t map_dim_vars_size = map_.GetDimVars().size();
    if (rt_dim_vars_size != map_dim_vars_size) {
      out << "rt variable " << i << " " << rt.ToString()
          << " number of dim vars " << rt_dim_vars_size
          << " does not match number of dim vars for operand"
          << xla::ToString(map_) << " " << map_dim_vars_size;
      ok = false;
    }
  }
  return ok;
}

void OperandIndexing::VerifyOrDie() const {
  std::stringstream ss;
  CHECK(Verify(ss)) << ss.str() << " map: " << ToString();
}

bool operator==(const OperandIndexing& lhs, const OperandIndexing& rhs) {
  return lhs.map_ == rhs.map_ && absl::c_equal(lhs.rt_vars_, rhs.rt_vars_);
}

bool operator==(const RuntimeVarIndexing& lhs, const RuntimeVarIndexing& rhs) {
  return lhs.map == rhs.map && lhs.instruction_ref == rhs.instruction_ref;
}

OperandIndexing ComposeOperandIndexing(const OperandIndexing& first,
                                       const OperandIndexing& second) {
  IndexingMap map = ComposeIndexingMaps(first.map(), second.map());
  std::vector<RuntimeVarIndexing> combined_runtime;
  combined_runtime.reserve(first.runtime_variables().size() +
                           second.runtime_variables().size());
  combined_runtime.insert(combined_runtime.end(),
                          first.runtime_variables().begin(),
                          first.runtime_variables().end());
  for (const auto& rt_var : second.runtime_variables()) {
    IndexingMap combined_map = ComposeIndexingMaps(first.map(), rt_var.map);
    combined_runtime.push_back(
        RuntimeVarIndexing{rt_var.instruction_ref, combined_map});
  }

  std::optional<IndexingMap> replica_id_map;
  if (first.replica_id_map().has_value()) {
    replica_id_map = first.replica_id_map();
    if (second.replica_id_map().has_value()) {
      // TODO(shyshkov): Support chaining collective ops.
      return OperandIndexing(IndexingMap::GetUndefined(), {});
    }
  }

  if (second.replica_id_map().has_value()) {
    replica_id_map =
        ComposeIndexingMaps(first.map(), second.replica_id_map().value());
  }

  return OperandIndexing(map, combined_runtime, replica_id_map);
}

std::string RuntimeVarIndexing::ToString() const {
  // Handle both HLO and MLIR operations producing a unified enough format to
  // avoid duplication in tests.
  std::string instruction_str;
  if (auto* hlo = std::get_if<const HloInstruction*>(&instruction_ref)) {
    if (*hlo) {
      // For HLO, print simplified format for parameter and constant
      if ((*hlo)->opcode() == HloOpcode::kParameter) {
        instruction_str =
            absl::StrCat("parameter(", (*hlo)->parameter_number(), ")");
      } else if ((*hlo)->opcode() == HloOpcode::kConstant) {
        instruction_str = "constant";
        // Print constant value for scalar constants
        const xla::Literal& literal = (*hlo)->literal();
        if (xla::ShapeUtil::IsScalar(literal.shape())) {
          instruction_str =
              absl::StrCat("constant(", literal.ToStringWithoutShape(), ")");
        }
      } else {
        instruction_str = (*hlo)->name();
      }
    } else {
      instruction_str = "<null hlo>";
    }
  } else if (auto* val = std::get_if<mlir::Value>(&instruction_ref)) {
    if (*val) {
      if (auto* op = val->getDefiningOp()) {
        // Try to extract constant value for stablehlo/mhlo constant ops
        llvm::StringRef op_name = op->getName().getStringRef();
        if (op_name == "stablehlo.constant" || op_name == "mhlo.constant") {
          instruction_str = "constant";
          if (auto attr = op->getAttrOfType<mlir::DenseElementsAttr>("value")) {
            if (attr.isSplat() && attr.getNumElements() == 1) {
              // Scalar constant - print the value
              auto elem_type = attr.getElementType();
              if (elem_type.isSignlessInteger()) {
                instruction_str = absl::StrCat(
                    "constant(",
                    attr.getSplatValue<llvm::APInt>().getSExtValue(), ")");
              } else if (elem_type.isF32()) {
                instruction_str =
                    absl::StrCat("constant(", attr.getSplatValue<float>(), ")");
              } else if (elem_type.isF64()) {
                instruction_str = absl::StrCat(
                    "constant(", attr.getSplatValue<double>(), ")");
              }
            }
          }
        } else {
          instruction_str = op_name.str();
        }
      } else {
        // Block argument is print as "parameter(N)" to match HLO format.
        auto block_arg = llvm::cast<mlir::BlockArgument>(*val);
        instruction_str =
            absl::StrCat("parameter(", block_arg.getArgNumber(), ")");
      }
    } else {
      instruction_str = "<null value>";
    }
  }
  return absl::StrCat(instruction_str, "; ", xla::ToString(map));
}

std::ostream& operator<<(std::ostream& os, const RuntimeVarIndexing& var) {
  os << var.ToString();
  return os;
}

IndexingMapSet ToIndexingMapSet(
    const OperandIndexingSet& operand_indexing_set) {
  IndexingMapSet result;
  result.reserve(operand_indexing_set.size());
  for (const auto& idx : operand_indexing_set) {
    result.insert(idx.map());
  }
  return result;
}

}  // namespace xla
