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

#include "xla/hlo/analysis/stablehlo_indexing_analysis.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"  // IWYU pragma: keep
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_analysis_utils.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using namespace ::mlir::stablehlo;  // NOLINT
namespace mhlo = ::mlir::mhlo;

using ::llvm::ArrayRef;
using ::llvm::enumerate;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::BlockArgument;
using ::mlir::DenseIntElementsAttr;
using ::mlir::dyn_cast;
using ::mlir::MLIRContext;
using ::mlir::Operation;
using ::mlir::RankedTensorType;
using ::mlir::SmallVector;
using ::mlir::Value;

HloInstructionIndexing CreateUnknownIndexing(int64_t count) {
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(count);
  for (int64_t i = 0; i < count; ++i) {
    indexing.indexing_maps[i].insert(
        OperandIndexing{IndexingMap::GetUndefined()});
  }
  return indexing;
}

Shape GetShape(Value value) {
  auto shaped_type = dyn_cast<RankedTensorType>(value.getType());
  if (!shaped_type) {
    return Shape();
  }
  std::vector<int64_t> dimensions(shaped_type.getShape().begin(),
                                  shaped_type.getShape().end());
  return ShapeUtil::MakeShape(F32, dimensions);
}

// Operation-specific helper implementations

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    AllGatherOp all_gather, int output_id) {
  MLIRContext* context = all_gather.getContext();
  int64_t all_gather_dim = all_gather.getAllGatherDim();
  auto output_shape = GetShape(all_gather.getResult(0));
  int64_t output_rank = output_shape.dimensions().size();

  // Input shape for the first operand
  auto input_shape = GetShape(all_gather.getOperand(0));
  int64_t all_gather_input_dim_size = input_shape.dimensions(all_gather_dim);

  std::vector<AffineExpr> exprs;
  exprs.reserve(output_rank);

  for (int64_t i = 0; i < output_rank; ++i) {
    auto dim = mlir::getAffineDimExpr(i, context);
    exprs.push_back(i == all_gather_dim ? dim % all_gather_input_dim_size
                                        : dim);
  }

  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, 0, exprs, context), output_shape.dimensions(),
      {});

  AffineExpr replica_id_expr = mlir::getAffineDimExpr(all_gather_dim, context)
                                   .floorDiv(all_gather_input_dim_size);

  IndexingMap replica_id_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, 0, replica_id_expr, context),
      output_shape.dimensions(), {});

  OperandIndexing operand_indexing(indexing_map, {}, replica_id_map);

  HloInstructionIndexing indexing;
  // HLO implementation only returns indexing for the first operand.
  // We mirror this behavior for consistency, although StableHLO ops might be
  // variadic.
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(operand_indexing);
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    BitcastConvertOp bitcast, int output_id) {
  MLIRContext* context = bitcast.getContext();
  auto input_shape = GetShape(bitcast.getOperand());
  auto output_shape = GetShape(bitcast.getResult());
  IndexingMap indexing_map = GetBitcastMap(output_shape, input_shape, context);
  indexing_map.Simplify();
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    BroadcastInDimOp bcast, int output_id) {
  MLIRContext* context = bcast.getContext();
  // Check if result has RankedTensorType
  if (!dyn_cast<RankedTensorType>(bcast.getResult().getType())) {
    return CreateUnknownIndexing(1);
  }
  auto output_shape = GetShape(bcast.getResult());
  IndexingMap indexing_map = ComputeBroadcastIndexingMap(
      output_shape.dimensions(), bcast.getBroadcastDimensions(), context);
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ConcatenateOp concat, int output_id) {
  MLIRContext* context = concat.getContext();
  int64_t concat_dim = concat.getDimension();
  auto output_shape = GetShape(concat.getResult());
  std::vector<int64_t> operand_concat_dim_sizes;
  operand_concat_dim_sizes.reserve(concat.getInputs().size());
  for (Value operand : concat.getInputs()) {
    operand_concat_dim_sizes.push_back(
        GetShape(operand).dimensions(concat_dim));
  }
  return ComputeConcatenateIndexing(output_shape.dimensions().size(),
                                    concat_dim, output_shape.dimensions(),
                                    operand_concat_dim_sizes, context);
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ConvolutionOp conv, int output_id) {
  MLIRContext* context = conv.getContext();
  auto input_shape = GetShape(conv.getLhs());
  auto kernel_shape = GetShape(conv.getRhs());
  auto output_shape = GetShape(conv.getResult());
  auto dnums = conv.getDimensionNumbers();
  size_t rank = output_shape.dimensions().size();

  // Collect sizes for input/output spatial dimensions.
  size_t spatial_rank = dnums.getInputSpatialDimensions().size();
  std::vector<int64_t> input_spatial_sizes(spatial_rank);
  std::vector<int64_t> kernel_spatial_sizes(spatial_rank);
  std::vector<int64_t> output_spatial_sizes(spatial_rank);
  for (int i = 0; i < spatial_rank; ++i) {
    input_spatial_sizes[i] =
        input_shape.dimensions(dnums.getInputSpatialDimensions()[i]);
    kernel_spatial_sizes[i] =
        kernel_shape.dimensions(dnums.getKernelSpatialDimensions()[i]);
    output_spatial_sizes[i] =
        output_shape.dimensions(dnums.getOutputSpatialDimensions()[i]);
  }

  SmallVector<int64_t> ones(spatial_rank, 1);
  auto strides = conv.getWindowStrides().value_or(ones);
  auto lhs_dilation = conv.getLhsDilation().value_or(ones);
  auto rhs_dilation = conv.getRhsDilation().value_or(ones);
  SmallVector<int64_t> padding_flat;
  if (conv.getPadding()) {
    for (auto val : conv.getPadding()->getValues<int64_t>()) {
      padding_flat.push_back(val);
    }
  } else {
    padding_flat.assign(spatial_rank * 2, 0);
  }

  // Indexing map for the input value (spatial dimensions only).
  // The dimension numbers in the resulting affine expressions have to be
  // remapped to correspond to the correct output dimensions.
  IndexingMap input_spatial_indexing = ComposeWindowIndexingMap(
      input_spatial_sizes, output_spatial_sizes, kernel_spatial_sizes, strides,
      rhs_dilation, lhs_dilation, padding_flat, context);
  std::vector<AffineExpr> replacement_dims(spatial_rank);
  for (int i = 0; i < spatial_rank; ++i) {
    replacement_dims[i] =
        mlir::getAffineDimExpr(dnums.getOutputSpatialDimensions()[i], context);
  }

  // Build affine expressions and constraints for input spatial dimensions.
  std::vector<AffineExpr> input_exprs(rank);
  for (int i = 0; i < spatial_rank; ++i) {
    input_exprs[dnums.getInputSpatialDimensions()[i]] =
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
    kernel_exprs[dnums.getKernelSpatialDimensions()[i]] =
        mlir::getAffineSymbolExpr(i, context);
  }
  AffineExpr dim_expr =
      mlir::getAffineDimExpr(dnums.getOutputFeatureDimension(), context);
  kernel_exprs[dnums.getKernelOutputFeatureDimension()] = dim_expr;

  // Build initial symbol ranges.
  std::vector<IndexingMap::Variable> input_symbols =
      input_spatial_indexing.GetRangeVars();
  std::vector<IndexingMap::Variable> kernel_symbols =
      RangeVarsFromTensorSizes(kernel_spatial_sizes);

  // Add symbol for input feature dimension.
  input_exprs[dnums.getInputFeatureDimension()] =
      mlir::getAffineSymbolExpr(input_symbols.size(), context);
  kernel_exprs[dnums.getKernelInputFeatureDimension()] =
      mlir::getAffineSymbolExpr(kernel_symbols.size(), context);

  int64_t input_group_size =
      kernel_shape.dimensions(dnums.getKernelInputFeatureDimension());
  Interval input_feature_range{0, input_group_size - 1};
  input_symbols.push_back(IndexingMap::Variable{input_feature_range});
  kernel_symbols.push_back(IndexingMap::Variable{input_feature_range});

  // With multiple feature groups, the input feature dimension is equally split.
  if (conv.getFeatureGroupCount() > 1) {
    AffineExpr& input_feature = input_exprs[dnums.getInputFeatureDimension()];
    int64_t output_group_size =
        output_shape.dimensions(dnums.getOutputFeatureDimension());
    int64_t feature_group_size =
        output_group_size / conv.getFeatureGroupCount();
    input_feature = dim_expr.floorDiv(feature_group_size) * input_group_size +
                    input_feature;
  }

  // With multiple batch groups, the input batch dimension is equally split.
  AffineExpr batch_dim_expr =
      mlir::getAffineDimExpr(dnums.getOutputBatchDimension(), context);
  if (conv.getBatchGroupCount() > 1) {
    int64_t batch_group_size =
        output_shape.dimensions(dnums.getOutputBatchDimension());
    AffineExpr batch_group_expr =
        mlir::getAffineSymbolExpr(input_symbols.size(), context);
    input_symbols.push_back(IndexingMap::Variable{
        {0, static_cast<int64_t>(conv.getBatchGroupCount()) - 1}});
    input_exprs[dnums.getInputBatchDimension()] =
        batch_group_expr * batch_group_size + batch_dim_expr;
  } else {
    input_exprs[dnums.getInputBatchDimension()] = batch_dim_expr;
  }

  // Indexing map for the input value.
  IndexingMap inputs_indexing(
      AffineMap::get(rank, input_symbols.size(), input_exprs, context),
      DimVarsFromTensorSizes(output_shape.dimensions()), input_symbols,
      /*rt_vars=*/{}, input_constraints);
  // We may need to simplify and remove unused symbols again, as the input
  // feature dimension size may be trivial.
  inputs_indexing.Simplify();
  inputs_indexing.RemoveUnusedSymbols();

  // Indexing map for the kernel value.
  IndexingMap kernel_indexing(
      AffineMap::get(rank, kernel_symbols.size(), kernel_exprs, context),
      DimVarsFromTensorSizes(output_shape.dimensions()), kernel_symbols,
      /*rt_vars=*/{});
  kernel_indexing.Simplify();
  kernel_indexing.RemoveUnusedSymbols();

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(2);
  indexing.indexing_maps[0].insert(OperandIndexing{inputs_indexing});
  indexing.indexing_maps[1].insert(OperandIndexing{kernel_indexing});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    DotGeneralOp dot_general, int output_id) {
  MLIRContext* context = dot_general.getContext();
  auto lhs_shape = GetShape(dot_general.getLhs());
  auto rhs_shape = GetShape(dot_general.getRhs());
  auto output_shape = GetShape(dot_general.getResult());
  auto dim_numbers = dot_general.getDotDimensionNumbers();

  auto lhs_batch_dims = dim_numbers.getLhsBatchingDimensions();
  auto rhs_batch_dims = dim_numbers.getRhsBatchingDimensions();
  auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
  auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();

  auto [lhs_map, rhs_map] = ComputeDotOperandsIndexing(
      lhs_shape.dimensions(), rhs_shape.dimensions(), output_shape.dimensions(),
      lhs_batch_dims, rhs_batch_dims, lhs_contracting_dims,
      rhs_contracting_dims, context);

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(2);
  indexing.indexing_maps[0].insert(OperandIndexing{lhs_map});
  indexing.indexing_maps[1].insert(OperandIndexing{rhs_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    DotOp dot, int output_id) {
  MLIRContext* context = dot.getContext();
  auto lhs_shape = GetShape(dot.getLhs());
  auto rhs_shape = GetShape(dot.getRhs());
  auto output_shape = GetShape(dot.getResult());

  // Following XLA's DotOp pattern:
  // For dot product: lhs[..., k] * rhs[k, ...] -> output[..., ...]
  // LHS: batch_dims + k (contracting)
  // RHS: k (contracting) + non_contracting
  int64_t lhs_rank = lhs_shape.dimensions().size();
  int64_t rhs_rank = rhs_shape.dimensions().size();
  int64_t output_rank = output_shape.dimensions().size();

  llvm::SmallVector<AffineExpr> lhs_exprs(lhs_rank);
  llvm::SmallVector<AffineExpr> rhs_exprs(rhs_rank);
  // LHS non-contracting dimensions map to output dims [0, output_rank-1)
  // For vector-matrix or matrix-vector: this is either batch dims or empty
  for (int64_t i = 0; i < lhs_rank - 1; ++i) {
    lhs_exprs[i] = mlir::getAffineDimExpr(i, context);
  }
  // RHS non-contracting dimensions map to output dims starting after LHS
  // For matrix-vector: output_rank may be < rhs_rank-1 (vector result)
  for (int64_t i = 0; i < rhs_rank - 1; ++i) {
    int64_t output_dim = (lhs_rank - 1) + i;
    if (output_dim < output_rank) {
      rhs_exprs[i + 1] = mlir::getAffineDimExpr(output_dim, context);
    } else {
      // Matrix-vector case: result is vector, extra RHS dims are implicit
      rhs_exprs[i + 1] = mlir::getAffineConstantExpr(0, context);
    }
  }

  // Contracting dimension (k): symbol for both LHS and RHS
  int64_t k_dim = lhs_shape.dimensions()[lhs_rank - 1];
  AffineExpr k_expr = mlir::getAffineSymbolExpr(0, context);
  lhs_exprs[lhs_rank - 1] = k_expr;
  rhs_exprs[0] = k_expr;
  IndexingMap lhs_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, 1, lhs_exprs, context),
      std::vector<int64_t>(output_shape.dimensions().begin(),
                           output_shape.dimensions().end()),
      {k_dim});
  IndexingMap rhs_map = IndexingMap::FromTensorSizes(
      AffineMap::get(output_rank, 1, rhs_exprs, context),
      std::vector<int64_t>(output_shape.dimensions().begin(),
                           output_shape.dimensions().end()),
      {k_dim});

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(2);
  indexing.indexing_maps[0].insert(OperandIndexing{lhs_map});
  indexing.indexing_maps[1].insert(OperandIndexing{rhs_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    DynamicSliceOp dynamic_slice, int output_id) {
  MLIRContext* context = dynamic_slice.getContext();
  auto input_shape = GetShape(dynamic_slice.getOperand());
  auto output_shape = GetShape(dynamic_slice.getResult());
  int64_t rank = output_shape.dimensions().size();

  std::vector<int64_t> dim_sizes(output_shape.dimensions().begin(),
                                 output_shape.dimensions().end());
  std::vector<IndexingMap::Variable> dim_vars;
  dim_vars.reserve(dim_sizes.size());
  for (auto size : dim_sizes) {
    dim_vars.push_back(IndexingMap::Variable{{0, size - 1}});
  }

  std::vector<AffineExpr> exprs;
  exprs.reserve(rank);
  std::vector<IndexingMap::Variable> rt_vars;
  std::vector<RuntimeVarIndexing> runtime_vars;

  // An empty affine map for scalar runtime variables.
  // Needed for indices_map construction below
  AffineMap empty_map = AffineMap::get(rank, 0, {}, context);

  for (auto [dim, slice_size] :
       llvm::enumerate(dynamic_slice.getSliceSizes())) {
    AffineExpr dim_expr = getAffineDimExpr(dim, context);
    Value rt_var_val = dynamic_slice.getStartIndices()[dim];
    int64_t max_index = input_shape.dimensions(dim) - slice_size;

    // Construct indexing map for the start index (scalar map keyed by output
    // dimensions) We reuse the scalar map logic: (d0...dN) -> ()
    IndexingMap rt_index_map = CreateScalarIndexingMap(output_shape, context);

    // Attempt constant folding/optimization
    RuntimeVarIndexing rt_indexing{rt_var_val, rt_index_map};
    Interval feasible_values{0, max_index};

    auto simplified_expr = OptimizeRTVar(rt_indexing, feasible_values, context);

    if (simplified_expr) {
      exprs.push_back(dim_expr + *simplified_expr);
    } else {
      exprs.push_back(dim_expr + getAffineSymbolExpr(rt_vars.size(), context));
      rt_vars.push_back(IndexingMap::Variable{{0, max_index}});
      runtime_vars.push_back(RuntimeVarIndexing{rt_var_val, rt_index_map});
    }
  }

  IndexingMap input_map{AffineMap::get(rank, rt_vars.size(), exprs, context),
                        dim_vars,
                        {},
                        rt_vars};

  OperandIndexing operand_indexing{input_map, runtime_vars};

  IndexingMap indices_map =
      IndexingMap::FromTensorSizes(empty_map, dim_sizes, {});

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(dynamic_slice.getNumOperands());
  indexing.indexing_maps[0].insert(operand_indexing);
  for (size_t i = 1; i < dynamic_slice.getNumOperands(); ++i) {
    indexing.indexing_maps[i].insert(OperandIndexing{indices_map});
  }
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    DynamicUpdateSliceOp dus, int output_id) {
  MLIRContext* context = dus.getContext();
  auto operand_shape = GetShape(dus.getOperand());
  auto update_shape = GetShape(dus.getUpdate());
  auto output_shape = GetShape(dus.getResult());
  int64_t rank = output_shape.dimensions().size();

  // Operand (input): identity mapping
  std::vector<AffineExpr> identity;
  identity.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    identity.push_back(getAffineDimExpr(dim, context));
  }
  std::vector<int64_t> dim_sizes(output_shape.dimensions().begin(),
                                 output_shape.dimensions().end());
  IndexingMap operand_map = IndexingMap::FromTensorSizes(
      AffineMap::get(rank, 0, identity, context), dim_sizes, {});

  // Update: (d0 - rt0, ..., d{N-1} - rt{N-1}) with runtime variables
  std::vector<AffineExpr> update_exprs;
  std::vector<IndexingMap::Variable> rt_vars;
  update_exprs.reserve(rank);
  rt_vars.reserve(rank);

  for (int64_t dim = 0; dim < rank; ++dim) {
    update_exprs.push_back(getAffineDimExpr(dim, context) -
                           getAffineSymbolExpr(dim, context));
    rt_vars.push_back(IndexingMap::Variable{
        {0, operand_shape.dimensions(dim) - update_shape.dimensions(dim)}});
  }

  std::vector<IndexingMap::Variable> dim_vars;
  dim_vars.reserve(dim_sizes.size());
  for (auto size : dim_sizes) {
    dim_vars.push_back(IndexingMap::Variable{{0, size - 1}});
  }

  IndexingMap update_map{
      AffineMap::get(rank, rank, update_exprs, context), dim_vars, {}, rt_vars};

  // Create RuntimeVarIndexing for offset operands
  std::vector<RuntimeVarIndexing> runtime_vars;
  runtime_vars.reserve(rank);
  AffineMap empty_map = AffineMap::get(rank, 0, {}, context);
  IndexingMap rt_index_map =
      IndexingMap::FromTensorSizes(empty_map, dim_sizes, {});

  for (auto offset_value : dus.getStartIndices()) {
    runtime_vars.push_back(RuntimeVarIndexing{offset_value, rt_index_map});
  }

  OperandIndexing update_indexing{update_map, runtime_vars};

  // Start indices: empty map
  IndexingMap indices_map =
      IndexingMap::FromTensorSizes(empty_map, dim_sizes, {});

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(dus.getNumOperands());
  indexing.indexing_maps[0].insert(OperandIndexing{operand_map});
  indexing.indexing_maps[1].insert(update_indexing);
  for (size_t i = 2; i < dus.getNumOperands(); ++i) {
    indexing.indexing_maps[i].insert(OperandIndexing{indices_map});
  }
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    GatherOp gather, int output_id) {
  MLIRContext* context = gather.getContext();
  auto operand_shape = GetShape(gather.getOperand());
  auto start_indices_shape = GetShape(gather.getStartIndices());
  auto output_shape = GetShape(gather.getResult());
  int64_t output_rank = output_shape.dimensions().size();

  auto dimension_numbers = gather.getDimensionNumbers();
  int64_t index_vector_dim = dimension_numbers.getIndexVectorDim();
  int64_t index_vector_length =
      start_indices_shape.dimensions(index_vector_dim);

  // Map for indices operand: (d0, ..., d{rank-1}) -> (d0, s0)
  // where s0 ranges over index vector dimension
  AffineExpr indices_id_dim = getAffineDimExpr(0, context);
  std::vector<int64_t> dim_sizes(output_shape.dimensions().begin(),
                                 output_shape.dimensions().end());
  std::vector<IndexingMap::Variable> dim_vars;
  dim_vars.reserve(dim_sizes.size());
  for (auto size : dim_sizes) {
    dim_vars.push_back(IndexingMap::Variable{{0, size - 1}});
  }

  IndexingMap indices_map{
      AffineMap::get(output_rank, 1,
                     {indices_id_dim, getAffineSymbolExpr(0, context)},
                     context),
      dim_vars,
      {IndexingMap::Variable{{0, index_vector_length - 1}}},
      /*rt_vars=*/{}};

  // Map for operand with runtime variables
  std::vector<AffineExpr> exprs;
  std::vector<RuntimeVarIndexing> runtime_vars;
  std::vector<IndexingMap::Variable> rt_vars;
  auto slice_sizes = gather.getSliceSizes();
  auto offset_dims = dimension_numbers.getOffsetDims();
  auto start_index_map = dimension_numbers.getStartIndexMap();

  exprs.reserve(operand_shape.dimensions().size());

  for (auto [operand_dim_id, slice_size] : enumerate(slice_sizes)) {
    int64_t output_dim_id = offset_dims[operand_dim_id];
    exprs.push_back(mlir::getAffineDimExpr(output_dim_id, context));

    // Check if this dimension is indexed by start_indices
    auto it = absl::c_find(start_index_map, operand_dim_id);
    if (it == start_index_map.end()) {
      continue;
    }

    int64_t start_index_map_idx = it - start_index_map.begin();

    // Create runtime variable for this index
    AffineMap rt_var_map = AffineMap::get(
        output_rank, 0,
        {indices_id_dim,
         mlir::getAffineConstantExpr(start_index_map_idx, context)},
        context);

    IndexingMap rt_index_map =
        IndexingMap::FromTensorSizes(rt_var_map, dim_sizes, {});

    int64_t upper_bound = operand_shape.dimensions(operand_dim_id) - slice_size;

    RuntimeVarIndexing rt_indexing{gather.getStartIndices(), rt_index_map};
    Interval feasible_values{0, upper_bound};

    if (auto simplified =
            OptimizeRTVar(rt_indexing, feasible_values, context)) {
      exprs.back() = exprs.back() + *simplified;
      continue;
    }

    runtime_vars.push_back(rt_indexing);
    rt_vars.push_back(IndexingMap::Variable{{0, upper_bound}});

    // Add runtime variable to expression
    exprs.back() = exprs.back() +
                   mlir::getAffineSymbolExpr(runtime_vars.size() - 1, context);
  }

  IndexingMap operand_map{
      AffineMap::get(output_rank, runtime_vars.size(), exprs, context),
      dim_vars,
      {},
      rt_vars};

  OperandIndexing operand_indexing{operand_map, runtime_vars};

  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(2);
  indexing.indexing_maps[0].insert(operand_indexing);
  indexing.indexing_maps[1].insert(OperandIndexing{indices_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    GetTupleElementOp gte, int output_id) {
  if (!dyn_cast<RankedTensorType>(gte.getResult().getType())) {
    return CreateUnknownIndexing(1);
  }
  auto output_shape = GetShape(gte.getResult());
  IndexingMap identity_map = IndexingMap::FromTensorSizes(
      AffineMap::getMultiDimIdentityMap(output_shape.dimensions().size(),
                                        gte.getContext()),
      std::vector<int64_t>(output_shape.dimensions().begin(),
                           output_shape.dimensions().end()),
      {});
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{identity_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    PadOp pad, int output_id) {
  MLIRContext* context = pad.getContext();
  auto output_shape = GetShape(pad.getResult());
  auto edge_padding_low = pad.getEdgePaddingLow();
  auto edge_padding_high = pad.getEdgePaddingHigh();
  auto interior_padding = pad.getInteriorPadding();
  IndexingMap input_indexing_map =
      ComputePadIndexingMap(output_shape.dimensions(), edge_padding_low,
                            edge_padding_high, interior_padding, context);
  IndexingMap padding_value_indexing_map =
      CreateScalarIndexingMap(output_shape, context);
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(2);
  indexing.indexing_maps[0].insert(OperandIndexing{input_indexing_map});
  indexing.indexing_maps[1].insert(OperandIndexing{padding_value_indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ReduceOp reduce, int output_id) {
  MLIRContext* context = reduce.getContext();

  auto input_shape = GetShape(reduce.getInputs()[0]);
  auto output_shape = GetShape(reduce.getResults()[0]);

  IndexingMap inputs_indexing_map = ComputeReduceInputIndexingMap(
      input_shape.dimensions(), output_shape.dimensions(),
      reduce.getDimensions(), context);

  IndexingMap inits_indexing_map =
      CreateScalarIndexingMap(output_shape, context);

  HloInstructionIndexing indexing;
  int64_t num_inputs = reduce.getInputs().size();
  int64_t num_operands = num_inputs + reduce.getInitValues().size();
  indexing.indexing_maps.resize(num_operands);

  for (int64_t id = 0; id < num_inputs; ++id) {
    indexing.indexing_maps[id].insert(OperandIndexing(inputs_indexing_map));
  }
  for (int64_t id = num_inputs; id < num_operands; ++id) {
    indexing.indexing_maps[id].insert(OperandIndexing(inits_indexing_map));
  }
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ReduceWindowOp reduce_window, int output_id) {
  MLIRContext* context = reduce_window.getContext();

  // Following XLA's ReduceWindowOp pattern:
  // Indexing for reduce-window with dilations and non-trivial padding
  // is represented as a composition using ComposeWindowIndexingMap

  auto input_shape = GetShape(reduce_window.getInputs()[0]);
  auto output_shape = GetShape(reduce_window.getResults()[0]);

  SmallVector<int64_t> default_dilations(input_shape.dimensions().size(), 1);
  SmallVector<int64_t> default_padding(input_shape.dimensions().size() * 2, 0);

  ArrayRef<int64_t> window_dilations =
      reduce_window.getWindowDilations()
          ? ArrayRef<int64_t>(*reduce_window.getWindowDilations())
          : ArrayRef(default_dilations);
  ArrayRef<int64_t> base_dilations =
      reduce_window.getBaseDilations()
          ? ArrayRef<int64_t>(*reduce_window.getBaseDilations())
          : ArrayRef(default_dilations);

  SmallVector<int64_t> padding_flat;
  if (reduce_window.getPadding()) {
    auto padding_attr = reduce_window.getPadding().value();
    for (auto val : padding_attr.getValues<int64_t>()) {
      padding_flat.push_back(val);
    }
  } else {
    padding_flat = default_padding;
  }

  // Indexing map for the input value
  IndexingMap inputs_indexing = ComposeWindowIndexingMap(
      input_shape.dimensions(), output_shape.dimensions(),
      reduce_window.getWindowDimensions(),
      reduce_window.getWindowStrides().value_or(
          reduce_window.getWindowDimensions()),
      window_dilations, base_dilations, padding_flat, context);

  // Indexing map for the init value
  IndexingMap inits_indexing_map =
      CreateScalarIndexingMap(output_shape, context);

  HloInstructionIndexing indexing;
  int64_t num_inputs = reduce_window.getInputs().size();
  int64_t num_operands = num_inputs + reduce_window.getInitValues().size();
  indexing.indexing_maps.resize(num_operands);

  for (int64_t id = 0; id < num_inputs; ++id) {
    indexing.indexing_maps[id].insert(OperandIndexing(inputs_indexing));
  }
  for (int64_t id = num_inputs; id < num_operands; ++id) {
    indexing.indexing_maps[id].insert(OperandIndexing(inits_indexing_map));
  }
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ReshapeOp reshape, int output_id) {
  MLIRContext* context = reshape.getContext();
  auto input_shape = GetShape(reshape.getOperand());
  auto output_shape = GetShape(reshape.getResult());
  IndexingMap indexing_map = GetBitcastMap(output_shape, input_shape, context);
  indexing_map.Simplify();
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    ReverseOp reverse, int output_id) {
  MLIRContext* context = reverse.getContext();
  auto output_shape = GetShape(reverse.getResult());
  IndexingMap indexing_map = ComputeReverseIndexingMap(
      output_shape.dimensions(), reverse.getDimensions(), context);
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    SliceOp slice, int output_id) {
  MLIRContext* context = slice.getContext();
  auto output_shape = GetShape(slice.getResult());
  IndexingMap indexing_map = ComputeSliceIndexingMap(
      output_shape.dimensions(), slice.getStartIndices(), slice.getStrides(),
      context);
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    TransposeOp transpose, int output_id) {
  MLIRContext* context = transpose.getContext();
  auto output_shape = GetShape(transpose.getResult());
  auto permutation = std::vector<int64_t>(transpose.getPermutation().begin(),
                                          transpose.getPermutation().end());
  IndexingMap indexing_map = IndexingMap::FromTensorSizes(
      ComputeTransposeIndexingMap(InversePermutation(permutation), context),
      output_shape.dimensions(), {});
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    TupleOp tuple_op, int output_id) {
  MLIRContext* context = tuple_op.getContext();
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(tuple_op->getNumOperands());
  for (auto [i, operand] : enumerate(tuple_op->getOperands())) {
    if (!dyn_cast<RankedTensorType>(operand.getType())) {
      continue;
    }
    auto operand_shape = GetShape(operand);
    IndexingMap identity_map = IndexingMap::FromTensorSizes(
        AffineMap::getMultiDimIdentityMap(operand_shape.dimensions().size(),
                                          context),
        std::vector<int64_t>(operand_shape.dimensions().begin(),
                             operand_shape.dimensions().end()),
        {});
    indexing.indexing_maps[i].insert(OperandIndexing{identity_map});
  }
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    mlir::mhlo::BitcastOp op, int output_id) {
  Shape input_shape = GetShape(op.getOperand());
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("source_layout")) {
    std::vector<int64_t> layout;
    for (const auto& val : attr.getValues<int64_t>()) {
      layout.push_back(val);
    }
    *input_shape.mutable_layout() = LayoutUtil::MakeLayout(layout);
  }

  Shape output_shape = GetShape(op.getResult());
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("result_layout")) {
    std::vector<int64_t> layout;
    for (const auto& val : attr.getValues<int64_t>()) {
      layout.push_back(val);
    }
    *output_shape.mutable_layout() = LayoutUtil::MakeLayout(layout);
  }
  IndexingMap indexing_map =
      GetBitcastMap(output_shape, input_shape, op.getContext());
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{indexing_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    mhlo::CopyOp op, int output_id) {
  auto output_shape = GetShape(op.getResult());
  IndexingMap identity_map = IndexingMap::FromTensorSizes(
      AffineMap::getMultiDimIdentityMap(output_shape.dimensions().size(),
                                        op.getContext()),
      std::vector<int64_t>(output_shape.dimensions().begin(),
                           output_shape.dimensions().end()),
      {});
  HloInstructionIndexing indexing;
  indexing.indexing_maps.resize(1);
  indexing.indexing_maps[0].insert(OperandIndexing{identity_map});
  return indexing;
}

[[maybe_unused]] HloInstructionIndexing ComputeOutputToInputIndexingImpl(
    mhlo::FusionOp op, int output_id) {
  auto& region = op.getRegion();
  if (region.empty()) {
    return CreateUnknownIndexing(op.getNumOperands());
  }

  auto& block = region.front();
  auto terminator = block.getTerminator();
  if (output_id >= terminator->getNumOperands()) {
    return CreateUnknownIndexing(op.getNumOperands());
  }

  HloInstructionIndexing fusion_indexing;
  fusion_indexing.indexing_maps.resize(op.getNumOperands());

  struct WorkItem {
    Value value;
    OperandIndexing indexing;
  };
  std::vector<WorkItem> worklist;

  // Start with the result of the fusion corresponding to output_id
  Value root_val = terminator->getOperand(output_id);
  Shape root_shape = GetShape(root_val);
  int64_t rank = root_shape.dimensions().size();

  IndexingMap identity_map = IndexingMap::FromTensorSizes(
      AffineMap::getMultiDimIdentityMap(rank, op.getContext()),
      std::vector<int64_t>(root_shape.dimensions().begin(),
                           root_shape.dimensions().end()),
      {});
  worklist.push_back({root_val, OperandIndexing{identity_map}});

  while (!worklist.empty()) {
    auto [val, current_indexing] = worklist.back();
    worklist.pop_back();

    if (current_indexing.IsUndefined()) {
      // Propagate undefined?
    }

    if (auto block_arg = dyn_cast<BlockArgument>(val)) {
      if (block_arg.getOwner() == &block) {
        int arg_idx = block_arg.getArgNumber();
        if (arg_idx < fusion_indexing.indexing_maps.size()) {
          fusion_indexing.indexing_maps[arg_idx].insert(current_indexing);
        }
      }
      continue;
    }

    Operation* producer = val.getDefiningOp();
    if (!producer) {
      continue;
    }

    // Recursive call to handle internal op
    int producer_result_idx = llvm::cast<mlir::OpResult>(val).getResultNumber();
    auto producer_indexing =
        ComputeOutputToInputIndexing(producer, producer_result_idx);

    for (size_t i = 0; i < producer->getNumOperands(); ++i) {
      Value operand = producer->getOperand(i);
      for (const auto& operand_indexing : producer_indexing.indexing_maps[i]) {
        if (operand_indexing.IsUndefined() || current_indexing.IsUndefined()) {
          worklist.push_back(
              {operand, OperandIndexing{IndexingMap::GetUndefined()}});
          continue;
        }
        // Note: ComposeOperandIndexing order is (Inner, Outer) aka (Consumer,
        // Producer) to compute Outer(Inner(x)).
        OperandIndexing composed =
            ComposeOperandIndexing(current_indexing, operand_indexing);
        if (!composed.IsUndefined()) {
          composed.Simplify();
          composed.RemoveUnusedSymbols();
        }
        worklist.push_back({operand, composed});
      }
    }
  }
  return fusion_indexing;
}

}  // namespace

HloInstructionIndexing ComputeOutputToInputIndexing(Operation* op,
                                                    int output_id) {
  MLIRContext* context = op->getContext();
  HloInstructionIndexing indexing =
      llvm::TypeSwitch<Operation*, HloInstructionIndexing>(op)
          // Operations with extracted helpers.
          .Case<AllGatherOp, BitcastConvertOp, BroadcastInDimOp, ConcatenateOp,
                ConvolutionOp, DotOp, DotGeneralOp, DynamicSliceOp,
                DynamicUpdateSliceOp, GatherOp, GetTupleElementOp, PadOp,
                ReduceOp, ReduceWindowOp, ReshapeOp, ReverseOp, SliceOp,
                TransposeOp, TupleOp,
                // MHLO ops.
                mhlo::BitcastOp, mhlo::CopyOp, mhlo::FusionOp>(
              [&](auto typed_op) {
                return ComputeOutputToInputIndexingImpl(typed_op, output_id);
              })

          // Elementwise identity operations, all operands use identity mapping.
          .Case<AddOp, SubtractOp, MulOp, DivOp, RemOp, MaxOp, MinOp, AndOp,
                OrOp, XorOp, AbsOp, NegOp, SignOp, CosineOp, SineOp, TanhOp,
                SqrtOp, RsqrtOp, ExpOp, Expm1Op, LogOp, Log1pOp, FloorOp,
                CeilOp, ConvertOp, SelectOp, ClampOp, CompareOp,
                PopulationCountOp, NotOp, IsFiniteOp, RoundNearestEvenOp,
                OptimizationBarrierOp, MapOp, SortOp>([&](Operation* op) {
            if (!dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
              return CreateUnknownIndexing(op->getNumOperands());
            }
            auto output_shape = GetShape(op->getResult(0));
            HloInstructionIndexing indexing = CreateElementwiseIndexing(
                op->getNumOperands(), output_shape, context);
            // Handle scalar broadcast for operands with no dimensions
            for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
              if (GetShape(operand).dimensions().empty()) {
                indexing.indexing_maps[i].clear();
                indexing.indexing_maps[i].insert(OperandIndexing{
                    CreateScalarIndexingMap(output_shape, context)});
              }
            }
            return indexing;
          })

          // Default:
          //  - IotaOp, ConstantOp, CreateTokenOp, AfterAllOp
          //  - unknown indexing for unsupported operations
          .Default([&](Operation* op) {
            return CreateUnknownIndexing(op->getNumOperands());
          });
  return indexing;
}

}  // namespace xla
