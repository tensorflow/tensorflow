/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/scatter_mlir.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace vector = ::mlir::vector;
namespace tensor = ::mlir::tensor;

using llvm::APFloat;
using llvm::APInt;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::DenseElementsAttr;
using mlir::getAffineDimExpr;
using mlir::getAffineSymbolExpr;
using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir_converter::CallTargetProvider;
using mlir_converter::EmitXlaLoopOp;
using mlir_converter::PartitionedComputations;
using mlir_converter::ProvideParameter;
using primitive_util::IsUnsignedIntegralType;

constexpr int64_t kNumWarpsPerBlock = 4;
constexpr int64_t kMaxVectorizedBits = 64;
constexpr int64_t kScatterOperandIndex = 0;
constexpr int64_t kScatterIndicesIndex = 1;
constexpr int64_t kScatterUpdateIndex = 2;

// Emit
// if (condition) {
//   updated_values = updated_values_fn();
//   yield updated_values;
// } else {
//   yield values;
// }
ValueRange EmitUpdateIf(
    ImplicitLocOpBuilder& b, Value condition, ValueRange values,
    llvm::function_ref<SmallVector<Value>(ImplicitLocOpBuilder&)>
        updated_values_fn) {
  return b
      .create<scf::IfOp>(
          condition,
          [&](OpBuilder& then_b, Location then_loc) -> void {
            ImplicitLocOpBuilder implicit_then_b(then_loc, then_b);
            then_b.create<scf::YieldOp>(then_loc,
                                        updated_values_fn(implicit_then_b));
          },
          [&](OpBuilder& else_b, Location else_loc) -> void {
            else_b.create<scf::YieldOp>(else_loc, values);
          })
      .getResults();
}

// Computes if the slice with the sizes `slice_shape` and the offsets `offsets`
// can be inserted into the operand with the shape `operand_shape`.
Value EmitBoundsCheck(ImplicitLocOpBuilder& b,
                      absl::Span<const int64_t> slice_shape,
                      absl::Span<const int64_t> operand_shape,
                      ValueRange offsets) {
  Value in_bounds = b.create<arith::ConstantIntOp>(1, b.getI1Type());
  for (auto [update_dim, operand_dim, offset] :
       llvm::zip(slice_shape, operand_shape, offsets)) {
    Value ub = b.create<arith::ConstantIndexOp>(operand_dim - update_dim);
    // One bounds check is enough even for signed indices: `sge 0` is
    // implied by `ule ub`, because `ub >= 0`.
    in_bounds = b.createOrFold<arith::AndIOp>(
        in_bounds,
        b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::ule, offset, ub));
  }
  return in_bounds;
}

Value EmitInequalityCheck(ImplicitLocOpBuilder& b, ValueRange lhs,
                          ValueRange rhs) {
  Value not_equal = b.create<arith::ConstantIntOp>(0, b.getI1Type());
  for (auto [lhs_elem, rhs_elem] : llvm::zip(lhs, rhs)) {
    not_equal = b.createOrFold<arith::OrIOp>(
        not_equal, b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                 lhs_elem, rhs_elem));
  }
  return not_equal;
}

Value UpdateIsInbounds(ImplicitLocOpBuilder& b, Value is_inbounds,
                       Value offsets_changed, ValueRange offsets,
                       absl::Span<const int64_t> slice_shape,
                       absl::Span<const int64_t> operand_shape) {
  return EmitUpdateIf(b, offsets_changed, is_inbounds,
                      [&](ImplicitLocOpBuilder& if_b) -> SmallVector<Value> {
                        return {EmitBoundsCheck(if_b, slice_shape,
                                                operand_shape, offsets)};
                      })
      .front();
}

SmallVector<Value> Pack(llvm::ArrayRef<ValueRange> ranges) {
  int64_t total_size = 0;
  for (auto& range : ranges) {
    total_size += range.size();
  }
  SmallVector<Value> result;
  result.reserve(total_size);
  for (auto range : ranges) {
    result.append(range.begin(), range.end());
  }
  return result;
}

SmallVector<ValueRange> Unpack(ValueRange range,
                               llvm::ArrayRef<int64_t> sizes) {
  int64_t total_size = 0;
  for (auto& size : sizes) {
    total_size += size;
  }
  assert(total_size == range.size());
  SmallVector<ValueRange> result;
  result.reserve(sizes.size());
  for (int64_t size : sizes) {
    result.push_back(range.take_front(size));
    range = range.drop_front(size);
  }
  return result;
}

// Pads the given values with zeros to the given container size.
SmallVector<Value, 4> PadWithZeros(ValueRange values, int64_t size,
                                   ImplicitLocOpBuilder& b) {
  SmallVector<Value, 4> padded_values(values.begin(), values.end());
  if (values.size() >= size) return padded_values;
  auto zero = b.create<arith::ConstantIndexOp>(0);
  for (int i = values.size(); i < size; ++i) {
    padded_values.push_back(zero);
  }
  return padded_values;
}

// Creates a new indexing map that is the same as `map` but with the range
// variable at `range_var_index` replaced with the new dimension variable at
// `dimension_{dim_var_size)`. Potentially, it can be moved to indexing_map.h.
IndexingMap ConvertRangeVariableToDimension(const IndexingMap& map,
                                            int64_t range_var_index) {
  auto* mlir_context = map.GetMLIRContext();

  AffineMap affine_map = map.GetAffineMap();
  // Update the affine map.
  SmallVector<AffineExpr, 4> symbol_replacements;
  symbol_replacements.reserve(affine_map.getNumSymbols());
  for (int i = 0; i < affine_map.getNumSymbols(); ++i) {
    if (i == range_var_index) {
      symbol_replacements.push_back(
          getAffineDimExpr(affine_map.getNumDims(), mlir_context));
    } else {
      symbol_replacements.push_back(
          getAffineSymbolExpr(i > range_var_index ? i - 1 : i, mlir_context));
    }
  }

  AffineMap converted_affine_map = affine_map.replaceDimsAndSymbols(
      {}, symbol_replacements, affine_map.getNumDims() + 1,
      affine_map.getNumSymbols() - 1);

  // Update the constraints.
  std::vector<std::pair<AffineExpr, Interval>> constraints;
  constraints.reserve(map.GetConstraintsCount());
  for (auto constraint : map.GetConstraints()) {
    constraints.push_back({constraint.first.replaceSymbols(symbol_replacements),
                           constraint.second});
  }
  // Update the variables.
  std::vector<IndexingMap::Variable> dims = map.GetDimVars();
  std::vector<IndexingMap::Variable> range_vars = map.GetRangeVars();
  std::vector<IndexingMap::Variable> rt_vars = map.GetRTVars();

  dims.push_back(range_vars[range_var_index]);
  range_vars.erase(range_vars.begin() + range_var_index);
  return IndexingMap{converted_affine_map, std::move(dims),
                     std::move(range_vars), std::move(rt_vars), constraints};
}

}  // namespace

class EmitterHelper {
 public:
  EmitterHelper(const ScatterDescription& description,
                const PartitionedComputations* computations,
                const CallTargetProvider* call_targets, FuncOp entry_function,
                const HloFusionInstruction& fusion)
      : description_(&description),
        entry_function_(entry_function),
        call_targets_(call_targets),
        root_computation_(&computations->FindPartitionedComputation(
            fusion.fused_instructions_computation())) {}

  Value GetOperandElement(ImplicitLocOpBuilder& b, ValueRange indices) const {
    return GetElement(b, kScatterOperandIndex, indices);
  }

  Value GetIndicesElement(ImplicitLocOpBuilder& b, ValueRange indices) const {
    return GetElement(b, kScatterIndicesIndex, indices);
  }

  Value GetUpdateElement(ImplicitLocOpBuilder& b, ValueRange indices) const {
    return GetElement(b, kScatterUpdateIndex, indices);
  }

  FuncOp GetReducer() const {
    return (*call_targets_)(
        description_->scatter->called_computations()[0]->root_instruction());
  }

  SmallVector<Value, 4> ExtractOffsets(ImplicitLocOpBuilder& b,
                                       Value slice_id) const;

  Value EmitScatterComputation(ImplicitLocOpBuilder& b, ValueRange indices,
                               Value update_elem, Value output_tensor) const;

  SmallVector<Value> WriteAccumulatedElementToOutput(
      ImplicitLocOpBuilder& b, Value accumulator,
      ValueRange accumulator_indices, ValueRange slice_indices,
      ValueRange offsets, Value output_tensor) const;

  Value WriteAccumulatorToOutput(ImplicitLocOpBuilder& b,
                                 Value write_to_output_required,
                                 ValueRange thread_and_block_ids, Value iv,
                                 const IndexingMap& slice_indexing,
                                 Value offsets_changed, ValueRange offsets,
                                 Value accumulator, Value output_tensor) const;

 private:
  Value GetElement(ImplicitLocOpBuilder& b, int operand_index,
                   ValueRange indices) const;

  const ScatterDescription* description_;
  FuncOp entry_function_;
  const mlir_converter::CallTargetProvider* call_targets_;
  const mlir_converter::PartitionedComputation* root_computation_;
};

SmallVector<Value, 4> EmitterHelper::ExtractOffsets(ImplicitLocOpBuilder& b,
                                                    Value slice_id) const {
  auto index_type = b.getIndexType();
  SmallVector<Value, 4> offsets;
  offsets.reserve(description_->index_vector_length);
  for (int i = 0; i < description_->index_vector_length; ++i) {
    SmallVector<Value, 4> indices_tensor_indices = {
        slice_id, b.create<arith::ConstantIndexOp>(i)};
    auto index = GetIndicesElement(b, indices_tensor_indices);
    index =
        IsUnsignedIntegralType(
            description_->scatter->scatter_indices()->shape().element_type())
            ? b.create<arith::IndexCastUIOp>(index_type, index).getResult()
            : b.create<arith::IndexCastOp>(index_type, index).getResult();
    offsets.push_back(index);
  }
  return offsets;
}

Value EmitterHelper::EmitScatterComputation(ImplicitLocOpBuilder& b,
                                            ValueRange indices,
                                            Value update_elem,
                                            Value output_tensor) const {
  FuncOp reducer = GetReducer();
  if (description_->scatter->unique_indices()) {
    auto operand_elem = GetOperandElement(b, indices);
    auto reduced_val = mlir_converter::InlineBlock(
        b, reducer.getBody().front(), {operand_elem, update_elem})[0];
    return b.create<tensor::InsertOp>(reduced_val, output_tensor, indices);
  }
  auto atomic_rmw = b.create<AtomicRMWOp>(output_tensor, indices);
  OpBuilder body_b = atomic_rmw.getBodyBuilder();
  auto reduced_val = mlir_converter::InlineBlock(
      body_b, reducer.getBody().front(),
      {atomic_rmw.getCurrentValue(), update_elem})[0];
  body_b.create<xla::YieldOp>(reducer->getLoc(), reduced_val);
  return atomic_rmw->getResult(0);
}

SmallVector<Value> EmitterHelper::WriteAccumulatedElementToOutput(
    ImplicitLocOpBuilder& b, Value accumulator, ValueRange accumulator_indices,
    ValueRange slice_indices, ValueRange offsets, Value output_tensor) const {
  Value accumulator_elem = b.create<vector::ExtractOp>(
      accumulator, mlir::getAsOpFoldResult(accumulator_indices));

  SmallVector<Value, 4> output_indices(offsets.begin(), offsets.end());
  for (int i = 0; i < output_indices.size(); ++i) {
    output_indices[i] =
        b.create<arith::AddIOp>(slice_indices[i + 1], output_indices[i]);
  }
  return {EmitScatterComputation(b, output_indices, accumulator_elem,
                                 output_tensor)};
}

Value EmitterHelper::WriteAccumulatorToOutput(
    ImplicitLocOpBuilder& b, Value write_to_output_required,
    ValueRange thread_and_block_ids, Value iv,
    const IndexingMap& slice_indexing, Value offsets_changed,
    ValueRange offsets, Value accumulator, Value output_tensor) const {
  SmallVector<Value> dims = Pack({thread_and_block_ids, iv});
  return EmitUpdateIf(
             b, write_to_output_required, output_tensor,
             [&](ImplicitLocOpBuilder& if_builder) -> SmallVector<Value> {
               return EmitXlaLoopOp(
                   if_builder, dims, output_tensor, slice_indexing,
                   [&](ImplicitLocOpBuilder& update_loop_b,
                       ValueRange accumulator_indices, ValueRange slice_indices,
                       ValueRange output_tensors) -> SmallVector<Value> {
                     return WriteAccumulatedElementToOutput(
                         update_loop_b, accumulator, accumulator_indices,
                         slice_indices, offsets, output_tensors.front());
                   });
             })
      .front();
}

Value EmitterHelper::GetElement(ImplicitLocOpBuilder& b, int operand_index,
                                ValueRange indices) const {
  return ProvideParameter(*root_computation_, description_->scatter,
                          operand_index, indices, *call_targets_,
                          entry_function_, b)[0];
}

MlirScatterFusion::MlirScatterFusion(const HloFusionAnalysis& analysis,
                                     const ScatterDescription& description,
                                     int64_t vector_size)
    : analysis_(analysis),
      description_(description),
      warp_size_(WarpSize(analysis_.device_info())),
      vector_size_(vector_size) {}

std::optional<IndexingMap> MlirScatterFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index, MLIRContext* ctx) const {
  CHECK(ScatterSimplifier::IsSimplifiedScatter(description_.scatter))
      << "Non-simplified HLO Scatter is not supported.";

  int64_t scatter_operand_count = description_.scatter->scatter_operand_count();
  // Scatter operands a packed in the following way:
  // Operand IDs [0, scatter_operand_count - 1] for `scatter operands`.
  // Operand ID  scatter_operand_count for `scatter indices`.
  // Operand IDs [scatter_operand_count + 1, 2 * scatter_operand_count] for
  // `scatter updates`.

  // For scatter operands we do not know the thread ID indexing.
  if (hero_operand_index < scatter_operand_count) {
    return std::nullopt;
  }

  bool is_indices_operand = hero_operand_index == scatter_operand_count;
  auto map = IndexingMap::GetUndefined();
  if (is_indices_operand) {
    ComputeIndexing(ctx, /*updates_map=*/nullptr, &map);
    return map;
  }
  ComputeIndexing(ctx, &map, /*indices_map=*/nullptr);
  return map;
}

std::vector<mlir_converter::EpilogueSpecification>
MlirScatterFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                MLIRContext* mlir_context) const {
  // We don't actually support epilogues for scatter, but this is how we tell
  // the base class that we don't want it to generate code for the scatter.
  return {mlir_converter::EpilogueSpecification::FromIdentityIndexing(
      &analysis_.fusion_hero(0).instruction(),
      &analysis_.fusion_root(0).instruction(), mlir_context)};
}

ScatterWithDistributedUpdates::ScatterWithDistributedUpdates(
    const HloFusionAnalysis& analysis, const ScatterDescription& description,
    int64_t vector_size)
    : MlirScatterFusion(analysis, description, vector_size) {
  // We have to make sure that there is no thread that processes elements of
  // two different update slice.
  auto launch_dimensions = CalculateLaunchDimensions(
      description_.update_shape, analysis_.device_info(),
      {static_cast<int>(vector_size_)});
  num_blocks_ = launch_dimensions.num_blocks();
  num_warps_ = CeilOfRatio(
      static_cast<int64_t>(launch_dimensions.num_threads_per_block()),
      warp_size_);
}

void ScatterWithDistributedUpdates::ComputeIndexing(
    MLIRContext* ctx, IndexingMap* updates_map,
    IndexingMap* indices_map) const {
  // Compute thread id mapping based on the first update operand.
  IndexingMap scatter_update_map = GetDefaultThreadIdIndexingMap(
      launch_dimensions(), vector_size_, description_.update_shape, ctx);

  // For scatter indices we project indexing for scatter updates and take the
  // first result of the affine map only, because they coincide.
  if (indices_map) {
    // Create a map from scatter update to scatter indices.
    *indices_map = IndexingMap{
        AffineMap::get(6, 1,
                       {scatter_update_map.GetAffineMap().getResult(0),
                        getAffineSymbolExpr(0, ctx)},
                       ctx),
        DimVarsFromGPUGrid({num_warps_ * warp_size_, 1, 1, num_blocks_, 1, 1}),
        RangeVarsFromTensorSizes({description_.index_vector_length}),
        /*rt_vars=*/{}};
    indices_map->Simplify();
  }
  if (updates_map) {
    *updates_map = std::move(scatter_update_map);
  }
}

absl::Status MlirScatterFusion::EmitEntryFunction(
    const PartitionedComputations& computations,
    const CallTargetProvider& call_targets, FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  EmitterHelper helper(description_, &computations, &call_targets,
                       entry_function, fusion);

  // Prepare the entry function.
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  auto thread_and_block_ids = EmitThreadAndBlockIds(b);
  Value output_tensor = entry_function.getArguments().back();

  // Compute indexing maps.
  MLIRContext* mlir_context = entry_function.getContext();
  IndexingMap updates_map = IndexingMap::GetUndefined();
  IndexingMap indices_map = IndexingMap::GetUndefined();
  ComputeIndexing(mlir_context, &updates_map, &indices_map);
  updates_map.Simplify();

  return EmitEntryFunctionImpl(b, helper, updates_map, indices_map,
                               thread_and_block_ids, output_tensor);
}

// Emits an inbounds check and a loop over updates inside it. Does not do any
// accumulation.
void EmitNaiveImplementation(ImplicitLocOpBuilder& b,
                             const ScatterDescription& description,
                             const EmitterHelper& helper,
                             const IndexingMap& updates_map,
                             const IndexingMap& indices_map,
                             ValueRange thread_and_block_ids,
                             Value output_tensor) {
  MLIRContext* mlir_context = b.getContext();
  auto thread_id_to_update_id_map = IndexingMap(
      AffineMap::get(6, 0, {updates_map.GetAffineMap().getResult(0)},
                     mlir_context),
      updates_map.GetDimVars(),
      /*range_vars = */ {}, /*rt vars = */ {});
  Value thread_id_to_index_id_value =
      mlir_converter::ApplyIndexing(thread_id_to_update_id_map,
                                    thread_and_block_ids, {}, b)
          .front();

  SmallVector<Value, 4> update_offsets =
      helper.ExtractOffsets(b, thread_id_to_index_id_value);

  Value in_bounds = EmitBoundsCheck(b, description.slice_shape,
                                    description.output_shape, update_offsets);

  Value predicated_update =
      EmitUpdateIf(
          b, in_bounds, {output_tensor},
          [&](ImplicitLocOpBuilder& nested_b) -> SmallVector<Value> {
            return EmitXlaLoopOp(
                nested_b, thread_and_block_ids, {output_tensor}, updates_map,
                [&](ImplicitLocOpBuilder& update_loop_b,
                    ValueRange symbol_values, ValueRange map_results,
                    ValueRange output_tensors) -> SmallVector<Value> {
                  // Extract update element.
                  auto update_elem =
                      helper.GetUpdateElement(update_loop_b, map_results);
                  auto output_indices = std::move(update_offsets);
                  int64_t output_rank = description.output_shape.size();
                  output_indices =
                      PadWithZeros(output_indices, output_rank, update_loop_b);
                  for (int i = 0; i < output_indices.size(); ++i) {
                    output_indices[i] = update_loop_b.create<arith::AddIOp>(
                        map_results[i + 1], output_indices[i]);
                  }
                  Value output_tensor = output_tensors.front();
                  Value updated_output = helper.EmitScatterComputation(
                      update_loop_b, output_indices, update_elem,
                      output_tensor);
                  return {updated_output};
                });
          })
          .front();
  b.create<ReturnOp>(predicated_update);
}

absl::Status ScatterWithDistributedUpdates::EmitEntryFunctionImpl(
    ImplicitLocOpBuilder& b, const EmitterHelper& helper,
    const IndexingMap& updates_map, const IndexingMap& indices_map,
    ValueRange thread_and_block_ids, Value output_tensor) const {
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "Settings for ScatterWithDistributedUpdates: \n"
                 << "vector_size_: " << vector_size_ << "\n"
                 << "num_warps_: " << num_warps_ << "\n"
                 << "num_blocks_: " << num_blocks_;
  }
  EmitNaiveImplementation(b, description_, helper, updates_map, indices_map,
                          thread_and_block_ids, output_tensor);
  return absl::OkStatus();
}

ScatterWithDistributedIndices::ScatterWithDistributedIndices(
    const HloFusionAnalysis& analysis, const ScatterDescription& description,
    int64_t vector_size, int64_t num_warps_per_slice,
    int64_t num_indices_per_warp)
    : MlirScatterFusion(analysis, description, vector_size),
      num_warps_per_slice_(num_warps_per_slice),
      num_indices_per_warp_(num_indices_per_warp) {
  num_warps_ = kNumWarpsPerBlock;
  num_blocks_ = CeilOfRatio(description.num_slices * num_warps_per_slice_,
                            num_indices_per_warp_ * num_warps_);
}

void ScatterWithDistributedIndices::ComputeIndexing(
    MLIRContext* ctx, IndexingMap* updates_map,
    IndexingMap* indices_map) const {
  // Compute thread id mapping based on the first update operand.
  auto thread_x = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx);
  auto block_x =
      getAffineDimExpr(KernelFusionInterface::kIndexingMapBlockIdxDims[0], ctx);
  auto warp_id = thread_x.floorDiv(warp_size_);
  auto slice_id =
      (block_x * num_warps_ + warp_id).floorDiv(num_warps_per_slice_);
  auto warp_id_in_slice =
      (block_x * num_warps_ + warp_id) % num_warps_per_slice_;
  auto lane_id = thread_x % warp_size_;
  auto index_id_loop = getAffineSymbolExpr(0, ctx);

  auto index_id_expr = slice_id * num_indices_per_warp_ + index_id_loop;
  std::pair<AffineExpr, Interval> index_id_constraint =
      std::make_pair(index_id_expr, Interval{0, description_.num_slices - 1});

  auto grid_vars =
      DimVarsFromGPUGrid({num_warps_ * warp_size_, 1, 1, num_blocks_, 1, 1});
  if (indices_map) {
    auto index_dim_loop = getAffineSymbolExpr(1, ctx);
    *indices_map = IndexingMap{
        AffineMap::get(6, 2, {index_id_expr, index_dim_loop}, ctx),
        grid_vars,
        {IndexingMap::Variable{{0, num_indices_per_warp_ - 1}, "index_id_loop"},
         IndexingMap::Variable{{0, description_.index_vector_length - 1},
                               "index_dim"}},
        /*rt_vars=*/{},
        {index_id_constraint}};

    indices_map->Simplify();
  }

  if (updates_map) {
    auto update_dim_loop = getAffineSymbolExpr(1, ctx);
    auto vector_id = getAffineSymbolExpr(2, ctx);
    auto num_elements_per_slice = Product(description_.slice_shape);

    auto linear_slice_index =
        warp_id_in_slice * warp_size_ * vector_size_ +
        update_dim_loop * vector_size_ * warp_size_ * num_warps_per_slice_ +
        lane_id * vector_size_ + vector_id;

    SmallVector<AffineExpr, 4> updates_indexing = {index_id_expr};
    updates_indexing.append(
        DelinearizeInBoundsIndex(linear_slice_index, description_.slice_shape));

    *updates_map = IndexingMap{
        AffineMap::get(6, 3, updates_indexing, ctx),
        grid_vars,
        {IndexingMap::Variable{{0, num_indices_per_warp_ - 1}, "index_id_loop"},
         IndexingMap::Variable{
             {0, CeilOfRatio(num_elements_per_slice,
                             num_warps_per_slice_ * warp_size_ * vector_size_) -
                     1},
             "update_loop"},
         IndexingMap::Variable{{0, vector_size_ - 1}, "vector_id"}},
        /*rt_vars=*/{},
        std::vector<std::pair<AffineExpr, Interval>>{
            index_id_constraint,
            std::make_pair(linear_slice_index,
                           Interval{0, num_elements_per_slice - 1})}};

    updates_map->Simplify();
  }
}

DenseElementsAttr GetShapedZeroConstantAttr(VectorType vector_type) {
  auto elem_type = vector_type.getElementType();
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type)) {
    std::vector<llvm::APFloat> values(
        vector_type.getNumElements(),
        APFloat::getZero(float_type.getFloatSemantics()));
    return DenseElementsAttr::get(vector_type, values);
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
    std::vector<llvm::APInt> values(
        vector_type.getNumElements(),
        APInt::getZero(int_type.getIntOrFloatBitWidth()));
    return DenseElementsAttr::get(vector_type, values);
  }
  llvm_unreachable("Unsupported vector element type");
}

Value ScatterWithDistributedIndices::InitializeAccumulator(
    ImplicitLocOpBuilder& b) const {
  auto elem_type =
      mlir_converter::PrimitiveTypeToMlirType(description_.elem_type, b);
  auto num_elements_per_slice = Product(description_.slice_shape);
  auto update_iterations_per_thread = CeilOfRatio(
      num_elements_per_slice, num_warps_per_slice_ * warp_size_ * vector_size_);
  auto accumulator_type =
      VectorType::get({update_iterations_per_thread, vector_size_}, elem_type);
  return b.create<arith::ConstantOp>(
      accumulator_type, GetShapedZeroConstantAttr(accumulator_type));
}

absl::Status ScatterWithDistributedIndices::EmitEntryFunctionImpl(
    ImplicitLocOpBuilder& b, const EmitterHelper& helper,
    const IndexingMap& updates_map, const IndexingMap& indices_map,
    ValueRange thread_and_block_ids, Value output_tensor) const {
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "Settings for ScatterWithDistributedIndices: \n"
                 << "vector_size_: " << vector_size_ << "\n"
                 << "num_warps_: " << num_warps_ << "\n"
                 << "num_blocks_: " << num_blocks_
                 << "num_warps_per_slice_: " << num_warps_per_slice_ << "\n"
                 << "num_indices_per_warp_: " << num_indices_per_warp_;
  }
  if (num_indices_per_warp_ == 1) {
    EmitNaiveImplementation(b, description_, helper, updates_map, indices_map,
                            thread_and_block_ids, output_tensor);
    return absl::OkStatus();
  }
  MLIRContext* mlir_context = b.getContext();

  auto thread_id_to_update_id_map = IndexingMap(
      AffineMap::get(6, 1, {updates_map.GetAffineMap().getResult(0)},
                     mlir_context),
      updates_map.GetDimVars(),
      /*range_vars = */ {updates_map.GetRangeVars().front()},
      /*rt vars = */ {});
  IndexingMap slice_indexing = ConvertRangeVariableToDimension(updates_map, 0);

  // Prepare loop initial values. Inits are packed as
  // [index_changed, is_inbounds, index_0,  ..., accumulator].
  Value is_inbounds_init = b.create<arith::ConstantIntOp>(0, b.getI1Type());
  std::vector<Value> indices_init(description_.index_vector_length,
                                  b.create<arith::ConstantIndexOp>(-1));
  Value accumulator_init = InitializeAccumulator(b);
  SmallVector<Value> inits =
      Pack({indices_init, is_inbounds_init, accumulator_init, output_tensor});

  auto loop_over_indices_fn =
      [&](ImplicitLocOpBuilder& nested_b, ValueRange ivs,
          ValueRange thread_id_to_index_id_value,
          ValueRange outer_iter_args) -> SmallVector<Value> {
    // Unpack the iter_args.
    SmallVector<ValueRange> iter_args_unpack =
        Unpack(outer_iter_args, {description_.index_vector_length, 1, 1, 1});
    ValueRange trimmed_offsets = iter_args_unpack[0];
    Value iter_is_inbounds = iter_args_unpack[1].front();
    Value iter_acc = iter_args_unpack[2].front();
    Value iter_output = iter_args_unpack[3].front();
    Value iter_slice_id = ivs.front();

    int64_t output_rank = description_.output_shape.size();
    SmallVector<Value> offsets =
        PadWithZeros(trimmed_offsets, output_rank, nested_b);

    auto new_trimmed_offsets =
        helper.ExtractOffsets(nested_b, thread_id_to_index_id_value.front());

    // Check if the offsets changed.
    Value offsets_changed =
        EmitInequalityCheck(nested_b, trimmed_offsets, new_trimmed_offsets);

    for (int i = 0; i < description_.index_vector_length; ++i) {
      new_trimmed_offsets[i] = nested_b.create<arith::SelectOp>(
          offsets_changed, new_trimmed_offsets[i], trimmed_offsets[i]);
    }

    auto new_offsets = PadWithZeros(new_trimmed_offsets, output_rank, nested_b);

    // Write accumulated values into the tensor if the offsets changed.
    Value is_not_first_iteration =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, iter_slice_id,
                                b.create<arith::ConstantIndexOp>(0));
    Value write_to_output_required = b.create<arith::AndIOp>(
        is_not_first_iteration,
        b.create<arith::AndIOp>(offsets_changed, iter_is_inbounds));
    iter_output = helper.WriteAccumulatorToOutput(
        b, write_to_output_required, thread_and_block_ids, iter_slice_id,
        slice_indexing, offsets_changed, offsets, iter_acc, iter_output);

    // Update `is_inbounds` if the offsets changed.
    Value new_is_inbounds = UpdateIsInbounds(
        nested_b, iter_is_inbounds, offsets_changed, new_offsets,
        description_.slice_shape, description_.output_shape);

    // Update accumulator and/or output.
    auto is_last_iteration = nested_b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, iter_slice_id,
        b.create<arith::ConstantIndexOp>(num_indices_per_warp_ - 1));

    SmallVector<Value> acc_and_output = {iter_acc, iter_output};
    auto loop_over_slices_fn =
        [&](ImplicitLocOpBuilder& update_loop_b, ValueRange accumulator_indices,
            ValueRange slice_indices,
            ValueRange inner_iter_args) -> SmallVector<Value> {
      Value acc_arg = inner_iter_args.front();
      Value output_arg = inner_iter_args.back();
      auto update_elem = helper.GetUpdateElement(update_loop_b, slice_indices);
      auto acc_ind_opfold = mlir::getAsOpFoldResult(accumulator_indices);
      // If the index changed, overwrite the accumulator element, otherwise
      // apply the scatter computation to reduce with the accumulator element.
      auto updated_accumulator =
          update_loop_b
              .create<scf::IfOp>(
                  offsets_changed,
                  [&](OpBuilder& then_b, Location then_loc) -> void {
                    Value updated_accumulator = then_b.create<vector::InsertOp>(
                        then_loc, update_elem, acc_arg, acc_ind_opfold);
                    then_b.create<scf::YieldOp>(then_loc, updated_accumulator);
                  },
                  [&](OpBuilder& else_b, Location else_loc) -> void {
                    ImplicitLocOpBuilder implicit_else_b(else_loc, else_b);
                    Value accumulator_elem =
                        implicit_else_b.create<vector::ExtractOp>(
                            acc_arg, acc_ind_opfold);
                    auto reduced_val = mlir_converter::InlineBlock(
                        implicit_else_b, helper.GetReducer().getBody().front(),
                        {accumulator_elem, update_elem})[0];
                    Value updated_ac = implicit_else_b.create<vector::InsertOp>(
                        reduced_val, acc_arg, acc_ind_opfold);
                    implicit_else_b.create<scf::YieldOp>(updated_ac);
                  })
              .getResult(0);
      // If this is the last index, that this warp has to process, then we write
      // to the output.
      auto updated_output =
          EmitUpdateIf(update_loop_b, is_last_iteration, output_arg,
                       [&](ImplicitLocOpBuilder& nested_b) {
                         return helper.WriteAccumulatedElementToOutput(
                             nested_b, updated_accumulator, accumulator_indices,
                             slice_indices, new_offsets, output_arg);
                       })
              .front();
      return {updated_accumulator, updated_output};
    };
    auto updated_accumulator_and_output =
        EmitUpdateIf(nested_b, new_is_inbounds, acc_and_output,
                     [&](ImplicitLocOpBuilder& if_b) {
                       return EmitXlaLoopOp(
                           if_b, Pack({thread_and_block_ids, iter_slice_id}),
                           acc_and_output, slice_indexing, loop_over_slices_fn);
                     });
    SmallVector<Value> updated_if_loop_results = Pack(
        {new_trimmed_offsets, new_is_inbounds, updated_accumulator_and_output});
    return updated_if_loop_results;
  };
  auto loop_over_indices_results =
      EmitXlaLoopOp(b, thread_and_block_ids, inits, thread_id_to_update_id_map,
                    loop_over_indices_fn);
  b.create<ReturnOp>(loop_over_indices_results.back());
  return absl::OkStatus();
}

ScatterDescription GetScatterDescription(const HloFusionAnalysis& analysis) {
  auto* hero = &analysis.fusion_hero(0).instruction();
  CHECK_NE(hero, nullptr);
  auto* scatter = Cast<HloScatterInstruction>(hero);
  auto indices_shape = scatter->scatter_indices()->shape();
  auto update_shape = scatter->scatter_updates().front()->shape();
  auto output_shape = scatter->scatter_operands().front()->shape();

  return ScatterDescription{
      scatter,
      indices_shape.dimensions(0),
      indices_shape.dimensions(1),
      output_shape.element_type(),
      update_shape,
      SmallVector<int64_t, 2>(update_shape.dimensions().begin() + 1,
                              update_shape.dimensions().end()),
      SmallVector<int64_t, 2>(output_shape.dimensions().begin(),
                              output_shape.dimensions().end()),
  };
}

// Compute the maximal vector size that can be used to process the given number
// of elements in a single slice.
int64_t GetSingleSliceVectorSize(int64_t num_elements_in_slice,
                                 int64_t max_vectorized_elements,
                                 int64_t warp_size) {
  int64_t vector_size =
      std::gcd(num_elements_in_slice, max_vectorized_elements);
  int64_t num_processed_elememts_per_warp = warp_size * vector_size;
  while (vector_size > 1 &&
         num_processed_elememts_per_warp > num_elements_in_slice) {
    vector_size /= 2;
    num_processed_elememts_per_warp /= 2;
  }
  return vector_size;
}

int64_t GetNumPossibleValidIndices(absl::Span<const int64_t> slice_shape,
                                   absl::Span<const int64_t> output_shape,
                                   int64_t index_vector_length) {
  int64_t num_possible_valid_indices = 1;
  for (int64_t i = 0; i < index_vector_length; ++i) {
    num_possible_valid_indices *= output_shape[i] - slice_shape[i] + 1;
  }
  return num_possible_valid_indices;
}

std::unique_ptr<MlirScatterFusion> CreateMlirScatterFusion(
    const HloFusionAnalysis& analysis) {
  auto description = GetScatterDescription(analysis);
  int64_t warp_size = WarpSize(analysis.device_info());
  int64_t num_elements_per_slice = Product(description.slice_shape);
  int64_t num_slices = description.num_slices;

  // Initialize the vector size with the maximum allowed vector size that does
  // not require masking/padding.
  int64_t elem_type_bits = primitive_util::BitWidth(description.elem_type);
  CHECK_EQ(kMaxVectorizedBits % elem_type_bits, 0);
  int64_t max_vectorized_elements = kMaxVectorizedBits / elem_type_bits;
  int64_t vector_size = GetSingleSliceVectorSize(
      num_elements_per_slice, max_vectorized_elements, warp_size);
  int64_t num_active_threads_per_warp =
      std::min(warp_size, num_elements_per_slice / vector_size);

  int64_t max_active_warps =
      kNumWarpsPerBlock * analysis.device_info().core_count();
  // For sorted scatter, we try to estimate the number of updates per warp by
  // computing the ratio of the number of the given updates to the number of the
  // possible valid indices. If we do not have multiple updates per warp, there
  // is no reason to use this algorithm.
  // TODO(b/385081952): Investigate why bf16 and f64 leads to incorrect results.
  if (description.scatter->indices_are_sorted() &&
      description.elem_type != BF16 && num_slices > 2 * max_active_warps) {
    int64_t num_indices_per_warp = CeilOfRatio(
        num_slices, GetNumPossibleValidIndices(
                        description.slice_shape, description.output_shape,
                        description.index_vector_length));
    int64_t num_warps_per_slice = CeilOfRatio(
        num_elements_per_slice, num_active_threads_per_warp * vector_size);
    if (num_indices_per_warp > 2 &&
        num_active_threads_per_warp > warp_size / 2) {
      return std::make_unique<ScatterWithDistributedIndices>(
          analysis, description, vector_size, num_warps_per_slice,
          num_indices_per_warp);
    }
  }
  // If we have enough data, we assign each warp to process a single
  // slice.
  if (num_slices > max_active_warps &&
      num_active_threads_per_warp > warp_size / 2) {
    return std::make_unique<ScatterWithDistributedIndices>(
        analysis, description, vector_size,
        /*num_warps_per_slice=*/1, /*num_indices_per_warp=*/1);
  }
  // Otherwise, we distribute the linearized updates tensor.
  vector_size =
      std::gcd(num_elements_per_slice,
               ComputeLoopFusionConfig(analysis, description.update_shape)
                   .unroll_factor);
  return std::make_unique<ScatterWithDistributedUpdates>(analysis, description,
                                                         vector_size);
}

}  // namespace gpu
}  // namespace xla
