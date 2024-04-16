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
#include "xla/service/gpu/fusions/reduction_mlir.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputations;

struct MlirReductionFusion::EmitterState {
  // Uses the given indexing map to reduce a subset of the inputs in a single
  // thread. The subset may be a single element.
  absl::StatusOr<SmallVector<Value>> EmitPerThreadReducedElements(
      const IndexingMap& input_indexing, const HloInstruction* hero,
      ValueRange inits);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  SmallVector<Value> AllocateSharedTiles(const HloInstruction* hero,
                                         absl::Span<const int64_t> shape);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  const MlirReductionFusion& owner;
  mlir::func::FuncOp entry_function;
  const HloFusionInstruction& fusion;
  const PartitionedComputations& computations;
  const mlir_converter::CallTargetProvider& call_target;
  mlir::ImplicitLocOpBuilder builder;
};

MlirReductionFusion::MlirReductionFusion(const HloFusionAnalysis& analysis)
    : ReductionFusionBase(analysis) {
  for (auto [index, hero] : llvm::enumerate(analysis.fusion_heroes())) {
    if (reduction_info().GetGroups().is_reduction_root[index]) {
      reduction_roots_[hero].push_back(index);
    }
  }

  for (const auto& [hero, _] : reduction_roots_) {
    reduction_heroes_.push_back(hero);
  }
}

bool MlirReductionFusion::IsSupported(const HloFusionAnalysis& analysis) {
  auto info = ReductionInfo::Create(analysis);
  return info.GetGroups().grouped_roots.size() == 1 &&
         !absl::c_linear_search(info.GetGroups().is_reduction_root, false) &&
         info.IsRaceFree();
}

std::optional<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogue(const HloFusionInstruction& fusion,
                                 mlir::MLIRContext* mlir_context) const {
  return mlir_converter::EpilogueSpecification::FromOutputIndexing(
      analysis(), reduction_heroes_, *this, mlir_context);
}

absl::Status MlirReductionFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  // Reduction groups will probably be implemented in a separate pass, since
  // they share nothing by definition.
  TF_RET_CHECK(reduction_info().GetGroups().grouped_roots.size() == 1)
      << "Only one reduction group is supported.";
  EmitterState state{*this,        entry_function,
                     fusion,       computations,
                     call_targets, {entry_function.getLoc(), entry_function}};
  state.builder.setInsertionPointToStart(entry_function.addEntryBlock());
  return EmitReduction(state);
}

absl::Status MlirReductionFusion::EmitReduction(EmitterState& state) const {
  CHECK(IsSupported(analysis()))
      << "Attempting to output code for an unsupported reduction";
  auto& builder = state.builder;
  const auto& tiling = reduction_info().GetTiling();

  // The number of warps working on one element in a row reduction.
  int num_warps_row = tiling.GetThreadsPerBlock()
                          [ReductionDimensions::kRowMinorReducedDimension] /
                      WarpSize();
  auto ctx = state.entry_function.getContext();

  auto zero = builder.create<mlir::arith::ConstantIndexOp>(0);
  auto lane_id = builder.create<mlir::gpu::LaneIdOp>();
  auto is_first_lane = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, lane_id, zero);
  auto thread_id = EmitThreadId(builder, 0);
  auto block_id = EmitBlockId(builder, 0);
  Value cstTrue = builder.create<mlir::arith::ConstantOp>(
      builder.getIntegerAttr(builder.getI1Type(), 1));

  auto thread_ids = mlir_converter::ApplyAffineMap(
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0,
          DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx),
                                   tiling.GetThreadsPerBlock(),
                                   tiling.GetThreadStrides()),
          ctx),
      {thread_id}, {}, builder);
  SmallVector<Value> thread_and_block_indices{thread_id, zero, zero,
                                              block_id,  zero, zero};

  auto warp_id = builder.create<mlir::arith::DivUIOp>(
      reduction_info().IsRowReduction()
          ? thread_ids[ReductionDimensions::kRowMinorReducedDimension]
          : thread_id,
      builder.create<mlir::arith::ConstantIndexOp>(WarpSize()));

  auto output_args = state.entry_function.getArguments().drop_front(
      state.fusion.fused_parameters().size());

  std::vector<int64_t> shared_tile_size;
  SmallVector<Value> shared_write_indices;
  SmallVector<Value> shared_read_indices;
  Value shared_write_condition = cstTrue;
  Value shared_read_condition = cstTrue;
  if (!reduction_info().IsRowReduction()) {
    shared_tile_size = {WarpSize(), WarpSize() + 1};
    shared_write_indices = {lane_id, warp_id};
    shared_read_indices = {warp_id, lane_id};
  } else if (reduction_info().GetRowsPerWarp() == 1 && num_warps_row > 1) {
    auto kKept = ReductionDimensions::kRowKeptDimension;
    shared_tile_size = {tiling.GetThreadsPerBlock()[kKept], num_warps_row};
    shared_write_condition = is_first_lane;
    shared_read_condition = builder.create<mlir::arith::CmpIOp>(
        mlir::arith::CmpIPredicate::ult,
        thread_ids[ReductionDimensions::kRowMinorReducedDimension],
        builder.create<mlir::arith::ConstantIndexOp>(num_warps_row));
    shared_write_indices = {thread_ids[kKept], warp_id};
    shared_read_indices = {thread_ids[kKept], lane_id};
  }
  bool use_shared = !shared_tile_size.empty();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> root_output_indices;
  root_output_indices.resize(analysis().fusion_roots().size());
  for (const auto& [hero, root_ids] : reduction_roots_) {
    auto hero_indices = mlir_converter::ApplyAffineMap(
        ComputeThreadIdToOutputIndexing(root_ids.front(), ctx)->GetAffineMap(),
        thread_and_block_indices, {}, builder);
    auto root_indices = mlir_converter::ApplyAffineMap(
        ComputeEpilogueInputToOutputIndexing(hero, ctx).GetAffineMap(),
        hero_indices, {}, builder);
    for (auto root_id : root_ids) {
      root_output_indices[root_id] = root_indices;
    }
  }

  if (analysis().fusion_roots().size() == 1 &&
      analysis().fusion_roots().front()->shape().IsTuple()) {
    // This is a variadic reduce. The root indices are the same for all
    // elements.
    int num_elements =
        analysis().fusion_roots().front()->shape().tuple_shapes_size();
    root_output_indices.reserve(num_elements);
    for (int i = 0; i < num_elements - 1; ++i) {
      root_output_indices.push_back(root_output_indices.front());
    }
  }

  auto thread_has_output =
      mlir_converter::CheckConstraints(*ComputeThreadIdToOutputIndexing(0, ctx),
                                       thread_and_block_indices, {}, builder);

  llvm::DenseMap<const HloInstruction*, SmallVector<Value>> inits;
  for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
    int num_inputs = hero->operand_count() / 2;
    const auto& computation =
        state.computations.FindPartitionedComputation(hero->parent());
    inits[hero] = ProvideParameterRange(
        computation.FindSubgraph(hero), hero, num_inputs, num_inputs, {},
        state.call_target, state.entry_function, builder);
  }

  auto evaluate_epilogue =
      [&](SmallVector<SmallVector<Value>> results) -> mlir::ValueRange {
    if (!state.computations.epilogue()) {
      return results.front();
    }

    llvm::SmallVector<Value> hero_values(reduction_heroes_.size());
    const auto& injected = state.computations.epilogue()->injected_values;
    for (auto [hero, result] : llvm::zip(reduction_heroes_, results)) {
      CHECK(result.size() == 1)
          << "Epilogue fusions are not supported with variadic reduce.";
      hero_values[injected.at(hero)] = result.front();
    }

    llvm::SmallVector<Value> indices = EmitThreadAndBlockIds(builder);
    int num_symbols =
        state.computations.epilogue()->root_indexing.front().getNumSymbols();
    for (int i = 0; i < num_symbols; ++i) {
      indices.push_back(zero);
    }

    return EmitEpilogue(state.computations, state.entry_function, hero_values,
                        indices, builder);
  };

  SmallVector<Value> updated_outputs;
  SmallVector<llvm::SmallVector<Value>> results;
  for (auto* hero : reduction_heroes_) {
    auto input_indexing = ComputeThreadIdToInputIndexing(
        reduction_roots_.at(hero).front(), 0, ctx);
    TF_ASSIGN_OR_RETURN(
        auto accumulated,
        state.EmitPerThreadReducedElements(*input_indexing, hero, inits[hero]));

    // In row reductions, we can do a warp shuffle before writing to shared
    // memory. In column reductions, the members of the warp process different
    // output elements, so we need to transpose first.
    if (reduction_info().IsRowReduction()) {
      auto reducer = state.GetReducer(hero);
      int max_dist = WarpSize() / 2 / reduction_info().GetRowsPerWarp();
      accumulated =
          builder.create<ShuffleReduceOp>(reducer, accumulated, max_dist)
              .getResults();
    }

    results.push_back(accumulated);
  }

  if (use_shared) {
    // Write results to shared memory.
    for (auto [hero, result] : llvm::zip(reduction_heroes_, results)) {
      auto dest = state.AllocateSharedTiles(hero, shared_tile_size);
      for (auto [value, output] : llvm::zip(result, dest)) {
        updated_outputs.push_back(builder.create<PredicatedInsertOp>(
            shared_write_condition, value, output, shared_write_indices));
      }
    }
  } else {
    // Evaluate the epilogue, if there is one.
    auto result_scalars = evaluate_epilogue(results);
    for (auto [value, output, indices] :
         llvm::zip(result_scalars, output_args, root_output_indices)) {
      updated_outputs.push_back(builder.create<PredicatedInsertOp>(
          thread_has_output, value, output, indices));
    }
    builder.create<mlir::func::ReturnOp>(updated_outputs);
    return absl::OkStatus();
  }

  // Wait for the entire tile to be written.
  auto shared_tiles = builder
                          .create<SyncThreadsOp>(
                              mlir::TypeRange(updated_outputs), updated_outputs)
                          .getResults();
  auto write_outputs = [&](mlir::OpBuilder then_builder, mlir::Location loc) {
    results.clear();
    mlir::ImplicitLocOpBuilder b(loc, then_builder);
    int tile_index = 0;
    llvm::SmallVector<Value> updated_outputs;
    for (auto* hero : reduction_heroes_) {
      // Load from shared memory.
      SmallVector<Value> reduced;
      for (auto init : inits[hero]) {
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(b.create<PredicatedExtractOp>(
                               shared_read_condition, init,
                               shared_tiles[tile_index++], shared_read_indices)
                              .getResult());
      }

      reduced = builder
                    .create<ShuffleReduceOp>(state.GetReducer(hero), reduced,
                                             WarpSize() / 2)
                    .getResults();
      results.push_back(reduced);
    }

    auto result_scalars = evaluate_epilogue(results);

    for (auto [output_value, dest, indices] :
         llvm::zip(result_scalars, output_args, root_output_indices)) {
      updated_outputs.push_back(b.create<PredicatedInsertOp>(
          thread_has_output, output_value, dest, indices));
    }
    b.create<mlir::scf::YieldOp>(loc, updated_outputs);
  };

  auto warp_writes = reduction_info().IsRowReduction()
                         ? builder.create<mlir::arith::CmpIOp>(
                               mlir::arith::CmpIPredicate::eq, warp_id, zero)
                         : cstTrue;
  auto written = builder.create<mlir::scf::IfOp>(
      warp_writes, write_outputs, [&](mlir::OpBuilder b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(loc, output_args);
      });
  builder.create<mlir::func::ReturnOp>(written.getResults());

  return absl::OkStatus();
}

absl::StatusOr<SmallVector<Value>>
MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    const IndexingMap& input_indexing, const HloInstruction* hero,
    ValueRange inits) {
  auto body_builder = [&](ValueRange outputs, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto indices = mlir_converter::ApplyAffineMap(
        input_indexing.GetAffineMap(), dim_values, symbol_values, builder);
    auto operands = FusionParams();
    absl::c_copy(indices, std::back_inserter(operands));
    auto values = ProvideParameterRange(computations.FindSubgraph(hero), hero,
                                        0, hero->operand_count() / 2, indices,
                                        call_target, entry_function, builder);

    SmallVector<Value> reduce_args = outputs;
    reduce_args.append(values.begin(), values.end());
    return builder.create<PureCallOp>(GetReducer(hero), reduce_args)
        .getResults();
  };
  return owner.EmitThreadLoopNest(builder, inits, input_indexing, body_builder);
}

SmallVector<Value> MlirReductionFusion::EmitterState::AllocateSharedTiles(
    const HloInstruction* hero, absl::Span<const int64_t> shape) {
  SmallVector<Value> tiles;
  for (int i = 0; i < hero->operand_count() / 2; ++i) {
    tiles.push_back(
        builder.create<AllocateSharedOp>(mlir_converter::TensorShapeToMlirType(
            ShapeUtil::MakeShapeWithDescendingLayout(
                hero->operand(i)->shape().element_type(), shape),
            builder)));
  }
  return tiles;
}

}  // namespace gpu
}  // namespace xla
