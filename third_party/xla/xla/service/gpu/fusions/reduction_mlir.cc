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
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
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

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputations;

using HloValueMap =
    absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>;

struct MlirReductionFusion::EmitterState {
  // Uses the given indexing map to reduce a subset of the inputs in a single
  // thread. The subset may be a single element.
  HloValueMap EmitPerThreadReducedElements(const HloValueMap& inits);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  SmallVector<Value> AllocateSharedTiles(const HloInstruction* hero,
                                         absl::Span<const int64_t> shape);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  int OutputIndex(const HloInstruction* root, int result_index) {
    if (root->shape().IsTuple()) {
      // If the root is a tuple, that means we're dealing with a variadic
      // reduction. Variadic reductions have no epilogues or side outputs.
      return result_index;
    }

    CHECK_EQ(result_index, 0);
    return absl::c_find(owner.analysis().fusion_roots(), root) -
           owner.analysis().fusion_roots().begin();
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
  absl::flat_hash_set<const HloInstruction*> seen_heroes;
  const auto& is_reduction_root =
      reduction_info().GetGroups().is_reduction_root;
  first_reduction_root_index_ = std::distance(
      is_reduction_root.begin(), absl::c_find(is_reduction_root, true));
  for (auto [root, hero, is_reduction] :
       llvm::zip(analysis.fusion_roots(), analysis.fusion_heroes(),
                 reduction_info().GetGroups().is_reduction_root)) {
    (is_reduction ? reduction_roots_ : side_output_roots_).push_back(root);
    if (is_reduction && seen_heroes.insert(hero).second) {
      reduction_heroes_.push_back(hero);
    }
  }
}

bool MlirReductionFusion::IsSupported(const HloFusionAnalysis& analysis) {
  auto info = ReductionInfo::Create(analysis);
  return info.GetGroups().grouped_roots.size() == 1 && info.IsRaceFree();
}

std::vector<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  mlir::MLIRContext* mlir_context) const {
  return {mlir_converter::EpilogueSpecification::FromOutputIndexing(
      analysis(), reduction_heroes_, reduction_roots_, *this, mlir_context)};
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
  Value cst_true = builder.create<mlir::arith::ConstantOp>(
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

  std::vector<int64_t> shared_tile_size;
  SmallVector<Value> shared_write_indices;
  SmallVector<Value> shared_read_indices;
  Value shared_write_condition = cst_true;
  Value shared_read_condition = cst_true;
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

  auto thread_has_output = mlir_converter::CheckConstraints(
      *ComputeThreadIdToOutputIndexing(first_reduction_root_index_, ctx),
      thread_and_block_indices, {}, builder);

  HloValueMap inits;
  llvm::SmallVector<Value> outputs =
      mlir::ValueRange(state.entry_function.getArguments().drop_front(
          state.fusion.fused_parameters().size()));
  HloValueMap root_output_indices;
  llvm::SmallVector<Value> epilogue_input_dims;
  const auto& epilogue = state.computations.epilogues().front();
  epilogue_input_dims = EmitThreadAndBlockIds(builder);
  llvm::SmallVector<Value> epilogue_input_symbols(
      epilogue.root_indexing.front().getNumSymbols(), zero);
  for (auto [index, root] : llvm::enumerate(epilogue.roots)) {
    root_output_indices[root] = mlir_converter::ApplyAffineMap(
        epilogue.root_indexing[index], epilogue_input_dims,
        epilogue_input_symbols, builder);
  }

  for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
    int arity = hero->operand_count() / 2;
    const auto& computation =
        state.computations.FindPartitionedComputation(hero->parent());
    inits[hero] =
        ProvideParameterRange(computation, hero, arity, arity, {},
                              state.call_target, state.entry_function, builder);
  }

  auto evaluate_epilogue = [&](const HloValueMap& results,
                               llvm::SmallVector<Value> outputs) {
    auto epilogue_indices = epilogue_input_dims;
    epilogue_indices.append(epilogue_input_symbols);
    auto values =
        EmitEpilogue(/*epilogue_index=*/0, state.computations,
                     state.entry_function, results, epilogue_indices, builder);
    const auto& epilogue = state.computations.epilogues().front();
    for (auto root : epilogue.roots) {
      for (auto [result_index, result] : llvm::enumerate(values.at(root))) {
        auto& output = outputs[state.OutputIndex(root, result_index)];
        output = builder.create<PredicatedInsertOp>(
            thread_has_output, result, output, root_output_indices[root]);
      }
    }
    return outputs;
  };

  auto accumulated = state.EmitPerThreadReducedElements(inits);
  for (auto root : side_output_roots_) {
    outputs[state.OutputIndex(root, 0)] = accumulated[root].front();
  }

  // In row reductions, we can do a warp shuffle before writing to shared
  // memory. In column reductions, the members of the warp process different
  // output elements, so we need to transpose first.
  if (reduction_info().IsRowReduction()) {
    for (auto* hero : reduction_heroes_) {
      auto reducer = state.GetReducer(hero);
      int max_dist = WarpSize() / 2 / reduction_info().GetRowsPerWarp();
      accumulated[hero] =
          builder.create<ShuffleReduceOp>(reducer, accumulated[hero], max_dist)
              .getResults();
    }
  }

  if (!use_shared) {
    builder.create<mlir::func::ReturnOp>(
        evaluate_epilogue(accumulated, std::move(outputs)));
    return absl::OkStatus();
  }

  SmallVector<Value> shared_tiles;
  // Write results to shared memory.
  for (auto hero : reduction_heroes_) {
    const auto& result = accumulated[hero];
    auto dest = state.AllocateSharedTiles(hero, shared_tile_size);
    for (auto [value, output] : llvm::zip(result, dest)) {
      shared_tiles.push_back(builder.create<PredicatedInsertOp>(
          shared_write_condition, value, output, shared_write_indices));
    }
  }

  // Wait for the entire tile to be written.
  auto synced_tiles =
      builder.create<SyncThreadsOp>(mlir::TypeRange(shared_tiles), shared_tiles)
          .getResults();
  auto write_outputs = [&](mlir::OpBuilder then_builder, mlir::Location loc) {
    mlir::ImplicitLocOpBuilder b(loc, then_builder);
    int tile_index = 0;
    for (auto* hero : reduction_heroes_) {
      // Load from shared memory.
      SmallVector<Value> reduced;
      for (auto init : inits[hero]) {
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(b.create<PredicatedExtractOp>(
                               shared_read_condition, init,
                               synced_tiles[tile_index++], shared_read_indices)
                              .getResult());
      }
      accumulated[hero] = builder
                              .create<ShuffleReduceOp>(state.GetReducer(hero),
                                                       reduced, WarpSize() / 2)
                              .getResults();
    }

    b.create<mlir::scf::YieldOp>(loc, evaluate_epilogue(accumulated, outputs));
  };

  auto warp_writes = reduction_info().IsRowReduction()
                         ? builder.create<mlir::arith::CmpIOp>(
                               mlir::arith::CmpIPredicate::eq, warp_id, zero)
                         : cst_true;
  auto written = builder.create<mlir::scf::IfOp>(
      warp_writes, write_outputs, [&](mlir::OpBuilder b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(loc, outputs);
      });
  builder.create<mlir::func::ReturnOp>(written.getResults());

  return absl::OkStatus();
}

HloValueMap MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    const HloValueMap& inits) {
  const auto& tiling = owner.reduction_info().GetTiling();
  auto tile_indexing = GetIndexingMapForTiling(tiling, builder.getContext());

  SmallVector<Value> iter_arg_inits;
  ValueRange output_args = entry_function.getArguments().drop_front(
      fusion.fused_parameters().size());
  for (auto [is_reduction, hero, output] :
       llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                 owner.analysis().fusion_heroes(), output_args)) {
    if (is_reduction) {
      iter_arg_inits.append(inits.at(hero));
    } else {
      iter_arg_inits.push_back(output);
    }
  }

  const auto& computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());

  auto body_builder = [&](ValueRange iter_args, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto tile_indices = mlir_converter::ApplyAffineMap(
        tile_indexing.GetAffineMap(), dim_values, symbol_values, builder);

    llvm::SmallVector<Value> results;
    struct SideOutput {
      Value tensor;
      llvm::SmallVector<Value> indices;
      Value scalar;
      int result_index;
    };
    llvm::SmallVector<SideOutput> side_outputs;
    int start = 0;
    for (auto [is_reduction, hero] :
         llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                   owner.analysis().fusion_heroes())) {
      const xla::Shape& input_shape =
          is_reduction ? hero->operand(0)->shape() : hero->shape();
      llvm::SmallVector<Value> input_indices = mlir_converter::ApplyAffineMap(
          GetBitcastMap(tiling.GetXlaShape(), input_shape, builder.getContext())
              .GetAffineMap(),
          tile_indices, {}, builder);
      if (is_reduction) {
        int num_outs = hero->operand_count() / 2;
        auto values = ProvideParameterRange(
            computations.FindPartitionedComputation(hero->parent()), hero, 0,
            num_outs, input_indices, call_target, entry_function, builder);
        SmallVector<Value> reduce_args = iter_args.slice(start, num_outs);
        reduce_args.append(values);
        absl::c_copy(builder.create<PureCallOp>(GetReducer(hero), reduce_args)
                         .getResults(),
                     std::back_inserter(results));
        start += num_outs;
      } else {
        auto* root_tuple = fusion.fused_expression_root();
        Value value = mlir_converter::ProvideParameter(
            computation, root_tuple, root_tuple->operand_index(hero),
            input_indices, call_target, entry_function, builder);
        // Tensor insertions turn into writes, so they have to happen in the
        // end. This could be considered a bug in the lowering, but since we
        // don't have bufferization, we need to handle it here.
        side_outputs.push_back(
            {iter_args[start], std::move(input_indices), value, start});
        results.push_back(nullptr);
        ++start;
      }
    }
    for (auto& side_output : side_outputs) {
      results[side_output.result_index] =
          builder.create<mlir::tensor::InsertOp>(
              side_output.scalar, side_output.tensor, side_output.indices);
    }
    return results;
  };

  auto results = owner.EmitThreadLoopNest(builder, iter_arg_inits,
                                          tile_indexing, body_builder);
  mlir::ValueRange result_range = results;
  HloValueMap results_per_hero;
  for (auto [is_reduction, hero] :
       llvm::zip(owner.reduction_info().GetGroups().is_reduction_root,
                 owner.analysis().fusion_heroes())) {
    int num_outs =
        hero->shape().IsTuple() ? hero->shape().tuple_shapes_size() : 1;
    results_per_hero[hero] = result_range.take_front(num_outs);
    result_range = result_range.drop_front(num_outs);
  }
  return results_per_hero;
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
