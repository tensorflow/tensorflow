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
#include <functional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace ma = mlir::arith;
using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputations;

using HloValueMap =
    absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>;

struct MlirReductionFusion::EmitterState {
  EmitterState(const MlirReductionFusion& owner,
               mlir::func::FuncOp entry_function,
               const HloFusionInstruction& fusion,
               const PartitionedComputations& computations,
               const mlir_converter::CallTargetProvider& call_target)
      : owner(owner),
        entry_function(entry_function),
        fusion(fusion),
        computations(computations),
        call_target(call_target),
        builder(entry_function.getLoc(), entry_function),
        computation(computations.FindPartitionedComputation(
            fusion.fused_instructions_computation())) {
    int index = 0;
    for (const auto& root : owner.analysis().fusion_roots()) {
      fusion_result_index_starts[&root.instruction()] = index;
      index += root.shape().IsTuple() ? root.shape().tuple_shapes_size() : 1;
    }
  }

  // Reduces a subset of the inputs in a single thread. Also writes side outputs
  // to the output tensors. The map contains the reduced values for reductions
  // and the written tensors for side outputs.
  HloValueMap EmitPerThreadReducedElements(int group_id,
                                           const HloValueMap& inits);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  SmallVector<Value> AllocateSharedTiles(
      absl::Span<const HloInstruction* const> heroes,
      absl::Span<const int64_t> shape);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  int OutputIndex(const HloInstruction* root, int result_index) {
    return fusion_result_index_starts[root] + result_index;
  }

  const MlirReductionFusion& owner;
  mlir::func::FuncOp entry_function;
  const HloFusionInstruction& fusion;
  const PartitionedComputations& computations;
  const mlir_converter::CallTargetProvider& call_target;
  ImplicitLocOpBuilder builder;
  const mlir_converter::PartitionedComputation& computation;
  absl::flat_hash_map<const HloInstruction*, int> fusion_result_index_starts;
  SmallVector<Value> thread_and_block_ids;
};

MlirReductionFusion::MlirReductionFusion(const HloFusionAnalysis& analysis)
    : ReductionFusionBase(analysis) {
  CHECK(reduction_info().IsRaceFree())
      << "Non-race-free reductions should have been decomposed. Did "
         "tree_reduction_rewriter run?";

  const auto& groups = reduction_info().GetGroups();
  int num_groups = groups.grouped_roots.size();
  side_output_roots_.resize(num_groups);
  reduction_heroes_.resize(num_groups);
  reduction_roots_.resize(num_groups);

  absl::flat_hash_set<const HloInstruction*> seen_heroes;
  for (auto [root_adaptor, hero_adaptor, is_reduction, group_id] :
       llvm::zip(analysis.fusion_roots(), analysis.fusion_heroes(),
                 groups.is_reduction_root, groups.group_id_per_root)) {
    const HloInstruction* root = &root_adaptor.instruction();
    const HloInstruction* hero = &hero_adaptor.instruction();
    if (is_reduction) {
      if (seen_heroes.insert(hero).second) {
        reduction_heroes_[group_id].push_back(hero);
      }
      reduction_roots_[group_id].push_back(root);
    } else {
      side_output_roots_[group_id].push_back(root);
    }
  }
}

std::vector<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  mlir::MLIRContext* mlir_context) const {
  std::vector<mlir_converter::EpilogueSpecification> epilogues;
  epilogues.reserve(reduction_heroes_.size());
  for (const auto& [heroes, roots] :
       llvm::zip(reduction_heroes_, reduction_roots_)) {
    epilogues.push_back(
        mlir_converter::EpilogueSpecification::FromOutputIndexing(
            analysis(), heroes, roots, *this, mlir_context));
  }
  return epilogues;
}

absl::Status MlirReductionFusion::EmitEntryFunction(
    const PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  EmitterState state{*this, entry_function, fusion, computations, call_targets};
  auto& b = state.builder;
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  state.thread_and_block_ids = EmitThreadAndBlockIds(b);
  if (reduction_heroes_.size() == 1) {
    b.create<mlir::func::ReturnOp>(EmitReduction(0, state));
    return absl::OkStatus();
  }
  SmallVector<int64_t> cases(reduction_heroes_.size() - 1);
  absl::c_iota(cases, 1);  // `default` is region 0.
  auto switch_op = b.create<mlir::scf::IndexSwitchOp>(
      entry_function.getResultTypes(), EmitBlockId(b, 1), cases, cases.size());
  b.create<mlir::func::ReturnOp>(switch_op.getResults());
  for (auto [id, region] : llvm::enumerate(switch_op->getRegions())) {
    b.setInsertionPointToStart(&region.emplaceBlock());
    b.create<mlir::scf::YieldOp>(EmitReduction(id, state));
  }
  return absl::OkStatus();
}

llvm::SmallVector<Value> MlirReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  const auto& tiling = reduction_info().GetTiling();
  const auto& threads_per_block = tiling.GetThreadsPerBlock();
  auto* ctx = state.entry_function.getContext();

  // The number of warps working on one element in a row reduction.
  int num_warps_row =
      threads_per_block[ReductionDimensions::kRowMinorReducedDimension] /
      WarpSize();

  Value zero = b.create<ma::ConstantIndexOp>(0);
  Value one = b.create<ma::ConstantIndexOp>(1);
  Value lane_id = b.create<mlir::gpu::LaneIdOp>();
  Value is_first_lane =
      b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, lane_id, zero);
  Value thread_id = state.thread_and_block_ids[0];
  Value cst_true = b.create<ma::ConstantOp>(b.getOneAttr(b.getI1Type()));

  auto thread_indexing = GetBitcastMap(
      ShapeUtil::MakeShapeWithDescendingLayout(
          U8, {tiling.GetNumThreadsPerBlock()}),
      ShapeUtil::MakeShapeWithDescendingLayout(U8, threads_per_block), ctx);
  auto thread_ids =
      mlir_converter::ApplyIndexing(thread_indexing, {thread_id}, {}, b);

  Value warp_id = b.create<ma::DivUIOp>(
      reduction_info().IsRowReduction()
          ? thread_ids[ReductionDimensions::kRowMinorReducedDimension]
          : thread_id,
      b.create<ma::ConstantIndexOp>(WarpSize()));
  // The number of results per thread.
  int64_t vector_size = tiling.GetThreadTileSize().back();
  Value vector_size_cst = b.create<ma::ConstantIndexOp>(vector_size);

  std::vector<int64_t> shared_tile_size;
  std::function<SmallVector<Value>(Value, ImplicitLocOpBuilder&)>
      shared_write_indices;
  std::function<SmallVector<Value>(Value, ImplicitLocOpBuilder&)>
      shared_read_indices;
  Value shared_write_condition = cst_true;
  Value shared_read_condition = cst_true;
  if (!reduction_info().IsRowReduction()) {
    shared_tile_size = {WarpSize(), WarpSize() * vector_size + 1};
    Value lane_id_times_v = b.create<ma::MulIOp>(lane_id, vector_size_cst);
    Value warp_id_times_v = b.create<ma::MulIOp>(warp_id, vector_size_cst);
    shared_write_indices = [=](Value vector_index,
                               ImplicitLocOpBuilder& builder) {
      mlir::Value col =
          builder.create<ma::AddIOp>(lane_id_times_v, vector_index);
      return SmallVector<Value>{warp_id, col};
    };
    shared_read_indices = [=](Value vector_index,
                              ImplicitLocOpBuilder& builder) {
      mlir::Value col =
          builder.create<ma::AddIOp>(warp_id_times_v, vector_index);
      return SmallVector<Value>{lane_id, col};
    };
  } else if (reduction_info().GetRowsPerWarp() == 1 && num_warps_row > 1) {
    CHECK_EQ(vector_size, 1);
    constexpr int kKept = ReductionDimensions::kRowKeptDimension;
    shared_tile_size = {tiling.GetThreadsPerBlock()[kKept], num_warps_row};
    shared_write_condition = is_first_lane;
    shared_read_condition = b.create<ma::CmpIOp>(
        ma::CmpIPredicate::ult,
        thread_ids[ReductionDimensions::kRowMinorReducedDimension],
        b.create<ma::ConstantIndexOp>(num_warps_row));
    shared_write_indices = [&](Value, ImplicitLocOpBuilder&) {
      return SmallVector<Value>{thread_ids[kKept], warp_id};
    };
    shared_read_indices = [&](Value, ImplicitLocOpBuilder&) {
      return SmallVector<Value>{thread_ids[kKept], lane_id};
    };
  }

  auto evaluate_epilogue = [&](ImplicitLocOpBuilder& b,
                               const HloValueMap& results,
                               llvm::SmallVector<Value> outputs,
                               Value vector_index = nullptr) {
    const auto& epilogue = state.computations.epilogues()[group_id];
    if (epilogue.roots.empty()) return outputs;

    llvm::SmallVector<Value> epilogue_input_symbols(
        epilogue.root_indexing.front().GetAffineMap().getNumSymbols(), zero);
    auto epilogue_input_indices = state.thread_and_block_ids;
    epilogue_input_indices.append(epilogue_input_symbols);

    if (!epilogue_input_symbols.empty() && vector_index) {
      epilogue_input_symbols.back() = epilogue_input_indices.back() =
          vector_index;
    }
    auto values =
        EmitEpilogue(group_id, state.computations, state.entry_function,
                     results, epilogue_input_indices, b);
    int first_root_index = state.OutputIndex(epilogue.roots.front(), 0);
    auto thread_has_output = mlir_converter::CheckConstraints(
        *ComputeThreadIdToOutputIndexing(first_root_index, ctx),
        state.thread_and_block_ids, epilogue_input_symbols, b);
    for (auto [index, root] : llvm::enumerate(epilogue.roots)) {
      auto output_indices = mlir_converter::ApplyIndexing(
          epilogue.root_indexing[index], state.thread_and_block_ids,
          epilogue_input_symbols, b);
      for (auto [result_index, result] : llvm::enumerate(values.at(root))) {
        auto& output = outputs[state.OutputIndex(root, result_index)];
        output = b.create<PredicatedInsertOp>(thread_has_output, result, output,
                                              output_indices);
      }
    }
    return mlir_converter::UnrealizedConversionCast(
        state.entry_function.getResultTypes(), outputs, b);
  };

  HloValueMap inits;
  const auto& reductions = reduction_heroes_[group_id];
  for (auto* hero : reductions) {
    int arity = hero->operand_count() / 2;
    inits[hero] =
        ProvideParameterRange(state.computation, hero, arity, arity, {},
                              state.call_target, state.entry_function, b);
  }
  llvm::SmallVector<Value> outputs =
      mlir::ValueRange(state.entry_function.getArguments().drop_front(
          state.fusion.fused_parameters().size()));
  for (auto* side_output : side_output_roots_[group_id]) {
    inits[side_output].push_back(outputs[state.OutputIndex(side_output, 0)]);
  }

  auto accumulated = state.EmitPerThreadReducedElements(group_id, inits);
  for (auto root : side_output_roots_[group_id]) {
    outputs[state.OutputIndex(root, 0)] = accumulated[root].front();
  }

  // In row reductions, we can do a warp shuffle before writing to shared
  // memory. In column reductions, the members of the warp process different
  // output elements, so we need to transpose first.
  if (reduction_info().IsRowReduction()) {
    for (auto* reduction : reductions) {
      auto reducer = state.GetReducer(reduction);
      int max_dist = WarpSize() / 2 / reduction_info().GetRowsPerWarp();
      const auto& inits_for_reduction = inits.at(reduction);
      auto& values = accumulated[reduction];
      values = mlir_converter::UnrealizedConversionCast(
          mlir::TypeRange(inits_for_reduction), values, b);
      values =
          b.create<ShuffleReduceOp>(reducer, values, max_dist).getResults();
    }
  }

  if (shared_tile_size.empty()) {
    return evaluate_epilogue(b, accumulated, std::move(outputs));
  }

  SmallVector<Value> shared_tiles =
      state.AllocateSharedTiles(reductions, shared_tile_size);
  auto write_loop = b.create<mlir::scf::ForOp>(
      zero, vector_size_cst, one, shared_tiles,
      [&](mlir::OpBuilder& body_builder, mlir::Location loc, Value vector_index,
          ValueRange tiles) {
        ImplicitLocOpBuilder b(loc, body_builder);
        int shared_index = 0;
        SmallVector<Value> written = tiles;
        for (auto* hero : reductions) {
          for (auto [value, init] : llvm::zip(accumulated[hero], inits[hero])) {
            if (mlir::isa<mlir::VectorType>(value.getType())) {
              value = b.create<mlir::vector::ExtractOp>(value, vector_index);
            }
            // Convert back to unsigned if necessary.
            value = mlir_converter::UnrealizedConversionCast(init.getType(),
                                                             value, b);
            auto indices = shared_write_indices(vector_index, b);
            auto& tile = written[shared_index++];
            tile = b.create<PredicatedInsertOp>(loc, shared_write_condition,
                                                value, tile, indices);
          }
        }
        b.create<mlir::scf::YieldOp>(written);
      });
  // Wait for the entire tile to be written.
  auto synced_tiles = b.create<SyncThreadsOp>(mlir::TypeRange(shared_tiles),
                                              write_loop.getResults())
                          .getResults();

  auto write_outputs = [&](mlir::OpBuilder& body_builder, mlir::Location loc,
                           Value vector_index, ValueRange outputs) {
    mlir::ImplicitLocOpBuilder b(loc, body_builder);
    int tile_index = 0;
    HloValueMap hero_values;
    for (auto* hero : reductions) {
      // Load from shared memory.
      SmallVector<Value> reduced;
      for (auto init : inits[hero]) {
        auto indices = shared_read_indices(vector_index, b);
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(
            b.create<PredicatedExtractOp>(shared_read_condition, init,
                                          synced_tiles[tile_index++], indices)
                .getResult());
      }
      const auto& reducer = state.GetReducer(hero);
      hero_values[hero] =
          b.create<ShuffleReduceOp>(reducer, reduced, WarpSize() / 2)
              .getResults();
    }

    b.create<mlir::scf::YieldOp>(
        loc, evaluate_epilogue(b, hero_values, outputs, vector_index));
  };

  if (reduction_info().IsRowReduction()) {
    CHECK_EQ(vector_size, 1);
    auto warp_writes =
        b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, warp_id, zero);
    auto if_op = b.create<mlir::scf::IfOp>(mlir::TypeRange(outputs),
                                           warp_writes, true, true);
    auto then_builder = if_op.getThenBodyBuilder();
    write_outputs(then_builder, b.getLoc(), zero, outputs);
    if_op.getElseBodyBuilder().create<mlir::scf::YieldOp>(b.getLoc(), outputs);
    return if_op.getResults();
  }

  return b
      .create<mlir::scf::ForOp>(zero, vector_size_cst, one, outputs,
                                write_outputs)
      .getResults();
}

HloValueMap MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    int group_id, const HloValueMap& inits) {
  const auto& tiling = owner.reduction_info().GetTiling();
  auto tile_indexing = GetIndexingMapForTiling(tiling, builder.getContext());
  tile_indexing
      .GetMutableDimensionBound(
          KernelFusionInterface::kIndexingMapBlockIdxDims[1])
      .upper = owner.reduction_heroes_.size();
  tile_indexing.Simplify();
  bool vectorize = tiling.GetThreadTileSize().back() > 1;

  SmallVector<Value> iter_arg_inits;
  const auto& side_outputs = owner.side_output_roots_[group_id];
  const auto& reductions = owner.reduction_heroes_[group_id];
  absl::flat_hash_map<const HloInstruction*, int> iter_arg_starts;
  for (const auto& [hero, init] : inits) {
    iter_arg_starts[hero] = iter_arg_inits.size();
    iter_arg_inits.append(init);
  }
  iter_arg_inits = mlir_converter::ConvertToSignless(iter_arg_inits, builder);

  auto body_builder = [&](ValueRange iter_args, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto tile_indices = mlir_converter::ApplyIndexing(tile_indexing, dim_values,
                                                      symbol_values, builder);

    llvm::SmallVector<Value> results(iter_args.size(), nullptr);
    auto get_input_indices = [&](auto* hero, bool is_reduction) {
      const auto& input_shape =
          is_reduction ? hero->operand(0)->shape() : hero->shape();
      return mlir_converter::ApplyIndexing(
          GetBitcastMap(tiling.GetXlaShape(), input_shape,
                        builder.getContext()),
          tile_indices, {}, builder);
    };
    for (auto* reduction : reductions) {
      int arity = reduction->operand_count() / 2;
      int start = iter_arg_starts[reduction];
      const auto& inits_for_reduction = inits.at(reduction);
      SmallVector<Value> reduce_args = mlir_converter::UnrealizedConversionCast(
          mlir::TypeRange(inits_for_reduction), iter_args.slice(start, arity),
          builder);
      reduce_args.append(ProvideParameterRange(
          computation, reduction, 0, arity, get_input_indices(reduction, true),
          call_target, entry_function, builder));
      const auto& reducer = GetReducer(reduction);
      absl::c_copy(
          mlir_converter::ConvertToSignless(
              builder.create<PureCallOp>(reducer, reduce_args).getResults(),
              builder),
          results.begin() + start);
    }
    struct SideOutput {
      llvm::SmallVector<Value> indices;
      Value scalar;
    };
    llvm::SmallVector<SideOutput> side_output_values;
    for (auto* side_output : side_outputs) {
      auto indices = get_input_indices(side_output, false);
      auto* root_tuple = fusion.fused_expression_root();
      Value value = mlir_converter::ProvideParameter(
          computation, root_tuple, root_tuple->operand_index(side_output),
          indices, call_target, entry_function, builder)[0];
      value = mlir_converter::ConvertToSignless(value, builder).front();
      side_output_values.push_back({std::move(indices), value});
    }
    for (const auto& [side_output, values] :
         llvm::zip(side_outputs, side_output_values)) {
      int offset = iter_arg_starts[side_output];
      results[offset] = builder.create<mlir::tensor::InsertOp>(
          values.scalar, iter_args[offset], values.indices);
    }
    return results;
  };

  auto results_vector = owner.EmitThreadLoopNest(
      builder, iter_arg_inits, tile_indexing, body_builder, vectorize);
  mlir::ValueRange results = results_vector;
  HloValueMap results_per_hero;
  for (const auto& [hero, init] : inits) {
    results_per_hero[hero] = results.slice(iter_arg_starts[hero], init.size());
  }
  for (auto* side_output : side_outputs) {
    auto& results_for_hero = results_per_hero[side_output];
    results_for_hero = mlir_converter::UnrealizedConversionCast(
        mlir::TypeRange(inits.at(side_output)), results_for_hero, builder);
  }
  return results_per_hero;
}

SmallVector<Value> MlirReductionFusion::EmitterState::AllocateSharedTiles(
    absl::Span<const HloInstruction* const> heroes,
    absl::Span<const int64_t> shape) {
  SmallVector<Value> tiles;
  for (auto* hero : heroes) {
    for (int i = 0; i < hero->operand_count() / 2; ++i) {
      auto tile_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          hero->operand(i)->shape().element_type(), shape);
      tiles.push_back(builder.create<AllocateSharedOp>(
          mlir_converter::TensorShapeToMlirType(tile_shape, builder)));
    }
  }
  return tiles;
}

}  // namespace gpu
}  // namespace xla
