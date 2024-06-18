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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
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
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace ma = mlir::arith;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputations;

LaunchDimensions MlirReductionFusion::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/total_num_blocks_,
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/total_num_threads_per_block_,
                        /*y=*/1, /*z=*/1)};
}

MlirReductionFusion::MlirReductionFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  Shape input_shape = hero_reduction->operand(0)->shape();
  reduction_dimensions_ =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  VLOG(10) << reduction_dimensions_;

  CHECK(ReductionIsRaceFree(hero_reduction->GetModule()->config(),
                            reduction_dimensions_))
      << "Non-race-free reductions should have been decomposed. Did "
         "tree_reduction_rewriter run?";

  groups_ = GroupDisjointReductions(analysis, /*for_mlir=*/true);
  first_reduce_ = hero_reduction;

  const auto& groups = GetGroups();
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
    for (const auto& root : owner.analysis_.fusion_roots()) {
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

std::vector<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  MLIRContext* mlir_context) const {
  std::vector<mlir_converter::EpilogueSpecification> epilogues;
  epilogues.reserve(reduction_heroes_.size());
  for (const auto& [heroes, roots] :
       llvm::zip(reduction_heroes_, reduction_roots_)) {
    epilogues.push_back(
        mlir_converter::EpilogueSpecification::FromOutputIndexing(
            analysis_, heroes, roots, *this, mlir_context));
  }
  // Add empty epilogues for the side outputs. This ensures their roots don't
  // get "fused" into the tuple function.
  for (const auto& roots : side_output_roots_) {
    for (const auto* root : roots) {
      epilogues.push_back(
          mlir_converter::EpilogueSpecification::FromIdentityIndexing(
              root, root, mlir_context));
    }
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

IndexingMap MlirRowReductionFusion::ComputeThreadIdToReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto rank = input_shape_.size();

  auto thread_offsets =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto block_offsets =
      DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx), num_blocks_);
  SmallVector<AffineExpr> results;
  results.resize(rank);
  for (int i = 0; i < rank; ++i) {
    results[i] =
        block_offsets[i] * tile_sizes_per_block_[i] + thread_offsets[i];
    if (tile_sizes_per_thread_[i] > 1) {
      results[i] = results[i] + getAffineSymbolExpr(i, ctx) * num_threads_[i];
    }
  }
  IndexingMap map{AffineMap::get(6, rank, results, ctx),
                  DimVarsFromTensorSizes({total_num_threads_per_block_, 1, 1,
                                          total_num_blocks_, 1, 1}),
                  RangeVarsFromTensorSizes(tile_sizes_per_thread_),
                  /*rt_vars=*/{}};
  for (auto [result, input_dim] : llvm::zip(results, input_shape_)) {
    map.AddConstraint(result, {0, input_dim - 1});
  }
  return map;
}

HloValueMap MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    int group_id, const HloValueMap& inits) {
  auto tile_indexing =
      owner.ComputeThreadIdToReductionInputIndexing(builder.getContext());
  tile_indexing
      .GetMutableDimensionBound(
          KernelFusionInterface::kIndexingMapBlockIdxDims[1])
      .upper = owner.reduction_heroes_.size();
  tile_indexing.Simplify();
  bool vectorize = owner.vector_size_ > 1;

  SmallVector<Value> iter_arg_inits;
  const auto& side_outputs = owner.side_output_roots_[group_id];
  const auto& reductions = owner.reduction_heroes_[group_id];
  absl::flat_hash_map<const HloInstruction*, int> iter_arg_starts;
  for (const auto& [hero, init] : inits) {
    iter_arg_starts[hero] = iter_arg_inits.size();
    iter_arg_inits.append(init);
  }

  auto body_builder = [&](ValueRange iter_args, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto tile_indices = mlir_converter::ApplyIndexing(tile_indexing, dim_values,
                                                      symbol_values, builder);

    llvm::SmallVector<Value> results(iter_args.size(), nullptr);
    for (auto* reduction : reductions) {
      int arity = reduction->operand_count() / 2;
      int start = iter_arg_starts[reduction];
      SmallVector<Value> reduce_args = iter_args.slice(start, arity);
      auto indices = mlir_converter::ApplyIndexing(
          GetBitcastMap(owner.input_shape_, reduction->operand(0)->shape(),
                        builder.getContext()),
          tile_indices, {}, builder);
      reduce_args.append(ProvideParameterRange(computation, reduction, 0, arity,
                                               indices, call_target,
                                               entry_function, builder));
      const auto& reducer = GetReducer(reduction);
      absl::c_copy(
          builder.create<PureCallOp>(reducer, reduce_args).getResults(),
          results.begin() + start);
    }
    struct SideOutput {
      llvm::SmallVector<Value> indices;
      Value scalar;
    };
    llvm::SmallVector<SideOutput> side_output_values;
    for (auto* side_output : side_outputs) {
      auto indices = mlir_converter::ApplyIndexing(
          GetBitcastMap(owner.input_shape_, side_output->shape(),
                        builder.getContext()),
          tile_indices, {}, builder);
      auto* root_tuple = fusion.fused_expression_root();
      Value value = mlir_converter::ProvideParameter(
          computation, root_tuple, root_tuple->operand_index(side_output),
          indices, call_target, entry_function, builder)[0];
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

std::optional<IndexingMap> MlirReductionFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index, MLIRContext* ctx) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (groups_.is_reduction_root[root_index] &&
      hero_operand_index >= hero.operand_count() / 2) {
    // We don't have indexing for the init values.
    return std::nullopt;
  }
  if (!groups_.is_reduction_root[root_index]) {
    return ComposeIndexingMaps(
        *ComputeThreadIdToOutputIndexing(root_index, ctx),
        *ComputeOutputToInputIndexing(
             &analysis_.fusion_root(root_index).instruction(), 0, ctx)
             .indexing_maps[hero_operand_index]
             .begin());
  }
  auto map = ComputeThreadIdToReductionInputIndexing(ctx);
  AddGroupIdConstraint(map, root_index, groups_);
  return map * GetBitcastMap(input_shape_,
                             hero.operand(hero_operand_index)->shape(), ctx);
}

SmallVector<Value> MlirReductionFusion::EvaluateEpilogue(
    ImplicitLocOpBuilder& b, const HloValueMap& results,
    llvm::SmallVector<Value> outputs, EmitterState& state, int group_id,
    MLIRContext* ctx, Value vector_index) const {
  Value zero = b.create<ma::ConstantIndexOp>(0);
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
  auto values = EmitEpilogue(group_id, state.computations, state.entry_function,
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
  return outputs;
}

MlirRowReductionFusion::MlirRowReductionFusion(
    const HloFusionAnalysis& analysis)
    : MlirReductionFusion(analysis) {
  CHECK(reduction_dimensions_.is_row_reduction);
  Vector3 shape = reduction_dimensions_.dimensions;
  Vector3 reduction_tiling = {
      std::min(reduction_dimensions_
                   .dimensions[ReductionDimensions::kRowMajorReducedDimension],
               BatchedReductionRaceFreeBound()),
      1, 16};

  int64_t num_threads_y = 1;
  int64_t rows_per_warp = RowReductionGetRowsPerWarp(
      shape[ReductionDimensions::kRowMinorReducedDimension]);
  int64_t num_threads_x = [&] {
    if (rows_per_warp > 1) {
      return shape[ReductionDimensions::kRowMinorReducedDimension];
    }
    int64_t max_block_size =
        MinThreadsXRowReduction(first_reduce_->GetModule()->config());
    return std::min(
        max_block_size,
        RoundUpTo(
            CeilOfRatio(shape[ReductionDimensions::kRowMinorReducedDimension],
                        reduction_tiling
                            [ReductionDimensions::kRowMinorReducedDimension]),
            WarpSize()));
  }();

  // If we're limited by the size of the x dimension, add additional parallelism
  // in the y dimension. The code generator doesn't currently support
  // parallelizing the z dimension (major reduced dimensions). The general
  // recommendation is to use between 128 and 512 threads, so we just go for
  // 256. See https://forums.developer.nvidia.com/t/55529
  constexpr int64_t kThreadsPerBlockTarget = 256;
  if (num_threads_x * 2 <= kThreadsPerBlockTarget) {
    int64_t kept_size = reduction_dimensions_
                            .dimensions[ReductionDimensions::kRowKeptDimension];
    // Increase the size of the y dimension as long as there's remaining
    // parallelism.
    if (kept_size * num_threads_x <= kThreadsPerBlockTarget) {
      num_threads_y = kept_size;
      // num_threads_x is a power of two, but it may be less than 32. If dim_y
      // is also small, we may have to increase the bound so the total number of
      // threads is a multiple of 32.
      while ((num_threads_x * num_threads_y) % 32) ++num_threads_y;
    } else {
      num_threads_y = kThreadsPerBlockTarget / num_threads_x;
    }
  }

  int vector_size =
      GetVectorSizeForMlir(analysis, reduction_dimensions_, num_threads_x);

  num_threads_ =
      absl::InlinedVector<int64_t, 4>{1, num_threads_y, num_threads_x};
  input_shape_ = {shape[0], shape[1], shape[2] / vector_size};
  tile_sizes_per_thread_ = {
      reduction_tiling[0], reduction_tiling[1],
      std::max<int64_t>(reduction_tiling[2] / vector_size, 1)};
  // The indexing map simplifier does not currently handle this correctly,
  // leading to loop bounds that are too large.
  // TODO(jreiffers): Implement tightening of ranges based on constraints
  // instead. For example, based on:
  //
  //   s1 in [0, 127]
  //   d0 floordiv 32 + s1 * 32 in [0, 63]
  //
  // Tighten the bound of s1 to [0, 1].
  for (int i = 0; i < num_threads_.size(); ++i) {
    tile_sizes_per_thread_[i] =
        std::min(tile_sizes_per_thread_[i],
                 CeilOfRatio(input_shape_[i], num_threads_[i]));
  }
  if (rows_per_warp > 1) {
    // If we produce more than one element per thread, that means the reduced
    // dimension is small and it can't be tiled - we already have more threads
    // in a warp than the size of the reduced dimension. The code generator
    // doesn't currently support tiling the kept dimension, because it just
    // uses the thread ID as the coordinate.
    tile_sizes_per_thread_[2] = 1;
  }
  if (vector_size != 1) {
    num_threads_.push_back(1);  // The vector dimension is a loop.
    input_shape_.push_back(vector_size);
    tile_sizes_per_thread_.push_back(vector_size);
  }

  // The MLIR emitter treats the last tiled dimension as the number of parallel
  // independent reductions per thread (to use vectorized loads). This is only
  // needed for column reductions: row reductions can use vectorized loads for
  // the same reduction.
  // row reduction:     [[a, b], [c, d]] -> [a + b, c + d]
  // column reduction:  [[a, b], [c, d]] -> [a + c, b + d]
  // In both cases [a, b] are loaded together, but only in the column reduction
  // they contribute to different result elements.
  num_threads_.push_back(1);
  input_shape_.push_back(1);
  tile_sizes_per_thread_.push_back(1);

  tile_sizes_per_block_.resize(input_shape_.size());
  num_blocks_.resize(input_shape_.size());
  for (int64_t i = 0; i < input_shape_.size(); ++i) {
    tile_sizes_per_block_[i] = tile_sizes_per_thread_[i] * num_threads_[i];
    CHECK_NE(tile_sizes_per_block_[i], 0);
    num_blocks_[i] = CeilOfRatio(input_shape_[i], tile_sizes_per_block_[i]);
    CHECK_NE(num_blocks_[i], 0);
  }

  total_num_blocks_ = Product(num_blocks_);
  total_num_threads_per_block_ = Product(num_threads_);
  vector_size_ = tile_sizes_per_thread_.back();
}

std::optional<IndexingMap>
MlirRowReductionFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        ComputeThreadIdToReductionInputIndexing(ctx),
        GetBitcastMap(input_shape_, analysis_.fusion_root(root_index).shape(),
                      ctx));
    AddGroupIdConstraint(map, root_index, groups_);
    return map;
  }
  const auto& hero = analysis_.fusion_hero(root_index).instruction();

  auto thread_ids =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_offsets = GetBlockOffsetsForTiling(
      num_blocks_, tile_sizes_per_block_, input_shape_.size(), ctx);

  auto physical_shape =
      ShapeUtil::DeleteDimensions(hero.dimensions(), hero.operand(0)->shape());
  std::vector<DimVar> dimension_ranges{
      {{0, total_num_threads_per_block_ - 1}},
      {},
      {},
      {{0, total_num_blocks_ - 1}},
      {{0, static_cast<int64_t>(groups_.grouped_roots.size() - 1)}},
      {},
  };

  constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
  constexpr int kRowMinorReduced =
      ReductionDimensions::kRowMinorReducedDimension;

  auto map = [&]() {
    IndexingMap linear_index(
        mlir::AffineMap::get(
            6, 0, block_offsets.getResult(kRowKept) + thread_ids[kRowKept],
            ctx),
        dimension_ranges, /*range_vars=*/{}, /*rt_vars=*/{});
    int rows_per_warp = GetRowsPerWarp();
    if (rows_per_warp > 1) {
      linear_index.AddConstraint(
          thread_ids[kRowMinorReduced] % (WarpSize() / rows_per_warp), {0, 0});
    } else {
      linear_index.AddConstraint(thread_ids[kRowMinorReduced], {0, 0});
    }
    return ComposeIndexingMaps(
        linear_index,
        GetBitcastMap({input_shape_[kRowKept]}, physical_shape, ctx));
  }();

  AddGroupIdConstraint(map, root_index, groups_);
  return map;
}

int MlirRowReductionFusion::GetRowsPerWarp() const {
  return RowReductionGetRowsPerWarp(
      input_shape_[ReductionDimensions::kRowMinorReducedDimension]);
}

llvm::SmallVector<mlir::Value> MlirRowReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  auto* ctx = state.entry_function.getContext();

  // The number of warps working on one element in a row reduction.
  int num_warps_row =
      num_threads_[ReductionDimensions::kRowMinorReducedDimension] / WarpSize();

  Value zero = b.create<ma::ConstantIndexOp>(0);
  Value one = b.create<ma::ConstantIndexOp>(1);
  Value thread_id = state.thread_and_block_ids[0];
  auto thread_indexing =
      GetBitcastMap({total_num_threads_per_block_},
                    ShapeUtil::MakeShapeWithDescendingLayout(U8, num_threads_),
                    b.getContext());
  auto thread_ids =
      mlir_converter::ApplyIndexing(thread_indexing, {thread_id}, {}, b);

  Value lane_id = b.create<mlir::gpu::LaneIdOp>();
  Value warp_id = b.create<ma::DivUIOp>(
      thread_ids[ReductionDimensions::kRowMinorReducedDimension],
      b.create<ma::ConstantIndexOp>(WarpSize()));
  Value is_first_lane =
      b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, lane_id, zero);

  // The number of results per thread.
  int64_t vector_size = tile_sizes_per_thread_.back();
  Value vector_size_cst = b.create<ma::ConstantIndexOp>(vector_size);

  std::vector<int64_t> shared_tile_size;
  if (GetRowsPerWarp() == 1 && num_warps_row > 1) {
    CHECK_EQ(vector_size, 1);
    shared_tile_size = {num_threads_[ReductionDimensions::kRowKeptDimension],
                        num_warps_row};
  }

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
  for (auto* reduction : reductions) {
    auto reducer = state.GetReducer(reduction);
    int max_dist = WarpSize() / 2 / GetRowsPerWarp();
    auto& values = accumulated[reduction];
    values = b.create<ShuffleReduceOp>(reducer, values, max_dist).getResults();
  }

  if (shared_tile_size.empty()) {
    return EvaluateEpilogue(b, accumulated, std::move(outputs), state, group_id,
                            ctx);
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
            SmallVector<Value> indices{
                thread_ids[ReductionDimensions::kRowKeptDimension], warp_id};
            auto& tile = written[shared_index++];
            tile = b.create<PredicatedInsertOp>(loc, is_first_lane, value, tile,
                                                indices);
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
    Value shared_read_condition = b.create<ma::CmpIOp>(
        ma::CmpIPredicate::ult,
        thread_ids[ReductionDimensions::kRowMinorReducedDimension],
        b.create<ma::ConstantIndexOp>(num_warps_row));
    for (auto* hero : reductions) {
      // Load from shared memory.
      SmallVector<Value> reduced;
      for (auto init : inits[hero]) {
        SmallVector<Value> indices{
            thread_ids[ReductionDimensions::kRowKeptDimension], lane_id};
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
        loc, EvaluateEpilogue(b, hero_values, outputs, state, group_id, ctx,
                              vector_index));
  };

  CHECK_EQ(vector_size, 1);
  auto warp_writes = b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, warp_id, zero);
  auto if_op = b.create<mlir::scf::IfOp>(mlir::TypeRange(outputs), warp_writes,
                                         true, true);
  auto then_builder = if_op.getThenBodyBuilder();
  write_outputs(then_builder, b.getLoc(), zero, outputs);
  if_op.getElseBodyBuilder().create<mlir::scf::YieldOp>(b.getLoc(), outputs);
  return if_op.getResults();
}

MlirColumnReductionFusion::MlirColumnReductionFusion(
    const HloFusionAnalysis& analysis)
    : MlirReductionFusion(analysis) {
  CHECK(!reduction_dimensions_.is_row_reduction);

  input_shape_ = {reduction_dimensions_.dimensions[0],
                  reduction_dimensions_.dimensions[1],
                  reduction_dimensions_.dimensions[2]};
  vector_size_ =
      GetVectorSizeForMlir(analysis, reduction_dimensions_, WarpSize());
  num_warps_per_column_ = WarpSize();
  total_num_threads_per_block_ = num_warps_per_column_ * WarpSize();

  int64_t major_kept_dim =
      reduction_dimensions_
          .dimensions[ReductionDimensions::kColMajorKeptDimension];
  int64_t minor_kept_dim =
      reduction_dimensions_
          .dimensions[ReductionDimensions::kColMinorKeptDimension];
  num_blocks_per_row_ = CeilOfRatio(minor_kept_dim, WarpSize() * vector_size_);
  total_num_blocks_ = major_kept_dim * num_blocks_per_row_;
}

std::optional<IndexingMap>
MlirColumnReductionFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        ComputeThreadIdToReductionInputIndexing(ctx),
        GetBitcastMap(input_shape_, analysis_.fusion_root(root_index).shape(),
                      ctx));
    AddGroupIdConstraint(map, root_index, groups_);
    return map;
  }
  AffineExpr th_x = getAffineDimExpr(0, ctx);
  AffineExpr bl_x = getAffineDimExpr(3, ctx);
  AffineExpr s_v = getAffineSymbolExpr(0, ctx);

  auto reduced_shape = ShapeUtil::DeleteDimension(
      ReductionDimensions::kColReducedDimension,
      ShapeUtil::MakeShape(PrimitiveType::F32, input_shape_));
  SmallVector<AffineExpr, 2> results{
      bl_x.floorDiv(num_blocks_per_row_),
      ((bl_x % num_blocks_per_row_) * WarpSize() + th_x.floorDiv(WarpSize())) *
              vector_size_ +
          s_v};
  IndexingMap map{AffineMap::get(6, 1, results, ctx),
                  DimVarsFromTensorSizes(
                      {total_num_threads_per_block_, 1, 1, total_num_blocks_,
                       static_cast<int64_t>(groups_.grouped_roots.size()), 1}),
                  RangeVarsFromTensorSizes({vector_size_}),
                  /*rt_vars=*/{}};
  for (auto [result, dim_size] :
       llvm::zip(results, reduced_shape.dimensions())) {
    map.AddConstraint(result, {0, dim_size - 1});
  }
  map.AddConstraint(th_x % WarpSize(), {0, 0});
  AddGroupIdConstraint(map, root_index, groups_);
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  auto physical_shape =
      ShapeUtil::DeleteDimensions(hero.dimensions(), hero.operand(0)->shape());
  return map * GetBitcastMap(reduced_shape, physical_shape, ctx);
}

IndexingMap MlirColumnReductionFusion::ComputeThreadIdToReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  AffineExpr th_x = getAffineDimExpr(0, ctx);
  AffineExpr bl_x = getAffineDimExpr(3, ctx);
  AffineExpr s_e = getAffineSymbolExpr(0, ctx);
  AffineExpr s_v = getAffineSymbolExpr(1, ctx);

  int64_t num_col_elements_per_thread =
      CeilOfRatio(reduction_dimensions_
                      .dimensions[ReductionDimensions::kColReducedDimension],
                  num_warps_per_column_);
  SmallVector<AffineExpr, 3> results{
      bl_x.floorDiv(num_blocks_per_row_),
      th_x.floorDiv(WarpSize()) + s_e * num_warps_per_column_,
      ((bl_x % num_blocks_per_row_) * WarpSize() + th_x % WarpSize()) *
              vector_size_ +
          s_v};
  IndexingMap map{
      AffineMap::get(6, 2, results, ctx),
      DimVarsFromTensorSizes(
          {total_num_threads_per_block_, 1, 1, total_num_blocks_,
           static_cast<int64_t>(groups_.grouped_roots.size()), 1}),
      RangeVarsFromTensorSizes({num_col_elements_per_thread, vector_size_}),
      /*rt_vars=*/{}};
  for (auto [result, dim_size] :
       llvm::zip(results, reduction_dimensions_.dimensions)) {
    map.AddConstraint(result, {0, dim_size - 1});
  }
  return map;
}

llvm::SmallVector<mlir::Value> MlirColumnReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  auto* ctx = state.entry_function.getContext();

  Value zero = b.create<ma::ConstantIndexOp>(0);
  Value one = b.create<ma::ConstantIndexOp>(1);
  Value cst_true = b.create<ma::ConstantOp>(b.getOneAttr(b.getI1Type()));

  Value thread_id = state.thread_and_block_ids[0];
  Value lane_id = b.create<mlir::gpu::LaneIdOp>();
  Value warp_id = b.create<ma::DivUIOp>(
      thread_id, b.create<ma::ConstantIndexOp>(WarpSize()));

  // The number of results per thread.
  Value vector_size_cst = b.create<ma::ConstantIndexOp>(vector_size_);

  std::vector<int64_t> shared_tile_size{WarpSize(),
                                        WarpSize() * vector_size_ + 1};
  Value lane_id_times_v = b.create<ma::MulIOp>(lane_id, vector_size_cst);
  Value warp_id_times_v = b.create<ma::MulIOp>(warp_id, vector_size_cst);

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
            mlir::Value col =
                b.create<ma::AddIOp>(lane_id_times_v, vector_index);
            auto indices = {warp_id, col};
            auto& tile = written[shared_index++];
            tile = b.create<PredicatedInsertOp>(loc, cst_true, value, tile,
                                                indices);
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
        Value col = b.create<ma::AddIOp>(warp_id_times_v, vector_index);
        SmallVector<Value> indices{lane_id, col};
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(
            b.create<PredicatedExtractOp>(cst_true, init,
                                          synced_tiles[tile_index++], indices)
                .getResult());
      }
      const auto& reducer = state.GetReducer(hero);
      hero_values[hero] =
          b.create<ShuffleReduceOp>(reducer, reduced, WarpSize() / 2)
              .getResults();
    }

    b.create<mlir::scf::YieldOp>(
        loc, EvaluateEpilogue(b, hero_values, outputs, state, group_id, ctx,
                              vector_index));
  };

  return b
      .create<mlir::scf::ForOp>(zero, vector_size_cst, one, outputs,
                                write_outputs)
      .getResults();
}

std::unique_ptr<MlirReductionFusion> CreateMlirReductionFusion(
    const HloFusionAnalysis& analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  if (reduction_dimensions.is_row_reduction) {
    return std::make_unique<MlirRowReductionFusion>(analysis);
  }
  return std::make_unique<MlirColumnReductionFusion>(analysis);
}

}  // namespace gpu
}  // namespace xla
