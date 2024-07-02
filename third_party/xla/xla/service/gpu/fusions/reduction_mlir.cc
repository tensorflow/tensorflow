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
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
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

constexpr int kRowMajorReduced = ReductionDimensions::kRowMajorReducedDimension;
constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
constexpr int kRowMinorReduced = ReductionDimensions::kRowMinorReducedDimension;

LaunchDimensions MlirReductionFusion::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/Product(num_blocks_),
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/Product(num_threads_),
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

IndexingMap MlirReductionFusion::GetIndexingMap(
    llvm::ArrayRef<mlir::AffineExpr> results) const {
  auto* ctx = results.front().getContext();
  auto affine_map =
      AffineMap::get(6, tile_sizes_per_thread_.size(), results, ctx);
  auto symbol_sizes = tile_sizes_per_thread_;
  // Set the tile size to 1 for symbols that are not used and remove trailing
  // unused symbols.
  int last_used_symbol = -1;
  for (int i = 0; i < symbol_sizes.size(); ++i) {
    if (!affine_map.isFunctionOfSymbol(i)) {
      symbol_sizes[i] = 1;
    } else {
      last_used_symbol = i;
    }
  }
  symbol_sizes.resize(last_used_symbol + 1);
  affine_map = AffineMap::get(6, symbol_sizes.size(), results, ctx);
  auto num_groups = static_cast<int64_t>(reduction_heroes_.size());
  return IndexingMap{
      affine_map,
      DimVarsFromTensorSizes(
          {Product(num_threads_), 1, 1, Product(num_blocks_), num_groups, 1}),
      RangeVarsFromTensorSizes(symbol_sizes),
      /*rt_vars=*/{}};
}

IndexingMap MlirReductionFusion::GetThreadVectorIndexingMap(
    llvm::ArrayRef<mlir::AffineExpr> results,
    absl::Span<std::pair<mlir::AffineExpr, Interval> const> constraints) const {
  auto affine_map = AffineMap::get(2, 0, results, results.front().getContext());
  return IndexingMap{
      affine_map, DimVarsFromTensorSizes({Product(num_threads_), vector_size_}),
      /*range_vars=*/{},
      /*rt_vars=*/{}, constraints};
}

struct PerThreadOutputs {
  // The partially reduced scalars for each thread.
  HloValueMap reduction_scalars;
  // The outputs after writing side outputs.
  SmallVector<Value> outputs;
};

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
  // to the output tensors.
  PerThreadOutputs EmitPerThreadElements(int group_id, const HloValueMap& inits,
                                         const SmallVector<Value>& outputs);

  HloValueMap ReduceSharedMemory(
      ImplicitLocOpBuilder& b, ValueRange tiles, const HloValueMap& inits,
      absl::Span<const HloInstruction* const> reductions,
      mlir::Value vector_index = nullptr);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  // Writes `values` to newly allocated shared memory tiles, at the indices
  // given by `GetSharedMemoryWriteMap`.
  SmallVector<Value> WriteToSharedMemory(
      absl::Span<const HloInstruction* const> reductions,
      const HloValueMap& values);

  HloValueMap ShuffleReduce(ImplicitLocOpBuilder& b,
                            absl::Span<const HloInstruction* const> reductions,
                            const HloValueMap& per_thread_values,
                            int max_dist = WarpSize() / 2);

  SmallVector<Value> FusionParams() {
    return ValueRange(entry_function.getArguments().take_front(
        fusion.fused_parameters().size()));
  }

  mlir::ValueRange FusionOutputs() {
    return entry_function.getArguments().drop_front(
        fusion.fused_parameters().size());
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

IndexingMap MlirRowReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(3, ctx), num_blocks_);
  auto major_reduced = GetLoopSymbol(0, ctx);
  auto minor_reduced = GetLoopSymbol(1, ctx);
  auto vector_index = GetLoopSymbol(2, ctx);

  SmallVector<AffineExpr> indices{
      major_reduced,
      block_id[0] * tile_sizes_per_block_[0] + thread_id[0],
      block_id[1] * tile_sizes_per_block_[1] +
          (minor_reduced * num_threads_[1]) + thread_id[1],
      vector_index,
  };

  auto map = GetIndexingMap(indices);
  for (auto [result, input_dim] : llvm::zip(indices, input_shape_)) {
    map.AddConstraint(result, {0, input_dim - 1});
  }
  return map;
}

IndexingMap MlirMultiRowReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id = num_blocks_.front() == 1 ? mlir::getAffineConstantExpr(0, ctx)
                                           : mlir::getAffineDimExpr(3, ctx);
  IndexingMap projected_index =
      GetIndexingMap(block_id * num_threads_[0] + thread_id[0]);
  projected_index.AddConstraint(thread_id[1] % (WarpSize() / GetRowsPerWarp()),
                                {0, 0});
  return projected_index;
}

IndexingMap MlirMultiRowReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id = num_blocks_.front() == 1 ? mlir::getAffineConstantExpr(0, ctx)
                                           : mlir::getAffineDimExpr(3, ctx);
  auto major_reduced = GetLoopSymbol(0, ctx);

  SmallVector<AffineExpr> indices{
      major_reduced, block_id * num_threads_[0] + thread_id[0], thread_id[1]};

  auto map = GetIndexingMap(indices);
  for (auto [result, input_dim] : llvm::zip(indices, input_shape_)) {
    map.AddConstraint(result, {0, input_dim - 1});
  }
  return map;
}

IndexingMap MlirRowReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(3, ctx), num_blocks_);
  IndexingMap projected_index =
      GetIndexingMap(block_id[0] * tile_sizes_per_block_[0] + thread_id[0]);
  projected_index.AddConstraint(thread_id[1], {0, 0});
  return projected_index;
}

HloValueMap MlirReductionFusion::GetInits(int group_id,
                                          EmitterState& state) const {
  HloValueMap result;
  const auto& reductions = reduction_heroes_[group_id];
  for (auto* hero : reductions) {
    int arity = hero->operand_count() / 2;
    result[hero] = ProvideParameterRange(state.computation, hero, arity, arity,
                                         {}, state.call_target,
                                         state.entry_function, state.builder);
  }
  return result;
}

PerThreadOutputs MlirReductionFusion::EmitterState::EmitPerThreadElements(
    int group_id, const HloValueMap& inits, const SmallVector<Value>& outputs) {
  auto tile_indexing =
      owner.ComputeReductionInputIndexing(builder.getContext());
  tile_indexing
      .GetMutableDimensionBound(
          KernelFusionInterface::kIndexingMapBlockIdxDims[1])
      .upper = owner.reduction_heroes_.size();
  tile_indexing.Simplify();
  bool vectorize = owner.vector_size_ > 1;

  SmallVector<Value> iter_arg_inits = outputs;
  const auto& side_outputs = owner.side_output_roots_[group_id];
  const auto& reductions = owner.reduction_heroes_[group_id];
  absl::flat_hash_map<const HloInstruction*, int> iter_arg_starts;

  for (const auto& [reduction, init] : inits) {
    iter_arg_starts[reduction] = iter_arg_inits.size();
    iter_arg_inits.append(init);
  }

  auto body_builder = [&](ValueRange iter_args, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto tile_indices = mlir_converter::ApplyIndexing(tile_indexing, dim_values,
                                                      symbol_values, builder);

    llvm::SmallVector<Value> results = iter_args;
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
      // The first iter args are the outputs.
      int offset = OutputIndex(side_output, 0);
      results[offset] = builder.create<mlir::tensor::InsertOp>(
          values.scalar, iter_args[offset], values.indices);
    }
    return results;
  };

  auto results_vector = owner.EmitThreadLoopNest(
      builder, iter_arg_inits, tile_indexing, body_builder, vectorize);
  mlir::ValueRange results = results_vector;

  PerThreadOutputs scalars_and_outputs;
  scalars_and_outputs.outputs = results.slice(0, outputs.size());
  for (const auto& [reduction, init] : inits) {
    scalars_and_outputs.reduction_scalars[reduction] =
        results.slice(iter_arg_starts[reduction], init.size());
  }
  return scalars_and_outputs;
}

SmallVector<Value> MlirReductionFusion::EmitterState::WriteToSharedMemory(
    absl::Span<const HloInstruction* const> reductions,
    const HloValueMap& values) {
  SmallVector<int64_t> shape;
  auto map = owner.GetSharedMemoryWriteMap(builder.getContext());
  for (auto result : map.GetAffineMap().getResults()) {
    shape.push_back(
        map.GetRangeEvaluator().ComputeExpressionRange(result).upper + 1);
  }
  if ((shape.back() % WarpSize()) == 0) {
    // Avoid bank conflicts.
    ++shape.back();
  }

  SmallVector<Value> tiles;
  for (auto* reduction : reductions) {
    for (int i = 0; i < reduction->operand_count() / 2; ++i) {
      auto tile_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          reduction->operand(i)->shape().element_type(), shape);
      tiles.push_back(builder.create<AllocateSharedOp>(
          mlir_converter::TensorShapeToMlirType(tile_shape, builder)));
    }
  }

  auto vector_size_cst =
      builder.create<ma::ConstantIndexOp>(owner.vector_size_);
  auto zero = builder.create<ma::ConstantIndexOp>(0);
  auto one = builder.create<ma::ConstantIndexOp>(1);
  auto thread_id = owner.EmitThreadId(builder, 0);

  auto write_loop = builder.create<mlir::scf::ForOp>(
      zero, vector_size_cst, one, tiles,
      [&](mlir::OpBuilder& body_builder, mlir::Location loc, Value vector_index,
          ValueRange tiles) {
        ImplicitLocOpBuilder b(loc, body_builder);
        auto write_condition = mlir_converter::CheckConstraints(
            map, {thread_id, vector_index}, {}, b);
        auto indices = mlir_converter::ApplyIndexing(
            map, {thread_id, vector_index}, {}, b);
        int shared_index = 0;
        SmallVector<Value> written = tiles;
        for (auto* hero : reductions) {
          for (auto value : values.at(hero)) {
            if (mlir::isa<mlir::VectorType>(value.getType())) {
              value = b.create<mlir::vector::ExtractOp>(value, vector_index);
            }
            auto& tile = written[shared_index++];
            tile = b.create<PredicatedInsertOp>(loc, write_condition, value,
                                                tile, indices);
          }
        }
        b.create<mlir::scf::YieldOp>(written);
      });

  // Wait for the entire tile to be written.
  auto synced_tiles = builder
                          .create<SyncThreadsOp>(mlir::TypeRange(tiles),
                                                 write_loop.getResults())
                          .getResults();

  return synced_tiles;
}

HloValueMap MlirReductionFusion::EmitterState::ShuffleReduce(
    ImplicitLocOpBuilder& b, absl::Span<const HloInstruction* const> reductions,
    const HloValueMap& per_thread_values, int max_dist) {
  HloValueMap results;
  for (auto* hero : reductions) {
    auto reduce = b.create<ShuffleReduceOp>(
        GetReducer(hero), per_thread_values.at(hero), max_dist);
    results[hero] = reduce.getResults();
  }
  return results;
}

HloValueMap MlirReductionFusion::EmitterState::ReduceSharedMemory(
    ImplicitLocOpBuilder& b, ValueRange tiles, const HloValueMap& inits,
    absl::Span<const HloInstruction* const> reductions,
    mlir::Value vector_index) {
  int64_t tile_index = 0;
  auto read_indexing = owner.GetSharedMemoryReductionReadMap(b.getContext());

  if (!vector_index) {
    vector_index = b.create<ma::ConstantIndexOp>(0);
  }

  auto thread_id = owner.EmitThreadId(b, 0);
  auto read_condition = mlir_converter::CheckConstraints(
      read_indexing, {thread_id, vector_index}, {}, b);
  auto indices = mlir_converter::ApplyIndexing(
      read_indexing, {thread_id, vector_index}, {}, b);

  HloValueMap results;
  HloValueMap reduce_args;
  for (auto* hero : reductions) {
    auto& args = reduce_args[hero];
    for (auto init : inits.at(hero)) {
      // If a warp didn't write anything, use the init values instead.
      auto extract = b.create<PredicatedExtractOp>(
          read_condition, init, tiles[tile_index++], indices);
      args.push_back(extract.getResult());
    }
  }
  return ShuffleReduce(b, reductions, reduce_args);
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
  auto map = ComputeReductionInputIndexing(ctx);
  AddGroupIdConstraint(map, root_index, groups_);
  return map * GetBitcastMap(input_shape_,
                             hero.operand(hero_operand_index)->shape(), ctx);
}

std::optional<IndexingMap> MlirReductionFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        ComputeReductionInputIndexing(ctx),
        GetBitcastMap(input_shape_, analysis_.fusion_root(root_index).shape(),
                      ctx));
    AddGroupIdConstraint(map, root_index, groups_);
    return map;
  }

  auto projected_indexing = ComputeReductionOutputIndexing(ctx);
  auto output_shape = reduction_dimensions_.GetOutputShape();
  CHECK_EQ(output_shape.size(),
           projected_indexing.GetAffineMap().getNumResults());
  for (auto [result, dim_size] : llvm::zip(
           projected_indexing.GetAffineMap().getResults(), output_shape)) {
    projected_indexing.AddConstraint(result, {0, dim_size - 1});
  }
  AddGroupIdConstraint(projected_indexing, root_index, groups_);

  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  auto physical_shape =
      ShapeUtil::DeleteDimensions(hero.dimensions(), hero.operand(0)->shape());
  return projected_indexing *
         GetBitcastMap(ShapeUtil::MakeShapeWithDescendingLayout(
                           PrimitiveType::U8, output_shape),
                       physical_shape, ctx);
}

mlir::AffineExpr MlirReductionFusion::GetLoopSymbol(
    int index, mlir::MLIRContext* ctx) const {
  if (tile_sizes_per_thread_[index] == 1) {
    return mlir::getAffineConstantExpr(0, ctx);
  }
  return mlir::getAffineSymbolExpr(index, ctx);
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
  CHECK_EQ(RowReductionGetRowsPerWarp(shape[kRowMinorReduced]), 1);
  constexpr int64_t kMinorReducedElementsPerThread = 16;

  int64_t num_threads_kept = 1;
  int64_t num_threads_reduced = [&] {
    int64_t max_block_size =
        MinThreadsXRowReduction(first_reduce_->GetModule()->config());
    return std::min(max_block_size,
                    RoundUpTo(CeilOfRatio(shape[kRowMinorReduced],
                                          kMinorReducedElementsPerThread),
                              WarpSize()));
  }();

  // If we're limited by the size of the x dimension, add additional parallelism
  // in the y dimension. The code generator doesn't currently support
  // parallelizing the z dimension (major reduced dimensions). The general
  // recommendation is to use between 128 and 512 threads, so we just go for
  // 256. See https://forums.developer.nvidia.com/t/55529
  constexpr int64_t kThreadsPerBlockTarget = 256;
  if (num_threads_reduced * 2 <= kThreadsPerBlockTarget) {
    int64_t kept_size = reduction_dimensions_.dimensions[kRowKept];
    // Increase the size of the y dimension as long as there's remaining
    // parallelism.
    if (kept_size * num_threads_reduced <= kThreadsPerBlockTarget) {
      num_threads_kept = kept_size;
    } else {
      num_threads_kept = kThreadsPerBlockTarget / num_threads_reduced;
    }
  }

  int vector_size = GetVectorSizeForMlir(analysis, reduction_dimensions_,
                                         num_threads_reduced);
  num_threads_ = {num_threads_kept, num_threads_reduced};
  // TODO(jreiffers): Get rid of `vector_size` in here.
  input_shape_ = {shape[0], shape[1], shape[2] / vector_size, vector_size};
  // TODO(jreiffers): Tighten ranges based on constraints when simplifying
  // instead of using min here. For example, based on
  //
  //   s1 in [0, 127]
  //   d0 floordiv 32 + s1 * 32 in [0, 63]
  //
  // Tighten the bound of s1 to [0, 1].
  int minor_reduced_tile_size =
      std::min(kMinorReducedElementsPerThread / vector_size,
               CeilOfRatio(input_shape_[2], num_threads_[1]));

  tile_sizes_per_thread_ = {shape[0], minor_reduced_tile_size, vector_size};
  tile_sizes_per_block_ = {num_threads_kept,
                           minor_reduced_tile_size * num_threads_reduced};
  num_blocks_ = {CeilOfRatio(input_shape_[1], tile_sizes_per_block_[0]),
                 CeilOfRatio(input_shape_[2], tile_sizes_per_block_[1])};

  vector_size_ = 1;
}

MlirMultiRowReductionFusion::MlirMultiRowReductionFusion(
    const HloFusionAnalysis& analysis)
    : MlirReductionFusion(analysis) {
  CHECK(reduction_dimensions_.is_row_reduction);
  Vector3 shape = reduction_dimensions_.dimensions;
  int64_t rows_per_warp = RowReductionGetRowsPerWarp(shape[kRowMinorReduced]);
  CHECK_GT(rows_per_warp, 1);

  int64_t num_threads_reduced = shape[kRowMinorReduced];

  constexpr int64_t kThreadsPerBlockTarget = 256;
  int64_t kept_size = reduction_dimensions_.dimensions[kRowKept];
  int64_t num_threads_kept = 1;
  if (kept_size * num_threads_reduced <= kThreadsPerBlockTarget) {
    num_threads_kept = kept_size;
  } else {
    num_threads_kept = kThreadsPerBlockTarget / num_threads_reduced;
  }

  num_threads_ = {num_threads_kept, num_threads_reduced};
  input_shape_ = {shape[0], shape[1], shape[2]};
  tile_sizes_per_thread_ = {shape[0]};
  num_blocks_ = {CeilOfRatio(input_shape_[kRowKept], num_threads_kept)};

  vector_size_ = 1;
}

int MlirMultiRowReductionFusion::GetRowsPerWarp() const {
  return RowReductionGetRowsPerWarp(
      input_shape_[ReductionDimensions::kRowMinorReducedDimension]);
}

int MlirRowReductionFusion::GetWarpsPerRow() const {
  return CeilOfRatio(num_threads_[1], WarpSize());
}

IndexingMap MlirRowReductionFusion::GetSharedMemoryReductionReadMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto lane_id = thread_id[1] % WarpSize();
  return GetThreadVectorIndexingMap(
      {thread_id[0], lane_id}, {{thread_id[1], {0, GetWarpsPerRow() - 1}}});
}

IndexingMap MlirRowReductionFusion::GetSharedMemoryWriteMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  // The reduced dimension is tiled; each warp writes one element to shared
  // memory (from lane 0).
  auto lane_id = thread_id[1] % WarpSize();
  auto warp_id = thread_id[1].floorDiv(WarpSize());
  return GetThreadVectorIndexingMap({thread_id[0], warp_id},
                                    {{lane_id, {0, 0}}});
}

llvm::SmallVector<mlir::Value> MlirRowReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  auto* ctx = state.entry_function.getContext();
  const auto& reductions = reduction_heroes_[group_id];

  HloValueMap inits = GetInits(group_id, state);
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  auto reduced =
      state.ShuffleReduce(b, reductions, per_thread.reduction_scalars);

  if (GetWarpsPerRow() == 1) {
    // If only a single warp works on an element, we don't need to go through
    // shared memory.
    return EvaluateEpilogue(b, reduced, std::move(per_thread.outputs), state,
                            group_id, ctx);
  }

  SmallVector<Value> shared_tiles =
      state.WriteToSharedMemory(reductions, reduced);
  HloValueMap reduced_from_shared_memory =
      state.ReduceSharedMemory(b, shared_tiles, inits, reductions);
  return EvaluateEpilogue(b, reduced_from_shared_memory, per_thread.outputs,
                          state, group_id, ctx);
}

llvm::SmallVector<mlir::Value> MlirMultiRowReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  auto* ctx = state.entry_function.getContext();

  HloValueMap inits = GetInits(group_id, state);
  const auto& reductions = reduction_heroes_[group_id];
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  auto reduced =
      state.ShuffleReduce(b, reductions, per_thread.reduction_scalars,
                          WarpSize() / 2 / GetRowsPerWarp());
  return EvaluateEpilogue(b, reduced, std::move(per_thread.outputs), state,
                          group_id, ctx);
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
  int64_t num_warps_per_column = WarpSize();
  num_threads_ = {num_warps_per_column, WarpSize()};
  int64_t num_col_elements_per_thread =
      CeilOfRatio(reduction_dimensions_
                      .dimensions[ReductionDimensions::kColReducedDimension],
                  num_warps_per_column);
  tile_sizes_per_thread_ = {num_col_elements_per_thread, vector_size_};

  int64_t major_kept_dim =
      reduction_dimensions_
          .dimensions[ReductionDimensions::kColMajorKeptDimension];
  int64_t minor_kept_dim =
      reduction_dimensions_
          .dimensions[ReductionDimensions::kColMinorKeptDimension];
  int64_t num_blocks_per_row =
      CeilOfRatio(minor_kept_dim, WarpSize() * vector_size_);
  num_blocks_ = {major_kept_dim, num_blocks_per_row};
}

IndexingMap MlirColumnReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx), num_blocks_);
  AffineExpr vector_index = GetLoopSymbol(1, ctx);
  SmallVector<AffineExpr, 2> results{
      block_id[0],
      (block_id[1] * WarpSize() + thread_id[0]) * vector_size_ + vector_index};
  IndexingMap projected_index = GetIndexingMap(results);
  projected_index.AddConstraint(thread_id[1], {0, 0});
  return projected_index;
}

IndexingMap MlirColumnReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx), num_blocks_);
  AffineExpr element_index = GetLoopSymbol(0, ctx);
  AffineExpr vector_index = GetLoopSymbol(1, ctx);

  SmallVector<AffineExpr, 3> results{
      block_id[0], thread_id[0] + element_index * num_threads_[1],
      (block_id[1] * WarpSize() + thread_id[1]) * vector_size_ + vector_index};
  IndexingMap map = GetIndexingMap(results);
  for (auto [result, dim_size] :
       llvm::zip(results, reduction_dimensions_.dimensions)) {
    map.AddConstraint(result, {0, dim_size - 1});
  }
  return map;
}

IndexingMap MlirColumnReductionFusion::GetSharedMemoryReductionReadMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto vector_index = getAffineDimExpr(1, ctx);
  return GetThreadVectorIndexingMap(
      {thread_id[0], thread_id[1] * vector_size_ + vector_index}, {});
}

IndexingMap MlirColumnReductionFusion::GetSharedMemoryWriteMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto vector_index = getAffineDimExpr(1, ctx);
  return GetThreadVectorIndexingMap(
      {thread_id[1], thread_id[0] * vector_size_ + vector_index}, {});
}

llvm::SmallVector<mlir::Value> MlirColumnReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  auto& b = state.builder;
  auto* ctx = state.entry_function.getContext();

  Value zero = b.create<ma::ConstantIndexOp>(0);
  Value one = b.create<ma::ConstantIndexOp>(1);

  // The number of results per thread.
  Value vector_size_cst = b.create<ma::ConstantIndexOp>(vector_size_);

  const auto& reductions = reduction_heroes_[group_id];
  HloValueMap inits = GetInits(group_id, state);
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  SmallVector<Value> shared_tiles =
      state.WriteToSharedMemory(reductions, per_thread.reduction_scalars);
  auto write_outputs = [&](mlir::OpBuilder& body_builder, mlir::Location loc,
                           Value vector_index, ValueRange outputs) {
    mlir::ImplicitLocOpBuilder b(loc, body_builder);
    HloValueMap reduced = state.ReduceSharedMemory(b, shared_tiles, inits,
                                                   reductions, vector_index);
    b.create<mlir::scf::YieldOp>(
        loc, EvaluateEpilogue(b, reduced, outputs, state, group_id, ctx,
                              vector_index));
  };

  return b
      .create<mlir::scf::ForOp>(zero, vector_size_cst, one, per_thread.outputs,
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
    if (RowReductionGetRowsPerWarp(
            reduction_dimensions.dimensions[kRowMinorReduced]) > 1) {
      return std::make_unique<MlirMultiRowReductionFusion>(analysis);
    }
    return std::make_unique<MlirRowReductionFusion>(analysis);
  }
  return std::make_unique<MlirColumnReductionFusion>(analysis);
}

}  // namespace gpu
}  // namespace xla
