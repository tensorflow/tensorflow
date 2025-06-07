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
#include "xla/backends/gpu/codegen/emitters/reduction.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/reduction_base.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using emitters::PartitionedComputations;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::Value;
using mlir::ValueRange;

constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
constexpr int kRowMinorReduced = ReductionDimensions::kRowMinorReducedDimension;
constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;

struct PerThreadOutputs {
  // The partially reduced scalars for each thread.
  HloValueMap reduction_scalars;
  // The outputs after writing side outputs.
  SmallVector<Value> outputs;
};

struct ReductionFusion::EmitterState {
  EmitterState(const ReductionFusion& owner, mlir::func::FuncOp entry_function,
               const HloFusionInstruction& fusion,
               const PartitionedComputations& computations,
               const emitters::CallTargetProvider& call_target)
      : owner(owner),
        entry_function(entry_function),
        fusion(fusion),
        computations(computations),
        call_target(call_target),
        builder(entry_function.getLoc(), entry_function),
        computation(computations.FindPartitionedComputation(
            fusion.fused_instructions_computation())) {
    int output_index = 0;
    for (const auto& [root_index, root] :
         llvm::enumerate(owner.analysis_.fusion_roots())) {
      root_indices[&root.instruction()] = root_index;
      fusion_result_index_starts[&root.instruction()] = output_index;
      output_index +=
          root.shape().IsTuple() ? root.shape().tuple_shapes().size() : 1;
    }
  }

  // Reduces a subset of the inputs in a single thread. Also writes side outputs
  // to the output tensors.
  PerThreadOutputs EmitPerThreadElements(int group_id, const HloValueMap& inits,
                                         const SmallVector<Value>& outputs);

  mlir::ValueRange ReduceViaSharedMemory(int group_id,
                                         const PerThreadOutputs& per_thread,
                                         const HloValueMap& inits,
                                         std::optional<int> padding,
                                         int max_dist);

  mlir::ValueRange ReduceViaSharedMemory(
      int group_id, const PerThreadOutputs& per_thread,
      const HloValueMap& inits, std::optional<int> padding = std::nullopt) {
    return ReduceViaSharedMemory(group_id, per_thread, inits, padding,
                                 owner.WarpSize() / 2);
  }

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  // Writes `values` to newly allocated shared memory tiles, at the indices
  // given by `GetSharedMemoryWriteMap`.
  SmallVector<Value> WriteToSharedMemory(
      absl::Span<const HloInstruction* const> reductions,
      const HloValueMap& values, std::optional<int> padding = std::nullopt);

  HloValueMap ShuffleReduce(absl::Span<const HloInstruction* const> reductions,
                            const HloValueMap& per_thread_values, int max_dist);

  HloValueMap ShuffleReduce(absl::Span<const HloInstruction* const> reductions,
                            const HloValueMap& per_thread_values) {
    return ShuffleReduce(reductions, per_thread_values, owner.WarpSize() / 2);
  }

  mlir::ValueRange FusionOutputs() {
    return entry_function.getArguments().drop_front(
        fusion.fused_parameters().size());
  }

  int OutputIndex(const HloInstruction* root, int result_index) {
    return fusion_result_index_starts[root] + result_index;
  }

  const ReductionFusion& owner;
  mlir::func::FuncOp entry_function;
  const HloFusionInstruction& fusion;
  const PartitionedComputations& computations;
  const emitters::CallTargetProvider& call_target;
  ImplicitLocOpBuilder builder;
  const emitters::PartitionedComputation& computation;
  absl::flat_hash_map<const HloInstruction*, int> fusion_result_index_starts;
  absl::flat_hash_map<const HloInstruction*, int> root_indices;
  SmallVector<Value> thread_and_block_ids;
};

PerThreadOutputs ReductionFusion::EmitterState::EmitPerThreadElements(
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

  auto body_builder = [&](ImplicitLocOpBuilder& nested_b,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange iter_args) -> SmallVector<Value> {
    llvm::SmallVector<Value> results = iter_args;
    for (auto* reduction : reductions) {
      int arity = reduction->operand_count() / 2;
      int start = iter_arg_starts[reduction];
      SmallVector<Value> reduce_args = iter_args.slice(start, arity);
      auto indices = emitters::ApplyIndexing(
          GetBitcastMap(owner.input_shape_, reduction->operand(0)->shape(),
                        nested_b.getContext()),
          map_results, {}, nested_b);
      reduce_args.append(ProvideParameterRange(computation, reduction, 0, arity,
                                               indices, call_target,
                                               entry_function, nested_b));
      auto reducer = GetReducer(reduction);
      // Annotate all AddF ops in the reducer with the no signed zeros fastmath
      // flag. This allows to fold the initial add with the zero init constant.
      auto no_signed_zeros = mlir::arith::FastMathFlagsAttr::get(
          nested_b.getContext(), mlir::arith::FastMathFlags::nsz);
      reducer.walk([&](mlir::arith::AddFOp addf) {
        addf->setAttr("fastmath", no_signed_zeros);
      });
      absl::c_copy(
          nested_b.create<PureCallOp>(reducer, reduce_args).getResults(),
          results.begin() + start);
    }
    struct SideOutput {
      llvm::SmallVector<Value> indices;
      Value scalar;
    };
    llvm::SmallVector<SideOutput> side_output_values;
    for (auto* side_output : side_outputs) {
      auto indices = emitters::ApplyIndexing(
          GetBitcastMap(owner.input_shape_, side_output->shape(),
                        builder.getContext()),
          map_results, {}, builder);
      auto* root_tuple = fusion.fused_expression_root();
      Value value = emitters::ProvideParameter(
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

  auto results_vector =
      emitters::EmitXlaLoopOp(builder, thread_and_block_ids, iter_arg_inits,
                              tile_indexing, body_builder, vectorize);
  mlir::ValueRange results = results_vector;

  PerThreadOutputs scalars_and_outputs;
  scalars_and_outputs.outputs = results.slice(0, outputs.size());
  for (const auto& [reduction, init] : inits) {
    scalars_and_outputs.reduction_scalars[reduction] =
        results.slice(iter_arg_starts[reduction], init.size());
  }
  return scalars_and_outputs;
}

SmallVector<Value> ReductionFusion::EmitterState::WriteToSharedMemory(
    absl::Span<const HloInstruction* const> reductions,
    const HloValueMap& values, std::optional<int> padding) {
  SmallVector<int64_t> shape;
  auto map = owner.GetSharedMemoryWriteMap(builder.getContext());
  for (auto result : map.GetAffineMap().getResults()) {
    shape.push_back(
        map.GetRangeEvaluator().ComputeExpressionRange(result).upper + 1);
  }
  if (padding) {
    shape.back() += *padding;
  } else if ((shape.back() % owner.WarpSize()) == 0) {
    // Avoid bank conflicts.
    ++shape.back();
  }

  SmallVector<Value> tiles;
  for (auto* reduction : reductions) {
    for (int i = 0; i < reduction->operand_count() / 2; ++i) {
      auto tile_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          reduction->operand(i)->shape().element_type(), shape);
      tiles.push_back(builder.create<AllocateSharedOp>(
          emitters::TensorShapeToMlirType(tile_shape, builder)));
    }
  }

  auto written_tiles = emitters::EmitLoopNest(
      builder, {thread_and_block_ids[0]}, tiles, map,
      [&](mlir::ValueRange iter_args, mlir::ValueRange dim_values,
          mlir::ValueRange symbol_values) {
        auto indices =
            emitters::ApplyIndexing(map, dim_values, symbol_values, builder);
        int shared_index = 0;
        SmallVector<Value> written = iter_args;
        for (auto* hero : reductions) {
          for (auto value : values.at(hero)) {
            if (mlir::isa<mlir::VectorType>(value.getType())) {
              value = builder.create<mlir::vector::ExtractOp>(
                  value, symbol_values.back());
            }
            auto& tile = written[shared_index++];
            tile = builder.create<mlir::tensor::InsertOp>(value, tile, indices);
          }
        }
        return written;
      });

  // Wait for the entire tile to be written.
  auto synced_tiles =
      builder.create<SyncThreadsOp>(mlir::TypeRange(tiles), written_tiles)
          .getResults();

  return synced_tiles;
}

HloValueMap ReductionFusion::EmitterState::ShuffleReduce(
    absl::Span<const HloInstruction* const> reductions,
    const HloValueMap& per_thread_values, int max_dist) {
  HloValueMap results;
  for (auto* hero : reductions) {
    auto reduce = builder.create<ShuffleReduceOp>(
        GetReducer(hero), per_thread_values.at(hero), max_dist);
    results[hero] = reduce.getResults();
  }
  return results;
}

mlir::ValueRange ReductionFusion::EmitterState::ReduceViaSharedMemory(
    int group_id, const PerThreadOutputs& per_thread, const HloValueMap& inits,
    std::optional<int> padding, int max_dist) {
  const auto& reductions = owner.reduction_heroes_[group_id];
  auto read_indexing =
      owner.GetSharedMemoryReductionReadMap(builder.getContext());
  auto loop_indexing = read_indexing;
  // All threads must participate in the shuffle, so we clear the constraints
  // for the iteration. Otherwise, some threads might not be part of the loop,
  // resulting in incorrect results for the warp shuffle.
  // The constraints are still checked inside the loop in the
  // PredicatedExtractOp.
  loop_indexing.ClearConstraints();
  // The constraints may have reduced the upper bound of the dimension. If
  // that's the case, we reset it to a multiple of the warp size.
  auto& bound = loop_indexing.GetMutableDimensionBound(0);
  bound.upper = RoundUpTo(bound.upper + 1, owner.WarpSize()) - 1;

  auto tiles =
      WriteToSharedMemory(reductions, per_thread.reduction_scalars, padding);
  return emitters::EmitLoopNest(
      builder, {thread_and_block_ids[0]}, per_thread.outputs, loop_indexing,
      [&](ValueRange outputs, ValueRange dim_values,
          ValueRange symbol_values) -> SmallVector<Value> {
        auto read_condition = emitters::CheckConstraints(
            read_indexing, dim_values, symbol_values, builder);
        auto indices = emitters::ApplyIndexing(read_indexing, dim_values,
                                               symbol_values, builder);

        int64_t tile_index = 0;
        HloValueMap reduce_args;
        for (auto* hero : reductions) {
          auto& args = reduce_args[hero];
          for (auto init : inits.at(hero)) {
            // If a warp didn't write anything, use the init values instead.
            auto extract = builder.create<PredicatedExtractOp>(
                read_condition, init, tiles[tile_index++], indices);
            args.push_back(extract.getResult());
          }
        }
        auto reduced = ShuffleReduce(reductions, reduce_args, max_dist);
        return owner.EvaluateEpilogue(reduced, outputs, *this, group_id,
                                      symbol_values);
      });
}

ReductionFusion::ReductionFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  reduction_dimensions_ =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  VLOG(10) << reduction_dimensions_;

  CHECK(ReductionIsRaceFree(reduction_dimensions_, analysis.device_info()))
      << "Non-race-free reductions should have been decomposed. Did "
         "tree_reduction_rewriter run?";

  groups_ = GroupDisjointReductions(analysis);
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

IndexingMap ReductionFusion::GetIndexingMap(
    llvm::ArrayRef<mlir::AffineExpr> results,
    absl::Span<int64_t const> symbol_sizes) const {
  auto* ctx = results.front().getContext();
  auto num_groups = static_cast<int64_t>(reduction_heroes_.size());
  return IndexingMap{AffineMap::get(6, symbol_sizes.size(), results, ctx),
                     DimVarsFromGPUGrid({Product(num_threads_), 1, 1,
                                         Product(num_blocks_), num_groups, 1}),
                     RangeVarsFromTensorSizes(symbol_sizes),
                     /*rt_vars=*/{}};
}

IndexingMap ReductionFusion::GetThreadIndexingMap(
    llvm::ArrayRef<mlir::AffineExpr> results,
    absl::Span<std::pair<mlir::AffineExpr, Interval> const> constraints,
    absl::Span<int64_t const> symbol_sizes) const {
  auto affine_map = AffineMap::get(1, symbol_sizes.size(), results,
                                   results.front().getContext());
  return IndexingMap{
      affine_map,
      {IndexingMap::Variable{0, Product(num_threads_) - 1,
                             ToVariableName(VariableKind::kThreadX)}},
      RangeVarsFromTensorSizes(symbol_sizes),
      /*rt_vars=*/{},
      constraints};
}

LaunchDimensions ReductionFusion::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/Product(num_blocks_),
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/Product(num_threads_),
                        /*y=*/1, /*z=*/1)};
}

std::vector<emitters::EpilogueSpecification> ReductionFusion::GetEpilogues(
    const HloFusionInstruction& fusion, MLIRContext* mlir_context) const {
  std::vector<emitters::EpilogueSpecification> epilogues;
  epilogues.reserve(reduction_heroes_.size());
  for (const auto& [heroes, roots] :
       llvm::zip(reduction_heroes_, reduction_roots_)) {
    epilogues.push_back(
        GetEpilogueForOutputIndexing(analysis_, heroes, roots, mlir_context));
  }
  // Add empty epilogues for the side outputs. This ensures their roots don't
  // get "fused" into the tuple function.
  for (const auto& roots : side_output_roots_) {
    for (const auto* root : roots) {
      epilogues.push_back(emitters::EpilogueSpecification::FromIdentityIndexing(
          root, root, mlir_context));
    }
  }
  return epilogues;
}

absl::Status ReductionFusion::EmitEntryFunction(
    const PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
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

HloValueMap ReductionFusion::GetInits(int group_id, EmitterState& state) const {
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

std::optional<IndexingMap> ReductionFusion::ComputeThreadIdToInputIndexing(
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
        ComputeOutputToInputIndexing(
            &analysis_.fusion_root(root_index).instruction(), 0, ctx)
            .indexing_maps[hero_operand_index]
            .begin()
            ->map());
  }
  auto projected_map = ComputeReductionInputIndexing(ctx);
  AddGroupIdConstraint(projected_map, root_index, groups_);
  auto map = projected_map *
             GetBitcastMap(input_shape_,
                           hero.operand(hero_operand_index)->shape(), ctx);
  map.Simplify();
  return map;
}

std::optional<IndexingMap> ReductionFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* ctx) const {
  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        ComputeReductionInputIndexing(ctx),
        GetBitcastMap(input_shape_, analysis_.fusion_root(root_index).shape(),
                      ctx));
    AddGroupIdConstraint(map, root_index, groups_);
    map.Simplify();
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
  auto map =
      projected_indexing * GetBitcastMap(output_shape, physical_shape, ctx);
  map.Simplify();
  return map;
}

SmallVector<Value> ReductionFusion::EvaluateEpilogue(
    const HloValueMap& results, llvm::SmallVector<Value> outputs,
    EmitterState& state, int group_id, ValueRange symbol_values) const {
  ImplicitLocOpBuilder& b = state.builder;
  const auto& epilogue = state.computations.epilogues()[group_id];
  if (epilogue.roots.empty()) return outputs;

  auto epilogue_input_indices = state.thread_and_block_ids;
  epilogue_input_indices.append(symbol_values.begin(), symbol_values.end());

  auto values = EmitEpilogue(group_id, state.computations, state.entry_function,
                             results, epilogue_input_indices, b);
  int first_root_index = state.root_indices[epilogue.roots.front()];
  auto thread_has_output = emitters::CheckConstraints(
      *ComputeThreadIdToOutputIndexing(first_root_index, b.getContext()),
      state.thread_and_block_ids, symbol_values, b);
  for (auto [index, root] : llvm::enumerate(epilogue.roots)) {
    auto output_indices =
        emitters::ApplyIndexing(epilogue.root_indexing[index],
                                state.thread_and_block_ids, symbol_values, b);
    for (auto [result_index, result] : llvm::enumerate(values.at(root))) {
      auto& output = outputs[state.OutputIndex(root, result_index)];
      output = b.create<PredicatedInsertOp>(thread_has_output, result, output,
                                            output_indices);
    }
  }
  return outputs;
}

ColumnReductionFusion::ColumnReductionFusion(const HloFusionAnalysis& analysis)
    : ReductionFusion(analysis) {
  CHECK(!reduction_dimensions_.is_row_reduction);

  input_shape_ = {reduction_dimensions_.dimensions[0],
                  reduction_dimensions_.dimensions[1],
                  reduction_dimensions_.dimensions[2]};
  vector_size_ = GetVectorSizeForMlir(
      analysis, /*minor_dim=*/input_shape_.back(), WarpSize());
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

IndexingMap ColumnReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx), num_blocks_);
  auto vector_index = getAffineSymbolExpr(0, ctx);
  SmallVector<AffineExpr, 2> results{
      block_id[0],
      (block_id[1] * WarpSize() + thread_id[0]) * vector_size_ + vector_index};
  IndexingMap projected_index =
      GetIndexingMap(results, /*symbol_sizes=*/{vector_size_});
  projected_index.AddConstraint(thread_id[1], {0, 0});
  return projected_index;
}

IndexingMap ColumnReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(3, ctx), num_blocks_);
  AffineExpr element_index = getAffineSymbolExpr(0, ctx);
  AffineExpr vector_index = getAffineSymbolExpr(1, ctx);

  SmallVector<AffineExpr, 3> results{
      block_id[0], thread_id[0] + element_index * num_threads_[1],
      (block_id[1] * WarpSize() + thread_id[1]) * vector_size_ + vector_index};
  IndexingMap map = GetIndexingMap(results, tile_sizes_per_thread_);
  for (auto [result, dim_size] :
       llvm::zip(results, reduction_dimensions_.dimensions)) {
    map.AddConstraint(result, {0, dim_size - 1});
  }
  return map;
}

IndexingMap ColumnReductionFusion::GetSharedMemoryReductionReadMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto vector_index = getAffineSymbolExpr(0, ctx);
  return GetThreadIndexingMap(
      {thread_id[0], thread_id[1] * vector_size_ + vector_index}, {},
      /*symbol_sizes=*/{vector_size_});
}

IndexingMap ColumnReductionFusion::GetSharedMemoryWriteMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto vector_index = getAffineSymbolExpr(0, ctx);
  return GetThreadIndexingMap(
      {thread_id[1], thread_id[0] * vector_size_ + vector_index}, {},
      /*symbol_sizes=*/{vector_size_});
}

llvm::SmallVector<mlir::Value> ColumnReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  HloValueMap inits = GetInits(group_id, state);
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  return state.ReduceViaSharedMemory(group_id, per_thread, inits);
}

SmallColumnReductionFusion::SmallColumnReductionFusion(
    const HloFusionAnalysis& analysis)
    : ReductionFusion(analysis) {
  CHECK(!reduction_dimensions_.is_row_reduction);

  input_shape_ = {reduction_dimensions_.dimensions[0],
                  reduction_dimensions_.dimensions[1],
                  reduction_dimensions_.dimensions[2]};
  // We emit a single loop over the dimensions 1 and 2, so we use their total
  // size when computing the vector size.
  vector_size_ = GetVectorSizeForMlir(
      analysis, /*minor_dim=*/input_shape_[1] * input_shape_[2], WarpSize());
  num_threads_ = {128};
  shared_rows_ = vector_size_ * num_threads_[0] / input_shape_[kColMinorKept];

  // If we have more than 32 shared rows, we'd have to go through shared
  // memory one extra time. We don't currently support that, and it's not been
  // tried, so we have to reduce the vector size/number of threads.
  while (shared_rows_ > WarpSize() && vector_size_ > 1) {
    vector_size_ /= 2;
    shared_rows_ /= 2;
  }
  if (shared_rows_ > WarpSize()) {
    num_threads_[0] /= (shared_rows_ / WarpSize());
    shared_rows_ = WarpSize();
  }

  num_blocks_ = {input_shape_[kColMajorKept]};
  loop_size_ = CeilOfRatio(input_shape_[1] * input_shape_[2],
                           vector_size_ * num_threads_[0]);
}

IndexingMap SmallColumnReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id = getAffineDimExpr(0, ctx);
  auto block_id = getAffineDimExpr(3, ctx);
  auto vector_index = getAffineSymbolExpr(0, ctx);
  SmallVector<AffineExpr, 2> results{
      block_id,
      (thread_id + vector_index * num_threads_[0]).floorDiv(shared_rows_)};
  IndexingMap projected_index =
      GetIndexingMap(results, /*symbol_sizes=*/{vector_size_});
  projected_index.AddConstraint(thread_id % shared_rows_, {0, 0});
  return projected_index;
}

IndexingMap SmallColumnReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id = getAffineDimExpr(0, ctx);
  auto block_id = getAffineDimExpr(3, ctx);
  AffineExpr loop_index = getAffineSymbolExpr(0, ctx);
  AffineExpr vector_index = getAffineSymbolExpr(1, ctx);

  AffineExpr linear_index = thread_id * vector_size_ + vector_index +
                            loop_index * (vector_size_ * num_threads_[0]);
  auto map =
      GetIndexingMap({block_id, linear_index}, {loop_size_, vector_size_}) *
      GetBitcastMap({num_blocks_[0], input_shape_[1] * input_shape_[2]},
                    ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::U8,
                                                             input_shape_),
                    ctx);

  for (auto [result, dim_size] :
       llvm::zip(map.GetAffineMap().getResults(), input_shape_)) {
    map.AddConstraint(result, {0, dim_size - 1});
  }
  return map;
}

IndexingMap SmallColumnReductionFusion::GetSharedMemoryReductionReadMap(
    mlir::MLIRContext* ctx) const {
  auto indices = DelinearizeInBoundsIndex(
      getAffineDimExpr(0, ctx) + getAffineSymbolExpr(0, ctx) * num_threads_[0],
      {input_shape_[2], shared_rows_});
  return GetThreadIndexingMap({indices[1], indices[0]}, {},
                              /*symbol_sizes=*/{vector_size_});
}

IndexingMap SmallColumnReductionFusion::GetSharedMemoryWriteMap(
    mlir::MLIRContext* ctx) const {
  auto indices = DelinearizeInBoundsIndex(
      getAffineDimExpr(0, ctx) * vector_size_ + getAffineSymbolExpr(0, ctx),
      {shared_rows_, input_shape_[2]});
  return GetThreadIndexingMap(indices, {},
                              /*symbol_sizes=*/{vector_size_});
}

llvm::SmallVector<mlir::Value> SmallColumnReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  HloValueMap inits = GetInits(group_id, state);
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  // This is the minimal padding that avoids all bank conflicts. We use at most
  // 1.5*thread count*vector size elements, which is much less than the
  // 1056*vector size**2 the other column reduction emitter uses.
  int padding =
      (num_threads_[0] == 32 && vector_size_ == 1)
          ? 0
          : CeilOfRatio(input_shape_[2], num_threads_[0] * vector_size_);
  return state.ReduceViaSharedMemory(group_id, per_thread, inits, padding,
                                     shared_rows_ / 2);
}

RowReductionFusion::RowReductionFusion(const HloFusionAnalysis& analysis)
    : ReductionFusion(analysis) {
  CHECK(reduction_dimensions_.is_row_reduction);
  Vector3 shape = reduction_dimensions_.dimensions;
  int64_t kMinorReducedElementsPerThread = 8;

  do {
    kMinorReducedElementsPerThread *= 2;
    int64_t num_threads_kept = 1;
    // Number of threads doing the reduction.
    int64_t num_threads_reduced = [&] {
      int64_t max_block_size = MinThreadsXRowReduction();
      return std::min(max_block_size,
                      RoundUpTo(CeilOfRatio(shape[kRowMinorReduced],
                                            kMinorReducedElementsPerThread),
                                WarpSize()));
    }();

    // If we're limited by the size of the x dimension, add additional
    // parallelism in the y dimension. The code generator doesn't currently
    // support parallelizing the z dimension (major reduced dimensions). The
    // general recommendation is to use between 128 and 512 threads, so we just
    // go for 256. See https://forums.developer.nvidia.com/t/55529
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

    int vector_size = GetVectorSizeForMlir(analysis, /*minor_dim=*/shape.back(),
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
    /* ROCm hipModuleLaunchKernel limitation
     * https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___module.html#ga2e4de5937aa8171e9eda16c881ed0674
     */
  } while (xla::PlatformUtil::CanonicalPlatformName("gpu").value() == "rocm" &&
           kMinorReducedElementsPerThread < 65536 &&
           ((Product(num_blocks_) * Product(num_threads_)) >
            std::numeric_limits<uint32_t>::max()));

  VLOG(3) << absl::StrFormat(
      "RowReductionFusion::RowReductionFusion selected parameters: num_threads "
      "= [%s], tile_sizes_per_thread = [%s], tile_sizes_per_block = [%s], "
      "num_blocks = [%s]",
      absl::StrJoin(num_threads_, ","),
      absl::StrJoin(tile_sizes_per_thread_, ","),
      absl::StrJoin(tile_sizes_per_block_, ","),
      absl::StrJoin(num_blocks_, ","));
}

IndexingMap RowReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(3, ctx), num_blocks_);
  auto major_reduced = getAffineSymbolExpr(0, ctx);
  auto minor_reduced = getAffineSymbolExpr(1, ctx);
  auto vector_index = getAffineSymbolExpr(2, ctx);

  SmallVector<AffineExpr> indices{
      major_reduced,
      block_id[0] * tile_sizes_per_block_[0] + thread_id[0],
      block_id[1] * tile_sizes_per_block_[1] +
          (minor_reduced * num_threads_[1]) + thread_id[1],
      vector_index,
  };

  auto map = GetIndexingMap(indices, tile_sizes_per_thread_);
  for (auto [result, input_dim] : llvm::zip(indices, input_shape_)) {
    map.AddConstraint(result, {0, input_dim - 1});
  }
  return map;
}

IndexingMap RowReductionFusion::ComputeReductionOutputIndexing(
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

int RowReductionFusion::GetWarpsPerRow() const {
  return CeilOfRatio(num_threads_[1], WarpSize());
}

IndexingMap RowReductionFusion::GetSharedMemoryReductionReadMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  auto lane_id = thread_id[1] % WarpSize();
  return GetThreadIndexingMap({thread_id[0], lane_id},
                              {{thread_id[1], {0, GetWarpsPerRow() - 1}}});
}

IndexingMap RowReductionFusion::GetSharedMemoryWriteMap(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(getAffineDimExpr(0, ctx), num_threads_);
  // The reduced dimension is tiled; each warp writes one element to shared
  // memory (from lane 0).
  auto lane_id = thread_id[1] % WarpSize();
  auto warp_id = thread_id[1].floorDiv(WarpSize());
  return GetThreadIndexingMap({thread_id[0], warp_id}, {{lane_id, {0, 0}}});
}

llvm::SmallVector<mlir::Value> RowReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  const auto& reductions = reduction_heroes_[group_id];

  HloValueMap inits = GetInits(group_id, state);
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  per_thread.reduction_scalars =
      state.ShuffleReduce(reductions, per_thread.reduction_scalars);

  if (GetWarpsPerRow() == 1) {
    // If only a single warp works on an element, we don't need to go through
    // shared memory.
    return EvaluateEpilogue(per_thread.reduction_scalars,
                            std::move(per_thread.outputs), state, group_id,
                            /*symbol_values=*/{});
  }

  return state.ReduceViaSharedMemory(group_id, per_thread, inits);
}

MultiRowReductionFusion::MultiRowReductionFusion(
    const HloFusionAnalysis& analysis, int vector_size)
    : ReductionFusion(analysis) {
  CHECK(reduction_dimensions_.is_row_reduction);
  Vector3 shape = reduction_dimensions_.dimensions;
  input_shape_ = {shape[0], shape[1], shape[2]};
  num_threads_ = GetNumThreads(reduction_dimensions_, vector_size);
  num_blocks_ = {GetNumBlocks(reduction_dimensions_, num_threads_)};
  tile_sizes_per_thread_ = {shape[0], vector_size};
}

std::unique_ptr<ReductionFusion> MultiRowReductionFusion::TryCreate(
    const HloFusionAnalysis& analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  auto reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  auto shape = reduction_dimensions.dimensions;
  // This emitter only supports reductions where the reduced dimension is a
  // power of 2.
  if (shape[kRowMinorReduced] & (shape[kRowMinorReduced] - 1)) {
    VLOG(3) << "MultiRowReductionFusion::TryCreate shape[kRowMinorReduced] = "
            << shape[kRowMinorReduced] << " is not a power of 2";
    return nullptr;
  }

  // Normally, we only consider input types for vectorization. However, in
  // multi-row reductions, the input:output ratio is much higher, so we consider
  // both inputs and outputs.
  int smallest_input_or_output_bits =
      std::min(analysis.input_output_info().smallest_input_dtype_bits,
               analysis.input_output_info().smallest_output_dtype_bits);
  int largest_input_or_output_bits =
      std::max(analysis.input_output_info().smallest_input_dtype_bits,
               analysis.input_output_info().smallest_output_dtype_bits);
  // Handle the case when there are no inputs.
  if (largest_input_or_output_bits == std::numeric_limits<int>::max()) {
    largest_input_or_output_bits =
        analysis.input_output_info().smallest_output_dtype_bits;
  }

  // Our codegen can't currently deal with vectorization across rows, so we
  // limit the vector size to the size of the row. Note that this emitter
  // essentially reverts to the loop emitter in this case, except for side
  // outputs.
  int vector_size = std::min(static_cast<int>(shape[kRowMinorReduced]),
                             64 / smallest_input_or_output_bits);

  // Very large vector sizes for f32 can be detrimental, so we limit the vector
  // size to 16 bytes if we have some >= 32 bit inputs or outputs. This is still
  // a bit on the high side, but remember that we also have very small inputs
  // or outputs.
  if (largest_input_or_output_bits >= 32) {
    VLOG(3) << "MultiRowReductionFusion::TryCreate limiting vector size to 16 "
               "bytes as largest_input_or_output_bits is "
            << largest_input_or_output_bits;
    vector_size = std::min(128 / largest_input_or_output_bits, vector_size);
  }

  // The reduced dimension must fit into a single warp.
  const int64_t warp_size = analysis.device_info().threads_per_warp();
  if (shape[kRowMinorReduced] > warp_size * vector_size) {
    VLOG(3) << "MultiRowReductionFusion::TryCreate reduced dimension "
            << shape[kRowMinorReduced] << " is larger than warp size "
            << warp_size << " * vector size " << vector_size
            << ". Will not use multi-row reduction";
    return nullptr;
  }

  // At the very least, we want to have work for every SM.
  // TODO(jreiffers): This limit is probably too low: if we have as many blocks
  // as SMs, we'll only run about 8 warps per SM, so occupancy will be very low.
  // Further measurements are needed to refine this heuristic.
  int64_t min_desired_blocks = analysis.device_info().core_count();
  while (vector_size > 1 &&
         GetNumBlocks(reduction_dimensions,
                      GetNumThreads(reduction_dimensions, vector_size)) <
             min_desired_blocks) {
    vector_size /= 2;
  }

  // Check again that the reduced dimension fits after potentially reducing the
  // vector size.
  if (shape[kRowMinorReduced] > warp_size * vector_size) {
    VLOG(3) << "MultiRowReductionFusion::TryCreate reduced dimension "
            << shape[kRowMinorReduced] << " is larger than warp size "
            << warp_size << " * vector size " << vector_size
            << ". Will not use multi-row reduction";
    return nullptr;
  }

  VLOG(3) << "MultiRowReductionFusion::TryCreate selected vector_size = "
          << vector_size;
  return std::make_unique<MultiRowReductionFusion>(analysis, vector_size);
}

absl::InlinedVector<int64_t, 4> MultiRowReductionFusion::GetNumThreads(
    const ReductionDimensions& reduction_dimensions, int vector_size) {
  int64_t num_threads_reduced =
      reduction_dimensions.dimensions[kRowMinorReduced] / vector_size;

  constexpr int64_t kThreadsPerBlockTarget = 256;
  int64_t kept_size = reduction_dimensions.dimensions[kRowKept];
  int64_t num_threads_kept = 1;
  if (kept_size * num_threads_reduced <= kThreadsPerBlockTarget) {
    num_threads_kept = kept_size;
  } else {
    num_threads_kept = kThreadsPerBlockTarget / num_threads_reduced;
  }
  return {num_threads_kept, num_threads_reduced};
}

int64_t MultiRowReductionFusion::GetNumBlocks(
    const ReductionDimensions& reduction_dimensions,
    const absl::InlinedVector<int64_t, 4>& num_threads) {
  CHECK_EQ(num_threads.size(), 2)
      << "Expected num_threads to contain the number of threads in the {kept, "
         "reduced} dimensions.";
  return CeilOfRatio(reduction_dimensions.dimensions[kRowKept],
                     num_threads.front());
}

IndexingMap MultiRowReductionFusion::ComputeReductionInputIndexing(
    mlir::MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id = num_blocks_.front() == 1 ? mlir::getAffineConstantExpr(0, ctx)
                                           : mlir::getAffineDimExpr(3, ctx);
  auto major_reduced = getAffineSymbolExpr(0, ctx);
  auto vector_index = getAffineSymbolExpr(1, ctx);

  SmallVector<AffineExpr> indices{
      major_reduced, block_id * num_threads_[0] + thread_id[0],
      thread_id[1] * tile_sizes_per_thread_[1] + vector_index};

  auto map = GetIndexingMap(indices, tile_sizes_per_thread_);
  for (auto [result, input_dim] : llvm::zip(indices, input_shape_)) {
    map.AddConstraint(result, {0, input_dim - 1});
  }
  return map;
}

IndexingMap MultiRowReductionFusion::ComputeReductionOutputIndexing(
    MLIRContext* ctx) const {
  auto thread_id =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);
  auto block_id = num_blocks_.front() == 1 ? mlir::getAffineConstantExpr(0, ctx)
                                           : mlir::getAffineDimExpr(3, ctx);
  IndexingMap projected_index =
      GetIndexingMap(block_id * num_threads_[0] + thread_id[0]);
  projected_index.AddConstraint(thread_id[1] % num_threads_[1], {0, 0});
  // We don't need a constraint on the loop dimensions, because they are removed
  // by GetIndexingMap (since they don't show up in the output index
  // computation).
  return projected_index;
}

llvm::SmallVector<mlir::Value> MultiRowReductionFusion::EmitReduction(
    int group_id, EmitterState& state) const {
  HloValueMap inits = GetInits(group_id, state);
  const auto& reductions = reduction_heroes_[group_id];
  auto per_thread =
      state.EmitPerThreadElements(group_id, inits, state.FusionOutputs());
  auto reduced = state.ShuffleReduce(reductions, per_thread.reduction_scalars,
                                     num_threads_[1] / 2);
  return EvaluateEpilogue(reduced, std::move(per_thread.outputs), state,
                          group_id, /*symbol_values=*/{});
}

std::unique_ptr<ReductionFusion> CreateReductionFusion(
    const HloFusionAnalysis& analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  if (reduction_dimensions.is_row_reduction) {
    auto multi_row_emitter = MultiRowReductionFusion::TryCreate(analysis);
    if (multi_row_emitter != nullptr) {
      return multi_row_emitter;
    }
    return std::make_unique<RowReductionFusion>(analysis);
  }

  const int64_t warp_size = analysis.device_info().threads_per_warp();
  if (warp_size % reduction_dimensions.dimensions[kColMinorKept] == 0) {
    return std::make_unique<SmallColumnReductionFusion>(analysis);
  }
  return std::make_unique<ColumnReductionFusion>(analysis);
}

}  // namespace gpu
}  // namespace xla
