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

LaunchDimensions MlirReductionFusion::launch_dimensions() const {
  size_t blocks_y = groups_.grouped_roots.size();
  return {se::BlockDim(/*x=*/Product(num_blocks_),
                       /*y=*/static_cast<int64_t>(blocks_y), /*z=*/1),
          se::ThreadDim(/*x=*/Product(num_threads_),
                        /*y=*/1, /*z=*/1)};
}

std::optional<IndexingMap> MlirReductionFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  auto block_offsets = GetBlockOffsetsForTiling(
      num_blocks_, tile_sizes_per_block_, tiled_shape_.size(), ctx);
  auto thread_offsets = GetThreadOffsetsForTiling(
      num_threads_, tile_sizes_per_thread_, tiled_shape_.size(), ctx);
  int64_t num_threads_per_block = Product(num_threads_);
  int64_t total_num_blocks = Product(num_blocks_);

  if (!groups_.is_reduction_root[root_index]) {
    auto map = ComposeIndexingMaps(
        GetIndexingMapForTiling(block_offsets, thread_offsets,
                                num_threads_per_block, total_num_blocks,
                                tile_sizes_per_thread_, tiled_shape_),
        GetBitcastMap(ShapeUtil::MakeShape(PrimitiveType::F32, tiled_shape_),
                      analysis_.fusion_root(root_index).shape(), ctx));
    AddGroupIdConstraint(map, root_index, ctx);
    return map;
  }
  const auto& hero = analysis_.fusion_hero(root_index).instruction();

  auto thread_ids =
      DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx), num_threads_);

  auto physical_shape =
      ShapeUtil::DeleteDimensions(hero.dimensions(), hero.operand(0)->shape());
  std::vector<DimVar> dimension_ranges{
      {{0, num_threads_per_block - 1}},
      {},
      {},
      {{0, total_num_blocks - 1}},
      {{0, static_cast<int64_t>(groups_.grouped_roots.size() - 1)}},
      {},
  };

  constexpr int kRowKept = ReductionDimensions::kRowKeptDimension;
  constexpr int kRowMinorReduced =
      ReductionDimensions::kRowMinorReducedDimension;

  constexpr int kColMajorKept = ReductionDimensions::kColMajorKeptDimension;
  constexpr int kColMinorKept = ReductionDimensions::kColMinorKeptDimension;
  constexpr int kColReduced = ReductionDimensions::kColReducedDimension;

  auto map = [&]() {
    if (is_row_reduction_) {
      IndexingMap linear_index(
          mlir::AffineMap::get(
              6, 0, block_offsets.getResult(kRowKept) + thread_ids[kRowKept],
              ctx),
          dimension_ranges, /*range_vars=*/{}, /*rt_vars=*/{});
      int rows_per_warp = GetRowsPerWarp();
      if (rows_per_warp > 1) {
        linear_index.AddConstraint(
            thread_ids[kRowMinorReduced] % (WarpSize() / rows_per_warp),
            {0, 0});
      } else {
        linear_index.AddConstraint(thread_ids[kRowMinorReduced], {0, 0});
      }
      return ComposeIndexingMaps(
          linear_index,
          GetBitcastMap(ShapeUtil::MakeShape(PRED, {tiled_shape_[kRowKept]}),
                        physical_shape, ctx));
    }

    mlir::SmallVector<mlir::AffineExpr> projected_dims{
        block_offsets.getResult(kColMajorKept),
        block_offsets.getResult(kColMinorKept) + thread_ids[kColReduced]};
    std::vector<RangeVar> range_vars;
    if (thread_ids.size() == 4) {
      int vector_size = tile_sizes_per_thread_.back();
      range_vars.push_back({0, vector_size - 1});
      projected_dims.push_back(mlir::getAffineSymbolExpr(0, ctx));
    }
    IndexingMap projected_index(
        mlir::AffineMap::get(6, range_vars.size(), projected_dims, ctx),
        dimension_ranges, range_vars, /*rt_vars=*/{});

    projected_index.AddConstraint(
        mlir::getAffineDimExpr(
            KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx) %
            WarpSize(),
        {0, 0});
    if (!is_row_reduction_) {
      projected_index.AddConstraint(
          projected_index.GetAffineMap().getResult(1),
          {0, tiled_shape_[ReductionDimensions::kColMinorKeptDimension] - 1});
    }

    return ComposeIndexingMaps(
        projected_index,
        GetBitcastMap(
            ShapeUtil::DeleteDimension(
                ReductionDimensions::kColReducedDimension,
                ShapeUtil::MakeShape(PrimitiveType::F32, tiled_shape_)),
            physical_shape, ctx));
  }();

  AddGroupIdConstraint(map, root_index, ctx);
  return map;
}

std::optional<IndexingMap> MlirReductionFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
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

  auto block_offsets = GetBlockOffsetsForTiling(
      num_blocks_, tile_sizes_per_block_, tiled_shape_.size(), ctx);
  auto thread_offsets = GetThreadOffsetsForTiling(
      num_threads_, tile_sizes_per_thread_, tiled_shape_.size(), ctx);
  auto map = ComposeIndexingMaps(
      GetIndexingMapForTiling(block_offsets, thread_offsets,
                              Product(num_threads_), Product(num_blocks_),
                              tile_sizes_per_thread_, tiled_shape_),
      GetBitcastMap(ShapeUtil::MakeShape(PrimitiveType::F32, tiled_shape_),
                    hero.operand(hero_operand_index)->shape(), ctx));
  AddGroupIdConstraint(map, root_index, ctx);
  return map;
}

void MlirReductionFusion::AddGroupIdConstraint(IndexingMap& map,
                                               int64_t root_index,
                                               mlir::MLIRContext* ctx) const {
  // Only threads with the right y block index actually do anything for each
  // particular root.
  int group_index = groups_.group_id_per_root[root_index];
  map.AddConstraint(
      mlir::getAffineDimExpr(KernelFusionInterface::kIndexingMapBlockIdxDims[1],
                             ctx),
      {group_index, group_index});
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

MlirReductionFusion::MlirReductionFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  auto* hero_reduction = analysis.FindHeroReduction();
  CHECK_NE(hero_reduction, nullptr);
  Shape input_shape = hero_reduction->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*hero_reduction);
  auto shape = reduction_dimensions.dimensions;

  is_row_reduction_ = reduction_dimensions.is_row_reduction;
  VLOG(10) << "is_row_reduction " << is_row_reduction_ << " " << shape[0] << " "
           << shape[1] << " " << shape[2];
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);

  int64_t num_threads_y = is_row_reduction_ ? 1 : WarpSize();
  int64_t rows_per_warp =
      is_row_reduction_
          ? RowReductionGetRowsPerWarp(
                shape[ReductionDimensions::kRowMinorReducedDimension])
          : 1;
  int64_t num_threads_x = [&] {
    if (is_row_reduction_) {
      if (rows_per_warp > 1) {
        return shape[ReductionDimensions::kRowMinorReducedDimension];
      }
      int64_t max_block_size =
          MinThreadsXRowReduction(hero_reduction->GetModule()->config());
      return std::min(
          max_block_size,
          RoundUpTo(
              CeilOfRatio(shape[ReductionDimensions::kRowMinorReducedDimension],
                          reduction_tiling
                              [ReductionDimensions::kRowMinorReducedDimension]),
              WarpSize()));
    }
    return WarpSize();
  }();

  // If we're limited by the size of the x dimension, add additional parallelism
  // in the y dimension. The code generator doesn't currently support
  // parallelizing the z dimension (major reduced dimensions). The general
  // recommendation is to use between 128 and 512 threads, so we just go for
  // 256. See https://forums.developer.nvidia.com/t/55529
  constexpr int64_t kThreadsPerBlockTarget = 256;
  if (is_row_reduction_ && num_threads_x * 2 <= kThreadsPerBlockTarget) {
    int64_t kept_size =
        reduction_dimensions.dimensions[ReductionDimensions::kRowKeptDimension];
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

  int vector_size = GetVectorSize(analysis, reduction_dimensions, num_threads_x,
                                  reduction_tiling, /*for_mlir=*/true);

  num_threads_ =
      absl::InlinedVector<int64_t, 4>{1, num_threads_y, num_threads_x};
  tiled_shape_ = {shape[0], shape[1], shape[2] / vector_size};
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
                 CeilOfRatio(tiled_shape_[i], num_threads_[i]));
  }
  if (rows_per_warp > 1) {
    // If we produce more than one element per thread, that means the reduced
    // dimension is small and it can't be tiled - we already have more threads
    // in a warp than the size of the reduced dimension. The code generator
    // doesn't currently support tiling the kept dimension, because it just
    // uses the thread ID as the coordinate.
    tile_sizes_per_thread_[2] = 1;
  }
  if (vector_size != 1 || !is_row_reduction_) {
    num_threads_.push_back(1);  // The vector dimension is a loop.
    tiled_shape_.push_back(vector_size);
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
  if (is_row_reduction_) {
    num_threads_.push_back(1);
    tiled_shape_.push_back(1);
    tile_sizes_per_thread_.push_back(1);
  }

  tile_sizes_per_block_.resize(tiled_shape_.size());
  num_blocks_.resize(tiled_shape_.size());
  for (int64_t i = 0; i < tiled_shape_.size(); ++i) {
    tile_sizes_per_block_[i] = tile_sizes_per_thread_[i] * num_threads_[i];
    CHECK_NE(tile_sizes_per_block_[i], 0);
    num_blocks_[i] = CeilOfRatio(tiled_shape_[i], tile_sizes_per_block_[i]);
    CHECK_NE(num_blocks_[i], 0);
  }

  is_race_free_ = ReductionIsRaceFree(hero_reduction->GetModule()->config(),
                                      reduction_dimensions);
  groups_ = GroupDisjointReductions(analysis, /*for_mlir=*/true);
  first_reduce_ = hero_reduction;

  CHECK(is_race_free_)
      << "Non-race-free reductions should have been decomposed. Did "
         "tree_reduction_rewriter run?";

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

std::vector<mlir_converter::EpilogueSpecification>
MlirReductionFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  mlir::MLIRContext* mlir_context) const {
  std::vector<mlir_converter::EpilogueSpecification> epilogues;
  epilogues.reserve(reduction_heroes_.size());
  for (const auto& [heroes, roots] :
       llvm::zip(reduction_heroes_, reduction_roots_)) {
    epilogues.push_back(
        mlir_converter::EpilogueSpecification::FromOutputIndexing(
            analysis_, heroes, roots, *this, mlir_context));
  }
  return epilogues;
}

int MlirReductionFusion::GetRowsPerWarp() const {
  if (!is_row_reduction_) return 1;
  return RowReductionGetRowsPerWarp(
      tiled_shape_[ReductionDimensions::kRowMinorReducedDimension]);
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
  auto* ctx = state.entry_function.getContext();

  // The number of warps working on one element in a row reduction.
  int num_warps_row =
      num_threads_[ReductionDimensions::kRowMinorReducedDimension] / WarpSize();

  Value zero = b.create<ma::ConstantIndexOp>(0);
  Value one = b.create<ma::ConstantIndexOp>(1);
  Value lane_id = b.create<mlir::gpu::LaneIdOp>();
  Value is_first_lane =
      b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, lane_id, zero);
  Value thread_id = state.thread_and_block_ids[0];
  Value cst_true = b.create<ma::ConstantOp>(b.getOneAttr(b.getI1Type()));

  auto thread_indexing = GetBitcastMap(
      ShapeUtil::MakeShapeWithDescendingLayout(U8, {Product(num_threads_)}),
      ShapeUtil::MakeShapeWithDescendingLayout(U8, num_threads_), ctx);
  auto thread_ids =
      mlir_converter::ApplyIndexing(thread_indexing, {thread_id}, {}, b);

  Value warp_id = b.create<ma::DivUIOp>(
      is_row_reduction_
          ? thread_ids[ReductionDimensions::kRowMinorReducedDimension]
          : thread_id,
      b.create<ma::ConstantIndexOp>(WarpSize()));
  // The number of results per thread.
  int64_t vector_size = tile_sizes_per_thread_.back();
  Value vector_size_cst = b.create<ma::ConstantIndexOp>(vector_size);

  std::vector<int64_t> shared_tile_size;
  std::function<SmallVector<Value>(Value, ImplicitLocOpBuilder&)>
      shared_write_indices;
  std::function<SmallVector<Value>(Value, ImplicitLocOpBuilder&)>
      shared_read_indices;
  Value shared_write_condition = cst_true;
  Value shared_read_condition = cst_true;
  if (!is_row_reduction_) {
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
  } else if (GetRowsPerWarp() == 1 && num_warps_row > 1) {
    CHECK_EQ(vector_size, 1);
    constexpr int kKept = ReductionDimensions::kRowKeptDimension;
    shared_tile_size = {num_threads_[kKept], num_warps_row};
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
  if (is_row_reduction_) {
    for (auto* reduction : reductions) {
      auto reducer = state.GetReducer(reduction);
      int max_dist = WarpSize() / 2 / GetRowsPerWarp();
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

  if (is_row_reduction_) {
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
  auto* ctx = builder.getContext();
  auto block_offsets =
      GetBlockOffsetsForTiling(owner.num_blocks_, owner.tile_sizes_per_block_,
                               owner.tiled_shape_.size(), ctx);

  auto thread_offsets = GetThreadOffsetsForTiling(
      owner.num_threads_, owner.tile_sizes_per_thread_,
      owner.tiled_shape_.size(), ctx);
  auto tile_indexing = GetIndexingMapForTiling(
      block_offsets, thread_offsets, Product(owner.num_threads_),
      Product(owner.num_blocks_), owner.tile_sizes_per_thread_,
      owner.tiled_shape_);
  tile_indexing
      .GetMutableDimensionBound(
          KernelFusionInterface::kIndexingMapBlockIdxDims[1])
      .upper = owner.reduction_heroes_.size();
  tile_indexing.Simplify();
  bool vectorize = owner.tile_sizes_per_thread_.back() > 1;

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
          GetBitcastMap(
              ShapeUtil::MakeShape(PrimitiveType::F32, owner.tiled_shape_),
              input_shape, builder.getContext()),
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
