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
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
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
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir_converter::PartitionedComputation;
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
  for (auto [hero, root, is_reduction] :
       llvm::zip(analysis.fusion_heroes(), analysis.fusion_roots(),
                 reduction_info().GetGroups().is_reduction_root)) {
    if (is_reduction) {
      reduction_heroes_.push_back(hero);
      reduction_roots_[hero] = root;
    }
  }
}

bool MlirReductionFusion::IsSupported(const HloFusionAnalysis& analysis) {
  auto info = ReductionInfo::Create(analysis);
  return info.GetGroups().grouped_roots.size() == 1 &&
         !absl::c_linear_search(info.GetGroups().is_reduction_root, false) &&
         info.IsRaceFree();
}

absl::flat_hash_set<const HloInstruction*>
MlirReductionFusion::GetInstructionsWithCustomCodegen(
    const HloFusionInstruction& fusion) const {
  absl::flat_hash_set<const HloInstruction*> instructions_to_isolate(
      reduction_heroes_.begin(), reduction_heroes_.end());
  if (fusion.IsMultiOutputFusion()) {
    instructions_to_isolate.insert(
        fusion.fused_instructions_computation()->root_instruction());
  }
  return instructions_to_isolate;
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

  int output_offset = 0;
  struct HeroInfo {
    SmallVector<Value> inits, outputs, output_indices;
    Value thread_has_output;
  };
  llvm::DenseMap<const HloInstruction*, HeroInfo> hero_info;
  for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
    int num_inputs = hero->operand_count() / 2;
    const auto& computation =
        state.computations.FindPartitionedComputation(hero->parent());
    auto indexing = ComputeThreadIdToOutputIndexing(index, ctx);
    auto& info = hero_info[hero];
    info.inits =
        ProvideParameterRange(computation, hero, num_inputs, num_inputs, {},
                              state.call_target, builder);
    info.outputs = ValueRange(output_args.slice(output_offset, num_inputs));
    info.output_indices = mlir_converter::ApplyAffineMap(
        indexing->GetAffineMap(), thread_and_block_indices, {}, builder);
    info.thread_has_output = mlir_converter::CheckConstraints(
        *indexing, thread_and_block_indices, {}, builder);
    output_offset += num_inputs;
  }

  auto evaluate_epilogue = [&](const HloInstruction* hero, Value output_value) {
    const auto& info = hero_info[hero];
    return EmitEpilogue(reduction_roots_.at(hero), hero, state.call_target,
                        output_value, info.output_indices, builder)
        .front();
  };

  SmallVector<Value> updated_outputs;
  for (auto [hero_index, hero] : llvm::enumerate(reduction_heroes_)) {
    const auto& info = hero_info[hero];
    auto input_indexing = ComputeThreadIdToInputIndexing(hero_index, 0, ctx);
    TF_ASSIGN_OR_RETURN(
        auto accumulated,
        state.EmitPerThreadReducedElements(*input_indexing, hero, info.inits));

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

    auto dest = use_shared ? state.AllocateSharedTiles(hero, shared_tile_size)
                           : info.outputs;
    // Write results to shared or global memory.
    for (auto [value, output] : llvm::zip(accumulated, dest)) {
      // If we don't use shared memory, evaluate the epilogue now.
      if (!use_shared) {
        value = evaluate_epilogue(hero, value);
      }
      updated_outputs.push_back(builder.create<PredicatedInsertOp>(
          use_shared ? shared_write_condition : info.thread_has_output, value,
          output, use_shared ? shared_write_indices : info.output_indices));
    }
  }

  if (!use_shared) {
    // If we didn't go through shared memory, we're done.
    builder.create<mlir::func::ReturnOp>(updated_outputs);
    return absl::OkStatus();
  }

  // Wait for the entire tile to be written.
  auto shared_tiles = builder
                          .create<SyncThreadsOp>(
                              mlir::TypeRange(updated_outputs), updated_outputs)
                          .getResults();
  auto write_outputs = [&](mlir::OpBuilder then_builder, mlir::Location loc) {
    mlir::ImplicitLocOpBuilder b(loc, then_builder);
    int tile_index = 0;
    llvm::SmallVector<Value> updated_outputs;
    for (auto* hero : reduction_heroes_) {
      const auto& info = hero_info[hero];
      // Load from shared memory.
      SmallVector<Value> reduced;
      for (int i = 0; i < hero->operand_count() / 2; ++i) {
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(b.create<PredicatedExtractOp>(
                               shared_read_condition, info.inits[i],
                               shared_tiles[tile_index++], shared_read_indices)
                              .getResult());
      }

      reduced = builder
                    .create<ShuffleReduceOp>(state.GetReducer(hero), reduced,
                                             WarpSize() / 2)
                    .getResults();

      for (auto [output_value, dest] : llvm::zip(reduced, info.outputs)) {
        updated_outputs.push_back(b.create<PredicatedInsertOp>(
            info.thread_has_output, evaluate_epilogue(hero, output_value), dest,
            info.output_indices));
      }
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
    auto values = ProvideParameterRange(
        computations.FindPartitionedComputation(hero->parent()), hero, 0,
        hero->operand_count() / 2, indices, call_target, builder);

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
