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

using mlir_converter::PartitionedComputation;
using mlir_converter::PartitionedComputations;

struct MlirReductionFusion::EmitterState {
  // Uses the given indexing map to reduce a subset of the inputs in a single
  // thread. The subset may be a single element.
  absl::StatusOr<llvm::SmallVector<mlir::Value>> EmitPerThreadReducedElements(
      const IndexingMap& input_indexing, const HloInstruction* hero,
      mlir::ValueRange inits);

  mlir::func::FuncOp GetReducer(const HloInstruction* hero) const {
    return call_target(hero->called_computations()[0]->root_instruction());
  }

  llvm::SmallVector<mlir::Value> AllocateSharedTiles(
      const HloInstruction* hero, absl::Span<const int64_t> shape);

  const MlirReductionFusion& owner;
  mlir::ModuleOp module;
  mlir::func::FuncOp entry_function;
  const HloFusionInstruction& fusion;
  std::unique_ptr<PartitionedComputations> computations;
  absl::flat_hash_map<const PartitionedComputation::Subgraph*,
                      mlir::func::FuncOp>
      subgraph_to_mlir_fn;
  mlir_converter::CallTargetProvider call_target;
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
  return info.GetGroups().grouped_roots.size() == 1 && info.IsRowReduction() &&
         info.GetRowsPerWarp() == 1 &&
         !absl::c_linear_search(info.GetGroups().is_reduction_root, false) &&
         info.IsRaceFree();
}

absl::Status MlirReductionFusion::EmitMlir(
    mlir::ModuleOp module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  // Reduction groups will probably be implemented in a separate pass, since
  // they share nothing by definition.
  TF_RET_CHECK(reduction_info().GetGroups().grouped_roots.size() == 1)
      << "Only one reduction group is supported.";

  absl::flat_hash_set<const HloInstruction*> instructions_to_isolate(
      reduction_heroes_.begin(), reduction_heroes_.end());
  if (fusion.IsMultiOutputFusion()) {
    instructions_to_isolate.insert(
        fusion.fused_instructions_computation()->root_instruction());
  }
  auto pc = std::make_unique<PartitionedComputations>(
      fusion.fused_instructions_computation(), instructions_to_isolate);
  auto subgraph_to_mlir_fn = pc->DeclareFunctions(module);
  // Erase subgraphs for all reductions - these will be code generated with
  // custom logic.
  for (auto* root : reduction_heroes_) {
    subgraph_to_mlir_fn.extract(&pc->FindSubgraph(root)).mapped().erase();
  }
  // Erase the subgraph for the tuple op.
  if (fusion.IsMultiOutputFusion()) {
    subgraph_to_mlir_fn
        .extract(&pc->FindSubgraph(
            fusion.fused_instructions_computation()->root_instruction()))
        .mapped()
        .erase();
  }

  auto call_target = pc->CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : pc->partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (subgraph_to_mlir_fn.contains(&subgraph)) {
        TF_RETURN_IF_ERROR(mlir_converter::SubgraphToMlirFunction(
            comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_target));
      }
    }
  }

  for (auto* root : reduction_heroes_) {
    subgraph_to_mlir_fn[&pc->FindSubgraph(root)] = entry_function;
  }

  EmitterState state{*this,
                     module,
                     entry_function,
                     fusion,
                     std::move(pc),
                     subgraph_to_mlir_fn,
                     std::move(call_target),
                     {module.getLoc(), entry_function}};
  state.builder.setInsertionPointToStart(entry_function.addEntryBlock());

  if (reduction_info().IsRowReduction() &&
      reduction_info().GetRowsPerWarp() == 1) {
    return EmitRowReduction(state);
  }
  return absl::UnimplementedError("Not implemented");
}

absl::Status MlirReductionFusion::EmitRowReduction(EmitterState& state) const {
  auto& builder = state.builder;
  const auto& tiling = reduction_info().GetTiling();

  // We need one shared element per warp.
  int num_warps = tiling.GetThreadsPerBlock()
                      [ReductionDimensions::kRowMinorReducedDimension] /
                  WarpSize();
  std::vector<int64_t> shared_tile_size{
      tiling.GetThreadsPerBlock()[ReductionDimensions::kRowKeptDimension],
      num_warps};

  auto ctx = state.module.getContext();
  auto input_indexing = ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, ctx);
  TF_RET_CHECK(input_indexing) << "Indexing is never nullopt";

  auto zero = builder.create<mlir::arith::ConstantIndexOp>(0);
  auto lane_id = builder.create<mlir::gpu::LaneIdOp>();
  auto is_first_lane = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, lane_id, zero);
  auto thread_id = EmitThreadId(builder, 0);
  auto block_id = EmitBlockId(builder, 0);

  auto thread_ids = mlir_converter::ApplyAffineMap(
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0,
          DelinearizeInBoundsIndex(mlir::getAffineDimExpr(0, ctx),
                                   tiling.GetThreadsPerBlock(),
                                   tiling.GetThreadStrides()),
          ctx),
      {thread_id}, {}, builder);

  auto warp_id = builder.create<mlir::arith::DivUIOp>(
      thread_ids[ReductionDimensions::kRowMinorReducedDimension],
      builder.create<mlir::arith::ConstantIndexOp>(WarpSize()));

  ConstHloInstructionMap<llvm::SmallVector<mlir::Value>> inits;
  for (auto* hero : reduction_heroes_) {
    int num_inputs = hero->operand_count() / 2;
    const auto& computation =
        state.computations->FindPartitionedComputation(hero->parent());
    TF_ASSIGN_OR_RETURN(
        inits[hero],
        ProvideParameterRange(computation, hero, num_inputs, num_inputs, {},
                              state.call_target, builder));
  }

  llvm::SmallVector<mlir::Value> shared_tiles;
  for (auto* hero : reduction_heroes_) {
    TF_ASSIGN_OR_RETURN(
        auto accumulated,
        state.EmitPerThreadReducedElements(*input_indexing, hero, inits[hero]));
    auto reducer = state.GetReducer(hero);
    accumulated =
        builder.create<ShuffleReduceOp>(reducer, accumulated, WarpSize() / 2)
            .getResults();

    // Write results to shared memory.
    llvm::SmallVector<mlir::Value> tiles_for_this_hero =
        state.AllocateSharedTiles(hero, shared_tile_size);
    for (auto [value, tile] : llvm::zip(accumulated, tiles_for_this_hero)) {
      shared_tiles.push_back(builder.create<PredicatedInsertOp>(
          is_first_lane, value, tile,
          mlir::ValueRange{thread_ids[ReductionDimensions::kRowKeptDimension],
                           warp_id}));
    }
  }

  // Wait for the entire tile to be written.
  shared_tiles =
      builder.create<SyncThreadsOp>(mlir::TypeRange(shared_tiles), shared_tiles)
          .getResults();

  auto outputs = state.entry_function.getArguments().drop_front(
      state.fusion.fused_parameters().size());

  auto write_outputs = [&](mlir::OpBuilder b, mlir::Location loc) {
    int tile_index = 0;
    int next_output = 0;
    llvm::SmallVector<mlir::Value> shared_index{
        thread_ids[ReductionDimensions::kRowKeptDimension], lane_id};
    llvm::SmallVector<mlir::Value> updated_outputs;
    for (auto [index, hero] : llvm::enumerate(reduction_heroes_)) {
      auto output_indexing = ComputeThreadIdToOutputIndexing(index, ctx);
      auto output_indices = mlir_converter::ApplyAffineMap(
          output_indexing->GetAffineMap(),
          {thread_id, zero, zero, block_id, zero, zero}, {}, builder);

      auto is_in_bounds = b.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult,
          thread_ids[ReductionDimensions::kRowMinorReducedDimension],
          b.create<mlir::arith::ConstantIndexOp>(loc, num_warps));

      // Load from shared memory.
      llvm::SmallVector<mlir::Value> reduced;
      for (int i = 0; i < hero->operand_count() / 2; ++i) {
        // If a warp didn't write anything, use the init values instead.
        reduced.push_back(b.create<PredicatedExtractOp>(
                               loc, is_in_bounds, inits[hero][i],
                               shared_tiles[tile_index++], shared_index)
                              .getResult());
      }

      if (num_warps > 1) {
        auto reducer = state.GetReducer(hero);
        reduced =
            builder.create<ShuffleReduceOp>(reducer, reduced, WarpSize() / 2)
                .getResults();
      }

      const auto* root = reduction_roots_.at(hero);
      for (int i = 0; i < hero->operand_count() / 2; ++i) {
        auto output_value = reduced[i];
        // If we have an epilogue, evaluate it now.
        if (root != hero) {
          llvm::SmallVector<mlir::Value> arguments =
              mlir::ValueRange(state.entry_function.getArguments().take_front(
                  root->parent()->num_parameters()));
          absl::c_copy(output_indices, std::back_inserter(arguments));
          arguments.push_back(output_value);
          output_value =
              builder.create<PureCallOp>(state.call_target(root), arguments)
                  .getResult(0);
        }
        updated_outputs.push_back(b.create<PredicatedInsertOp>(
            loc, is_first_lane, output_value, outputs[next_output++],
            output_indices));
      }
    }

    b.create<mlir::scf::YieldOp>(loc, updated_outputs);
  };

  auto is_first_warp = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, warp_id, zero);
  auto written = builder.create<mlir::scf::IfOp>(
      is_first_warp, write_outputs, [&](mlir::OpBuilder b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(loc, outputs);
      });
  builder.create<mlir::func::ReturnOp>(written.getResults());

  return absl::OkStatus();
}

absl::StatusOr<llvm::SmallVector<mlir::Value>>
MlirReductionFusion::EmitterState::EmitPerThreadReducedElements(
    const IndexingMap& input_indexing, const HloInstruction* hero,
    mlir::ValueRange inits) {
  auto body_builder = [&](mlir::ValueRange outputs, mlir::ValueRange dim_values,
                          mlir::ValueRange symbol_values)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    auto indices = mlir_converter::ApplyAffineMap(
        input_indexing.GetAffineMap(), dim_values, symbol_values, builder);
    llvm::SmallVector<mlir::Value> operands(
        entry_function.getArguments().take_front(
            fusion.fused_parameters().size()));
    absl::c_copy(indices, std::back_inserter(operands));
    TF_ASSIGN_OR_RETURN(
        auto values,
        ProvideParameterRange(
            computations->FindPartitionedComputation(hero->parent()), hero, 0,
            hero->operand_count() / 2, indices, call_target, builder));

    llvm::SmallVector<mlir::Value> reduce_args = outputs;
    reduce_args.append(values.begin(), values.end());
    return builder.create<PureCallOp>(GetReducer(hero), reduce_args)
        .getResults();
  };
  return owner.EmitLoopNest(builder, inits, input_indexing, body_builder);
}

llvm::SmallVector<mlir::Value>
MlirReductionFusion::EmitterState::AllocateSharedTiles(
    const HloInstruction* hero, absl::Span<const int64_t> shape) {
  llvm::SmallVector<mlir::Value> tiles;
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
