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
#include "xla/service/gpu/fusions/transpose_mlir.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using absl::StatusOr;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::ModuleOp;
using mlir::RankedTensorType;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::InsertOp;
using mlir_converter::ApplyAffineMap;
using mlir_converter::CallTargetProvider;
using mlir_converter::PartitionedComputation;

Tiling ComputeTransposeTiling(const TransposeDescription& tiled_transpose) {
  constexpr int kNumRows = 4;
  static_assert(WarpSize() % kNumRows == 0);

  // 3D view over the output shape.
  Vector3 transposed_dims = tiled_transpose.dimensions;
  Vector3 permutation = tiled_transpose.permutation;

  // Note: the supported permutations are their own inverses. Therefore we
  // always use the permutation, even when we want the inverse.
  CHECK((permutation == Vector3{0, 2, 1}) || (permutation == Vector3{2, 1, 0}));

  absl::InlinedVector<int64_t, 4> input_dims{transposed_dims[permutation[0]],
                                             transposed_dims[permutation[1]],
                                             transposed_dims[permutation[2]]};

  // We tile along the minor dimensions pre- and post-transpose.
  absl::InlinedVector<int64_t, 4> tile_sizes{1, 1, 1};
  tile_sizes[permutation[2]] = WarpSize() / kNumRows;
  absl::InlinedVector<int64_t, 4> num_threads{1, 1, WarpSize()};
  num_threads[permutation[2]] = kNumRows;

  return Tiling(input_dims, tile_sizes, num_threads);
}

// Returns transpose heroes that should be codegened via shmem.
absl::flat_hash_set<const HloInstruction*> GetShMemTranposes(
    const HloFusionAnalysis& analysis) {
  absl::flat_hash_set<const HloInstruction*> tranposes_to_tile;
  for (const auto [hero, root] :
       llvm::zip(analysis.fusion_heroes(), analysis.fusion_roots())) {
    if (!GetDescriptionForTiledTransposeEmitter(*root, *hero)) {
      continue;
    }
    tranposes_to_tile.insert(hero);
  }
  return tranposes_to_tile;
}

}  // namespace

MlirTransposeFusion::MlirTransposeFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      tiling_(ComputeTransposeTiling(analysis.tiled_transpose())) {
  for (auto [root, hero] :
       llvm::zip(analysis_.fusion_roots(), analysis_.fusion_heroes())) {
    if (auto transpose = GetDescriptionForTiledTransposeEmitter(*root, *hero)) {
      permutation_ = transpose->permutation;
      break;
    }
  }
}

/*static*/ bool MlirTransposeFusion::IsSupported(
    const HloFusionAnalysis& analysis) {
  return mlir_converter::IsHloConversionSupported(
      analysis.fusion(), analysis.device_info().gpu_compute_capability());
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  const auto& hero = *analysis_.fusion_heroes()[root_index];
  const auto& root = *analysis_.fusion_roots()[root_index];
  if (!GetDescriptionForTiledTransposeEmitter(root, hero)) {
    // Non-transpose roots are elementwise by definition.
    return ComputeThreadIdToInputIndexing(root_index, 0, ctx);
  }

  // The block offsets are permuted, but the thread offsets remain the same.
  auto block_offset = GetBlockOffsetsForTiling(tiling_, ctx)
                          .getSubMap(std::vector<unsigned>{permutation_.begin(),
                                                           permutation_.end()});
  auto thread_offset = GetThreadOffsetsForTiling(tiling_, ctx);
  auto permuted_tiled_shape =
      ShapeUtil::MakeShape(U8, Permute(tiling_.GetShape(), permutation_));

  return ComposeIndexingMaps(
      GetIndexingMapForTiling(block_offset, thread_offset, tiling_),
      GetBitcastMap(permuted_tiled_shape, hero.shape(), ctx));
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  const auto& hero = *analysis_.fusion_heroes()[root_index];

  return ComposeIndexingMaps(
      GetIndexingMapForTiling(tiling_, ctx),
      GetBitcastMap(tiling_.GetXlaShape(), hero.operand(0)->shape(), ctx));
}

LaunchDimensions MlirTransposeFusion::launch_dimensions() const {
  return LaunchDimensions(tiling_.GetNumBlocks(),
                          tiling_.GetNumThreadsPerBlock());
}

// Returns an indexing map with block_x, block_y, block_z set to 0.
IndexingMap GetSharedMemoryWriteIndexingMap(
    const IndexingMap& thread_id_indexing) {
  auto* mlir_context = thread_id_indexing.GetMLIRContext();

  AffineExpr c0 = mlir::getAffineConstantExpr(0, mlir_context);
  AffineExpr th_x = mlir::getAffineDimExpr(0, mlir_context);
  SmallVector<AffineExpr, 3> tile_sizes(3);
  mlir::bindSymbolsList(mlir_context, llvm::MutableArrayRef(tile_sizes));

  IndexingMap shmem_write_indexing{
      AffineMap::get(thread_id_indexing.GetDimensionCount(),
                     thread_id_indexing.GetSymbolCount(),

                     {c0, th_x.floorDiv(32) + 4 * tile_sizes[1], th_x % 32},
                     mlir_context),
      thread_id_indexing.GetDimensionRanges(),
      thread_id_indexing.GetSymbolRanges(),
      thread_id_indexing.GetConstraints()};
  shmem_write_indexing.Simplify();
  return shmem_write_indexing;
}

// Returns an indexing map with block_x, block_y, block_z set to 0 and swapped
// 2nd and 3rd results.
IndexingMap GetSharedMemoryReadIndexingMap(
    const IndexingMap& thread_id_indexing) {
  IndexingMap write_indexing =
      GetSharedMemoryWriteIndexingMap(thread_id_indexing);
  return IndexingMap{write_indexing.GetAffineMap().getSubMap({0, 2, 1}),
                     write_indexing.GetDimensionRanges(),
                     write_indexing.GetSymbolRanges(),
                     write_indexing.GetConstraints()};
}

absl::StatusOr<SmallVector<Value, 4>> MlirTransposeFusion::EmitWriteToShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, ModuleOp module, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const PartitionedComputation& root_computation,
    CallTargetProvider& call_target_provider) const {
  std::vector<int64_t> shmem_tensor_size(tiling_.GetBlockTileSize().begin(),
                                         tiling_.GetBlockTileSize().end());

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  int num_outputs = entry_function.getArguments().size() - num_inputs;

  SmallVector<Value> shmem_intermediate_result;
  for (const auto& [root_index, hero_and_root] : llvm::enumerate(
           llvm::zip(analysis_.fusion_heroes(), analysis_.fusion_roots()))) {
    const HloInstruction* transpose = std::get<0>(hero_and_root);
    const HloInstruction* root = std::get<1>(hero_and_root);
    // Skip non-transpose heroes and handle them in EmitReadFromShMemMlir.
    auto description =
        GetDescriptionForTiledTransposeEmitter(*root, *transpose);
    if (!description.has_value()) {
      continue;
    }

    auto input_indexing = ComputeThreadIdToInputIndexing(
        root_index, /*hero_operand_index=*/0, module.getContext());
    TF_RET_CHECK(input_indexing) << "Indexing is never nullopt";
    IndexingMap shmem_input_indexing =
        GetSharedMemoryWriteIndexingMap(*input_indexing);

    // Allocate shared memory.
    const HloInstruction* transpose_operand = transpose->operand(0);
    auto elem_type = *ConvertPrimitiveTypeToMLIRType(
        transpose_operand->shape().element_type(), builder);
    auto shmem = builder.create<AllocateSharedOp>(
        RankedTensorType::get(shmem_tensor_size, elem_type));

    // Emit loop that writes subgraphs of transpose operands to shmem.
    auto shmem_result =
        EmitThreadLoopNest(
            builder, {shmem}, *input_indexing,
            [&](ValueRange output_tensors, ValueRange dim_values,
                ValueRange symbol_values) -> SmallVector<Value> {
              auto input_indices =
                  ApplyAffineMap(input_indexing->GetAffineMap(), dim_values,
                                 symbol_values, builder);
              auto shmem_indices =
                  ApplyAffineMap(shmem_input_indexing.GetAffineMap(),
                                 dim_values, symbol_values, builder);

              auto result_scalars = mlir_converter::ProvideParameter(
                  root_computation, transpose, /*operand_index=*/0,
                  input_indices, call_target_provider, builder);

              SmallVector<Value> result_tensors;
              result_tensors.reserve(num_outputs);
              for (auto [tensor, value] :
                   llvm::zip(output_tensors, result_scalars)) {
                result_tensors.push_back(
                    builder.create<InsertOp>(value, tensor, shmem_indices));
              }
              return result_tensors;
            });
    shmem_intermediate_result.append(shmem_result.begin(), shmem_result.end());
  }

  return shmem_intermediate_result;
}

absl::Status MlirTransposeFusion::EmitReadFromShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, ModuleOp module, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    CallTargetProvider& call_target_provider,
    const absl::flat_hash_set<
        const mlir_converter::PartitionedComputation::Subgraph*>&
        hero_subgraphs,
    const mlir_converter::PartitionedComputation::Subgraph* root_subgraph,
    ValueRange shmem_tensors) const {
  SmallVector<Value, 4> result_tensors;

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();

  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  int transpose_hero_count = 0;
  for (const auto& [root_index, hero_and_root] : llvm::enumerate(
           llvm::zip(analysis_.fusion_heroes(), analysis_.fusion_roots()))) {
    const HloInstruction* transpose = std::get<0>(hero_and_root);
    const HloInstruction* root = std::get<1>(hero_and_root);

    auto output_indexing =
        ComputeThreadIdToOutputIndexing(root_index, module.getContext());
    TF_RET_CHECK(output_indexing) << "Indexing is never nullopt";
    IndexingMap shmem_output_indexing =
        GetSharedMemoryReadIndexingMap(*output_indexing);
    auto description =
        GetDescriptionForTiledTransposeEmitter(*root, *transpose);

    SmallVector<Value> result_scalars;
    if (description.has_value()) {
      auto subresult_tensors =
          EmitThreadLoopNest(
              builder, output_tensor_args, *output_indexing,
              [&](ValueRange output_tensors, ValueRange dim_values,
                  ValueRange symbol_values) -> SmallVector<Value> {
                auto output_indices =
                    ApplyAffineMap(output_indexing->GetAffineMap(), dim_values,
                                   symbol_values, builder);
                auto shmem_indices =
                    ApplyAffineMap(shmem_output_indexing.GetAffineMap(),
                                   dim_values, symbol_values, builder);

                if (!hero_subgraphs.contains(root_subgraph)) {
                  // Transpose is not the root of the computation. Call
                  // epilogue function.
                  auto epilogue_fn = call_target_provider(root);

                  // Generate the operands for the root function: input tensors
                  // + output indices + shmem element.
                  SmallVector<Value> operands(
                      entry_function.getArguments().take_front(num_inputs));
                  absl::c_copy(output_indices, std::back_inserter(operands));

                  operands.push_back(builder.create<ExtractOp>(
                      shmem_tensors[transpose_hero_count], shmem_indices));
                  auto pure_call =
                      builder.create<PureCallOp>(epilogue_fn, operands);
                  result_scalars.append(pure_call.getResults().begin(),
                                        pure_call.getResults().end());
                } else {
                  result_scalars.push_back(builder.create<ExtractOp>(
                      shmem_tensors[transpose_hero_count], shmem_indices));
                }
                ++transpose_hero_count;
                SmallVector<Value> results;
                results.reserve(output_tensor_args.size());
                for (auto [tensor, value] :
                     llvm::zip(output_tensors, result_scalars)) {
                  results.push_back(
                      builder.create<InsertOp>(value, tensor, output_indices));
                }
                return results;
              });
      result_tensors.append(subresult_tensors.begin(), subresult_tensors.end());
    } else {
      auto indexing = ComputeThreadIdToOutputIndexing(0, module.getContext());
      TF_RET_CHECK(indexing) << "Indexing is never nullopt";

      int num_inputs =
          fusion.fused_instructions_computation()->num_parameters();
      auto output_tensor_args =
          entry_function.getArguments().drop_front(num_inputs);

      auto subresult_tensors =
          EmitThreadLoopNest(
              builder, output_tensor_args, *indexing,
              [&](ValueRange output_tensors, ValueRange dim_values,
                  ValueRange symbol_values) -> SmallVector<Value> {
                auto output_indices =
                    ApplyAffineMap(indexing->GetAffineMap(), dim_values,
                                   symbol_values, builder);

                // Generate the operands for the root function: input tensors +
                // output indices.
                llvm::SmallVector<Value> operands(
                    entry_function.getArguments().take_front(num_inputs));
                absl::c_copy(output_indices, std::back_inserter(operands));

                auto result_scalars = builder.create<PureCallOp>(
                    call_target_provider(root), operands);

                SmallVector<Value> results;
                results.reserve(output_tensor_args.size());
                for (auto [tensor, value] :
                     llvm::zip(output_tensors, result_scalars.getResults())) {
                  results.push_back(
                      builder.create<InsertOp>(value, tensor, output_indices));
                }
                return results;
              });
      result_tensors.append(subresult_tensors.begin(), subresult_tensors.end());
    }
  }
  builder.create<ReturnOp>(result_tensors);
  return absl::OkStatus();
}

absl::Status MlirTransposeFusion::EmitMlir(
    mlir::ModuleOp module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  // Render subgraphs.
  auto shmem_transposes = GetShMemTranposes(analysis_);
  mlir_converter::PartitionedComputations computations(
      fusion.fused_instructions_computation(), shmem_transposes);
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  const auto& root_graph = root_computation.GetRootSubgraph();
  auto subgraph_to_mlir_fn = computations.DeclareFunctions(module);

  // Transpose is inlined in the entry function. This is needed for
  // mlir_converter::ProvideParameter to correctly get parameter value.
  absl::flat_hash_set<const mlir_converter::PartitionedComputation::Subgraph*>
      hero_subgraphs;
  for (const HloInstruction* transpose : shmem_transposes) {
    auto& hero_graph = root_computation.FindSubgraph(transpose);
    subgraph_to_mlir_fn.extract(&hero_graph).mapped().erase();
    subgraph_to_mlir_fn[&hero_graph] = entry_function;
    hero_subgraphs.insert(&hero_graph);
  }
  auto call_targets =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (hero_subgraphs.contains(&subgraph)) {
        continue;
      }
      TF_RETURN_IF_ERROR(mlir_converter::SubgraphToMlirFunction(
          comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_targets));
    }
  }

  // Write intermediate results to shmem.
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  TF_ASSIGN_OR_RETURN(
      auto shmem_tensors,
      EmitWriteToShMemMlir(builder, module, entry_function, fusion,
                           root_computation, call_targets));
  // Sync GPU threads before reading from shmem.
  auto sync_threads = builder.create<SyncThreadsOp>(
      mlir::TypeRange(shmem_tensors), shmem_tensors);

  // Read intermediate results from shmem and compute epilogues.
  return EmitReadFromShMemMlir(builder, module, entry_function, fusion,
                               call_targets, hero_subgraphs, &root_graph,
                               sync_threads.getResults());
}

}  // namespace gpu
}  // namespace xla
