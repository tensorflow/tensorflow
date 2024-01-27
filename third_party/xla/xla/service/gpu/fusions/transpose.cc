/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/transpose.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

TilingScheme ComputeTransposeTilingScheme(
    const TransposeDescription& tiled_transpose) {
  constexpr int kNumRows = 4;
  static_assert(WarpSize() % kNumRows == 0);

  // 3D view over the output shape.
  Vector3 transposed_dims = tiled_transpose.dimensions;
  Vector3 permutation = tiled_transpose.permutation;

  // Note: the supported permutations are their own inverses. Therefore we
  // always use the permutation, even when we want the inverse.
  CHECK((permutation == Vector3{0, 2, 1}) || (permutation == Vector3{2, 1, 0}));

  Vector3 input_dims{transposed_dims[permutation[0]],
                     transposed_dims[permutation[1]],
                     transposed_dims[permutation[2]]};
  // The tiling corresponds to the two minor dimensions before and after the
  // transpose. The remaining dimension is the batch dimension.
  // The order is {batch, minor post-transpose, minor pre-transpose}.
  //
  // Examples for transposed_dims {200, 300, 700}:
  // order             {0, 2, 1}         {2, 1, 0}
  // input_dims        {200, 700, 300}   {700, 300, 200}
  // tiled_shape       {200, 700, 300}   {300, 700, 200}
  // tile -> input     {0, 1, 2}         {1, 0, 2}
  Vector3 tiled_shape{input_dims[1 - permutation[2]], transposed_dims[2],
                      input_dims[2]};

  Vector3 tile_sizes{1, WarpSize() / kNumRows, 1};
  Vector3 num_threads{1, kNumRows, WarpSize()};

  return TilingScheme(
      /*dims_in_elems=*/tiled_shape,
      /*tile_sizes=*/tile_sizes,
      /*num_threads=*/num_threads,
      /*vector_size=*/1,
      /*scaling_factor=*/1);
}

Vector3 TileToInoutPermutation(Vector3 permutation) {
  // See ComputeTransposeTilingScheme.
  // Note: this is also the tile to output permutation because we swap the
  // last two components.
  return permutation[2] == 1 ? Vector3{0, 1, 2} : Vector3{1, 0, 2};
}

llvm::GlobalVariable* AllocateShared(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name) {
  CHECK(!dimensions_major_to_minor.empty());
  llvm::Type* ty = element_type;
  for (auto dim : llvm::reverse(dimensions_major_to_minor)) {
    ty = llvm::ArrayType::get(ty, dim);
  }
  ty = llvm::ArrayType::get(ty, tiling_scheme.GetThreadIdScalingFactor());
  return llvm_ir::AllocateSharedMemoryTile(
      builder->GetInsertBlock()->getModule(), ty, buffer_name);
}

void MaybeEmitFenceForAMDGPU(llvm::IRBuilder<>* builder,
                             IrEmitterContext& ir_emitter_context) {
  auto* module = builder->GetInsertBlock()->getModule();
  if (IsAMDGPU(module) &&
      ir_emitter_context.rocm_compute_capability().fence_before_barrier()) {
    builder->CreateFence(
        llvm::AtomicOrdering::SequentiallyConsistent,
        builder->getContext().getOrInsertSyncScopeID("workgroup"));
  }
}

void EmitSyncThreads(llvm::IRBuilder<>* builder,
                     IrEmitterContext& ir_emitter_context) {
  MaybeEmitFenceForAMDGPU(builder, ir_emitter_context);
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, builder);
}

llvm_ir::IrArray::Index PermuteIndex(const llvm_ir::IrArray::Index& index,
                                     absl::Span<const int64_t> permutation) {
  return llvm_ir::IrArray::Index{Permute(index.multidim(), permutation),
                                 Permute(index.dims(), permutation),
                                 index.GetType()};
}

}  // namespace

TransposeFusion::TransposeFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      tiling_scheme_(ComputeTransposeTilingScheme(analysis.tiled_transpose())) {
}

absl::Status TransposeFusion::EmitKernel(IrEmitterContext& ir_emitter_context,
                                         const HloFusionInstruction& fusion,
                                         const LaunchDimensions& launch_dims,
                                         std::vector<llvm_ir::IrArray> inputs,
                                         std::vector<llvm_ir::IrArray> outputs,
                                         llvm::IRBuilder<>* builder) const {
  const auto& hlo_roots = analysis_.fusion_roots();
  GpuElementalIrEmitter elemental_emitter(ir_emitter_context, builder);
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (auto [i, input] : llvm::enumerate(inputs)) {
    HloInstruction* fused_operand = fusion.fused_parameter(i);
    fused_emitter.BindGenerator(
        *fused_operand, [input = input, builder,
                         fused_operand](const llvm_ir::IrArray::Index& index) {
          return input.EmitReadArrayElement(index, builder,
                                            fused_operand->name());
        });
  }

  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, const HloInstruction*>>>
      transposes_to_roots;
  // Keep a list of deduplicated transpose heroes separate from
  // transposes_to_roots to make the CodeGen deterministic.
  std::vector<TransposeDescription> transposes;
  transposes.reserve(hlo_roots.size());
  std::vector<std::pair<int64_t, const HloInstruction*>> extra_outputs;

  for (const auto& [output_idx, root] : llvm::enumerate(hlo_roots)) {
    const auto& hero = FindNonTrivialHero(*root);
    auto transpose_descr = GetDescriptionForTiledTransposeEmitter(*root, hero);
    if (transpose_descr.has_value()) {
      auto iterator_inserted = transposes_to_roots.insert(std::make_pair(
          &hero, std::vector<std::pair<int64_t, const HloInstruction*>>{
                     {output_idx, root}}));
      if (iterator_inserted.second) {
        transposes.push_back(*transpose_descr);
      } else {
        iterator_inserted.first->second.push_back({output_idx, root});
      }
    } else {
      extra_outputs.push_back({output_idx, root});
    }
  }

  absl::flat_hash_map<const HloInstruction*, llvm_ir::SharedMemoryTile> tiles;
  Vector3 permutation;
  for (const auto& [tile_idx, tr] : llvm::enumerate(transposes)) {
    permutation = tr.permutation;
    auto tile_size = tiling_scheme_.GetBlockTileSize();
    ++tile_size.back();  // Prevent bank conflicts.
    auto* module = ir_emitter_context.llvm_module();
    tiles[tr.instr] = llvm_ir::AllocateSharedMemoryTile(
        module,
        llvm_ir::PrimitiveTypeToIrType(tr.instr->shape().element_type(),
                                       module),
        tile_size, absl::StrCat("tr_tile_", tile_idx));
  }

  auto tile_to_inout = TileToInoutPermutation(permutation);
  auto input_shape = Permute(tiling_scheme_.GetShape(), tile_to_inout);
  auto tile_generator = [&](const TilingThreadIdInfo& thread_id_info,
                            const llvm_ir::IrArray::Index& tile_start_index,
                            std::array<llvm::Value*, 3> tile_dimensions) {
    // Copy input parameter values to shared memory buffers:
    // tile[thread_id_y, thread_id_x] = input[index]
    EmitTile(
        builder, tiling_scheme_, thread_id_info, tile_dimensions,
        [&](std::array<llvm::Value*, 3> index_in_tile) {
          auto index =
              PermuteIndex(tile_start_index.AddOffset(index_in_tile, builder),
                           tile_to_inout);
          for (const auto& tr : transposes) {
            auto input_gen = *fused_emitter.GetGenerator(*tr.instr->operand(0));
            auto input_index = GetUnnormalizedIndex(
                index, tr.instr->operand(0)->shape(), builder, input_shape);
            llvm::Value* value = *input_gen(input_index);
            tiles[tr.instr].Store(value, index_in_tile, builder);
          }

          // Compute all extra output values before writing them. This
          // avoids overwriting aliased input/output values before all reads
          // occurred.
          std::vector<std::tuple<llvm_ir::IrArray, llvm_ir::IrArray::Index,
                                 llvm::Value*>>
              scheduled_writes;
          for (const auto& [output_idx, root] : extra_outputs) {
            llvm_ir::IrArray::Index extra_output_index = GetUnnormalizedIndex(
                index, root->shape(), builder, input_shape);
            auto output_gen = *fused_emitter.GetGenerator(*root);
            llvm::Value* output_value = *output_gen(extra_output_index);
            scheduled_writes.emplace_back(outputs[output_idx],
                                          extra_output_index, output_value);
          }

          for (const auto& [output, idx, value] : scheduled_writes) {
            output.EmitWriteArrayElement(idx, value, builder);
          }
        });

    EmitSyncThreads(builder, ir_emitter_context);

    auto output_tile_index = PermuteIndex(tile_start_index, {0, 2, 1});
    auto transposed_tile_dimensions = Permute(tile_dimensions, {0, 2, 1});

    EmitTile(
        builder, tiling_scheme_, thread_id_info, transposed_tile_dimensions,
        /*emit_elem_function=*/
        [&](std::array<llvm::Value*, 3> index_in_tile) {
          auto index =
              PermuteIndex(output_tile_index.AddOffset(index_in_tile, builder),
                           tile_to_inout);
          for (const auto& tr : transposes) {
            llvm::Value* loaded = tiles[tr.instr].Load(
                Permute(index_in_tile, {0, 2, 1}), builder);

            FusedIrEmitter fused_emitter(elemental_emitter);
            fused_emitter.BindGenerator(
                *tr.instr,
                [&](const llvm_ir::IrArray::Index&) { return loaded; });
            for (int64_t i = 0;
                 i < fusion.fused_instructions_computation()->num_parameters();
                 ++i) {
              llvm_ir::IrArray ir_array = inputs[i];
              HloInstruction* fused_operand = fusion.fused_parameter(i);
              fused_emitter.BindGenerator(
                  *fused_operand, [=](const llvm_ir::IrArray::Index& index) {
                    return ir_array.EmitReadArrayElement(index, builder,
                                                         fused_operand->name());
                  });
            }

            // Apply code generation for the code after the real hero.
            // Compute all output values before writing them. This avoids
            // overwriting aliased input/output values before all reads
            // occurred.
            std::vector<std::tuple<llvm_ir::IrArray, llvm_ir::IrArray::Index,
                                   llvm::Value*>>
                scheduled_writes;
            for (const auto& [output_idx, root] :
                 transposes_to_roots[tr.instr]) {
              TF_ASSIGN_OR_RETURN(llvm_ir::ElementGenerator gen,
                                  fused_emitter.GetGenerator(*root));

              // Both for emission and writing it should be
              // index-as-transformed by the computation.
              llvm_ir::IrArray::Index untiled_index =
                  GetUnnormalizedIndex(index, root->shape(), builder,
                                       Permute(input_shape, permutation));
              TF_ASSIGN_OR_RETURN(llvm::Value * generated, gen(untiled_index));
              scheduled_writes.emplace_back(outputs[output_idx], untiled_index,
                                            generated);
            }
            for (const auto& [output, idx, value] : scheduled_writes) {
              output.EmitWriteArrayElement(idx, value, builder);
            }
          }
          return absl::OkStatus();
        });
  };

  llvm::Type* index_type =
      GetIndexTypeForKernel(&fusion, launch_dims.launch_bound(), builder);
  return EmitTilingKernel(builder, tiling_scheme_, index_type, tile_generator)
      .status();
}

LaunchDimensions TransposeFusion::launch_dimensions() const {
  return LaunchDimensions(tiling_scheme_.GetNumBlocksPhysical(),
                          tiling_scheme_.GetNumThreadsPerBlockPhysical());
}

}  // namespace gpu
}  // namespace xla
