/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace gpu {
namespace {

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
      ir_emitter_context.rocm_compute_capability().gcn_arch_name().substr(
          0, 6) == "gfx90a") {
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

Status TransposeFusion::EmitKernel(
    IrEmitterContext& ir_emitter_context, ElementalIrEmitter& elemental_emitter,
    const HloFusionInstruction& fusion, const LaunchDimensions& launch_dims,
    std::vector<llvm_ir::IrArray> inputs, std::vector<llvm_ir::IrArray> outputs,
    llvm::IRBuilder<>* builder, int kernel_index) const {
  const auto& tiling_scheme = *analysis_.GetTransposeTilingScheme();
  const auto& hlo_roots = analysis_.fusion_roots();
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

  std::vector<const HloInstruction*> heroes;
  std::vector<std::optional<TransposeDescription>> transposes;
  heroes.reserve(hlo_roots.size());
  for (const auto& root : hlo_roots) {
    heroes.push_back(&FindNonTrivialHero(*root));
    transposes.push_back(
        GetDescriptionForTiledTransposeEmitter(*root, *heroes.back()));
  }

  absl::flat_hash_map<const HloInstruction*, llvm::GlobalVariable*> tiles;
  Vector3 permutation;
  for (const auto& [tile_idx, root] : llvm::enumerate(hlo_roots)) {
    if (const auto& tr = transposes[tile_idx]) {
      const auto& hero = *heroes[tile_idx];
      permutation = tr->permutation;
      tiles[&hero] = AllocateShared(
          builder, tiling_scheme,
          llvm_ir::PrimitiveTypeToIrType(
              hero.operand(0)->shape().element_type(),
              ir_emitter_context.llvm_module()),
          {tiling_scheme.GetBlockTileSizeFor(permutation[TilingScheme::DimX]),
           tiling_scheme.GetBlockTileSizeFor(TilingScheme::DimX) + 1},
          absl::StrCat("tr_tile_", tile_idx));
    }
  }

  TileElementGenerator tile_generator =
      [&](const TilingThreadIdInfo& thread_id_info,
          const llvm_ir::IrArray::Index& index,
          std::array<llvm::Value*, 2> tile_dimensions) {
        // Copy input parameter values to shared memory buffers:
        // tile[thread_id_y, thread_id_x] = input[index]
        // Note that tile_width and tile_height are flipped here because we
        // are reading a transposed tile.
        EmitTile(
            builder, tiling_scheme, index, thread_id_info, tile_dimensions,
            [&](const TilingThreadIdInfo& thread_id_info,
                const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
                llvm::Value* x_loc) {
              // Compute all extra output values before writing them. This
              // avoids overwriting aliased input/output values before all reads
              // occurred.
              std::vector<std::tuple<llvm_ir::IrArray, llvm_ir::IrArray::Index,
                                     llvm::Value*>>
                  scheduled_writes;

              for (const auto& [output_idx, root] :
                   llvm::enumerate(hlo_roots)) {
                if (transposes[output_idx].has_value()) {
                  const HloInstruction& hero = *heroes[output_idx];
                  llvm_ir::ElementGenerator input_gen =
                      *fused_emitter.GetGenerator(*hero.operand(0));
                  llvm_ir::IrArray::Index untiled_index = GetUnnormalizedIndex(
                      index, hero.operand(0)->shape(), builder,
                      tiling_scheme.GetDimsInElems());
                  llvm::Value* value = *input_gen(untiled_index);
                  llvm::Value* addr = thread_id_info.GEPIntoSharedMemory(
                      builder, tiles[&hero], {y_loc, x_loc});

                  builder->CreateStore(value, addr);
                } else {
                  llvm_ir::IrArray::Index untiled_index =
                      GetUnnormalizedIndex(index, root->shape(), builder,
                                           tiling_scheme.GetDimsInElems());
                  llvm_ir::ElementGenerator output_gen =
                      *fused_emitter.GetGenerator(*root);
                  llvm::Value* output_value = *output_gen(untiled_index);
                  scheduled_writes.emplace_back(outputs[output_idx],
                                                untiled_index, output_value);
                }
              }

              for (const auto& [output, idx, value] : scheduled_writes) {
                output.EmitWriteArrayElement(idx, value, builder);
              }
            });

        EmitSyncThreads(builder, ir_emitter_context);

        llvm_ir::IrArray::Index output_tile_index =
            PermuteIndex(index, permutation);
        std::array<llvm::Value*, 2> transposed_tile_dimensions = {
            tile_dimensions[1], tile_dimensions[0]};

        EmitTile(
            builder, tiling_scheme, output_tile_index, thread_id_info,
            transposed_tile_dimensions,
            /*emit_elem_function=*/
            [&](const TilingThreadIdInfo& thread_id_info,
                const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
                llvm::Value* x_loc) {
              for (const auto& [output_idx, root] :
                   llvm::enumerate(hlo_roots)) {
                if (transposes[output_idx].has_value()) {
                  const HloInstruction& hero = *heroes[output_idx];

                  std::vector<llvm::Value*> idx = {x_loc, y_loc};
                  llvm::Value* gep = thread_id_info.GEPIntoSharedMemory(
                      builder, tiles[&hero], idx);
                  llvm::Type* type =
                      thread_id_info.GEPIntoSharedMemoryType(tiles[&hero], idx);
                  llvm::Value* loaded =
                      builder->CreateLoad(type, gep, "tiled_buffer");

                  FusedIrEmitter fused_emitter(elemental_emitter);
                  fused_emitter.BindGenerator(
                      hero, [&](const llvm_ir::IrArray::Index& index) {
                        return loaded;
                      });
                  for (int64_t i = 0;
                       i < fusion.fused_instructions_computation()
                               ->num_parameters();
                       ++i) {
                    llvm_ir::IrArray ir_array = inputs[i];
                    HloInstruction* fused_operand = fusion.fused_parameter(i);
                    fused_emitter.BindGenerator(
                        *fused_operand,
                        [=](const llvm_ir::IrArray::Index& index) {
                          return ir_array.EmitReadArrayElement(
                              index, builder, fused_operand->name());
                        });
                  }

                  // Apply codegeneration for the code after the real hero.
                  TF_ASSIGN_OR_RETURN(llvm_ir::ElementGenerator gen,
                                      fused_emitter.GetGenerator(*root));

                  // Both for emission and writing it should be
                  // index-as-transformed by the computation.
                  llvm_ir::IrArray::Index untiled_index = GetUnnormalizedIndex(
                      index, root->shape(), builder,
                      Permute(tiling_scheme.GetDimsInElems(), permutation));
                  TF_ASSIGN_OR_RETURN(llvm::Value * generated,
                                      gen(untiled_index));
                  outputs[output_idx].EmitWriteArrayElement(untiled_index,
                                                            generated, builder);
                }
              }
              return OkStatus();
            });
      };

  llvm::Type* index_type =
      GetIndexTypeForKernel(&fusion, launch_dims.launch_bound(), builder);
  return EmitTilingKernel(builder, tiling_scheme, index_type, tile_generator)
      .status();
}

}  // namespace gpu
}  // namespace xla
