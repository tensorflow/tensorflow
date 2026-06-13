/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace ge = ::xla::gpu::experimental;

namespace {

int64_t PerTileCacheLines(const TiledHloInstruction& inst) {
  const Shape& shape = inst.hlo()->shape();
  if (ShapeUtil::IsEffectiveScalar(shape)) {
    return 1;
  }

  int64_t minor_dim_idx = LayoutUtil::Minor(shape.layout(), 0);
  // The tiled emitter pads all tile dimensions to the next power of 2, we
  // therefore must take that into account.
  int64_t tile_minor_size = llvm::PowerOf2Ceil(inst.tile_size(minor_dim_idx));
  constexpr int64_t kCacheLineSize = 64;
  int64_t element_bytes =
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t tile_minor_bytes = tile_minor_size * element_bytes;

  int64_t non_min_size = 1;
  for (auto [dim_idx, size] : llvm::enumerate(inst.tile_sizes())) {
    if (dim_idx != minor_dim_idx) {
      // See above comment
      non_min_size *= llvm::PowerOf2Ceil(size);
    }
  }
  return non_min_size * CeilOfRatio(tile_minor_bytes, kCacheLineSize);
}

// Super simple cost model that calculates the total number of cache line hits
// for the tiled computation. This doesn't take into account cache re-use
// between tiles or computation overheads, but it is a quick and easy heuristic
// that seems to give ok results.
// TODO(willfroom): Implement a cost model similar to
// GpuPerformanceModelWithIndexingAnalysis.
int64_t TotalCacheLineHits(
    const TiledHloComputation& tiling,
    const absl::flat_hash_set<const HloInstruction*>& operands) {
  int64_t per_tile_cost = 0;
  for (const auto* root : tiling.roots()) {
    per_tile_cost += PerTileCacheLines(*root);
  }

  for (const auto* inst : tiling.instructions()) {
    // The tiling computation doesn't explicitly contain the parameter
    // instructions so we instead just check which instructions are operands to
    // the fusion.
    if (operands.contains(inst->hlo())) {
      per_tile_cost += PerTileCacheLines(*inst);
    }
  }

  return tiling.num_output_tiles() * per_tile_cost;
}

absl::StatusOr<Tiling> GetTiling(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis) {
  ASSIGN_OR_RETURN(std::vector<Tiling> valid_tilings,
                   symbolic_tile_analysis.GetValidTilings());
  if (valid_tilings.empty()) {
    return Internal("No valid tilings found for fusion: %s", fusion.name());
  }

  const HloInstruction* root_hlo =
      fusion.fused_instructions_computation()->root_instruction();
  std::vector<int64_t> filtered_tilings;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  FlatTiling best_tile_sizes;
  absl::flat_hash_set<const HloInstruction*> operands(fusion.operands().begin(),
                                                      fusion.operands().end());
  for (const auto& tiling : valid_tilings) {
    const FlatTiling& tile_sizes = tiling.tile_sizes().at(root_hlo);
    ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                     symbolic_tile_analysis.ComputeTiledComputation(tiling));
    const int64_t cost = TotalCacheLineHits(tiled_hlo_computation, operands);

    if (cost < best_cost) {
      best_cost = cost;
      best_tile_sizes.assign(tile_sizes.begin(), tile_sizes.end());
    }
  }

  std::vector<FlatTiling> result{best_tile_sizes};
  Tiling::TileMapping tile_mapping{{root_hlo, best_tile_sizes}};
  return Tiling(tile_mapping);
}

bool IsSupportedShape(const Shape& shape) {
  bool is_supported = true;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray()) {
          if (!IsSupportedTilingType(subshape.element_type())) {
            is_supported = false;
          }
        }
      });

  return is_supported;
}

bool IsSupportedInstruction(const HloInstruction& inst) {
  HloOpcode opcode = inst.opcode();
  switch (opcode) {
    case HloOpcode::kBitcast:
    case HloOpcode::kIota:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kParameter:
      return true;
    case HloOpcode::kConstant:
      return ShapeUtil::IsEffectiveScalar(inst.shape());
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kMap:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kImag:
    case HloOpcode::kSign:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kClz:
      return false;
      break;
    default:
      return inst.IsElementwise();
  }
}

absl::Status VerifyTensorRanks(const HloFusionInstruction& fusion) {
  constexpr int kMaxRank = 8;
  for (const xla::HloInstruction* instruction : fusion.fused_instructions()) {
    if (instruction->shape().dimensions().size() > kMaxRank) {
      return Internal(
          "Unsupported fusion in EmitGeneric: tensor rank too large");
    }

    for (const xla::HloInstruction* operand : instruction->operands()) {
      if (operand->shape().dimensions().size() > kMaxRank) {
        return Internal(
            "Unsupported fusion in EmitGeneric: tensor rank too large");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<SymbolicTileAnalysis> GetSymbolicTileAnalysis(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion) {
  RETURN_IF_ERROR(VerifyTensorRanks(fusion));

  EmitterSpecificConstraintsBuilder constraints_builder =
      TiledEmitterConstraints::GetBuilder();
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *fusion.fused_instructions_computation(), &context,
          constraints_builder);
  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  return std::get<SymbolicTileAnalysis>(std::move(symbolic_tile_analysis_or));
}

absl::Status IsSupportedTiledFusion(const HloFusionInstruction& fusion) {
  // TODO(willfroom): Support multi-output fusions.
  if (!fusion.shape().IsArray()) {
    return Internal(
        "Multi-output fusions are not supported by the tiled CPU emitter.");
  }

  for (const HloInstruction* operand : fusion.operands()) {
    if (!operand->shape().IsArray()) {
      return Internal(
          "Non-array operands are not supported by the tiled CPU emitter.");
    }
  }

  for (const HloInstruction* inst : fusion.fused_instructions()) {
    if (!IsSupportedShape(inst->shape())) {
      return Internal(
          "Instruction %s has a type, which is not supported by the "
          "tiled CPU emitter.",
          inst->ToString());
    }

    if (!IsSupportedInstruction(*inst)) {
      return Internal(
          "Instruction %s is not supported by the tiled CPU emitter.",
          inst->ToString());
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> CreateTiledKernelDefinition(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups, int64_t num_tiles,
    mlir::OwningOpRef<mlir::ModuleOp> module) {
  module->setName(absl::StrCat("__compute_module", "_", name));

  int64_t tiles_per_workgroup =
      CeilOfRatio<int64_t>(num_tiles, num_work_groups);
  module->walk([&](xtile::EntryFuncOp op) {
    xtile::TilingInfoAttr info = xtile::TilingInfoAttr::get(
        op->getContext(), num_tiles, tiles_per_workgroup);
    op->setAttr("xtile.tiling_info", info);
  });

  module->getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      mlir::StringAttr::get(
          &context, BuildModuleMemoryRegionName("tiled_emitter", &fusion)));

  WorkDimensions work_dimensions;
  work_dimensions.num_work_groups.x = num_work_groups;
  ASSIGN_OR_RETURN(KernelSpec kernel_spec,
                   emitters::GetKernelSpec(name, fusion, buffer_assignment,
                                           work_dimensions));
  return KernelDefinition<MlirKernelSource>(
      std::move(kernel_spec), MlirKernelSource(std::move(module)));
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitTiledFusionKernelImpl(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups, const SymbolicTileAnalysis& symbolic_tile_analysis,
    const Tiling& tiling) {
  EmitterSpecificConstraintsBuilder constraints_builder =
      TiledEmitterConstraints::GetBuilder();
  ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                   xtile::EmitXTileModule(name, fusion, symbolic_tile_analysis,
                                          tiling, context));

  const HloInstruction* root = symbolic_tile_analysis.GetRoot(0);
  int64_t num_tiles = 1;
  for (auto [dim, tile_size] :
       llvm::zip(root->shape().dimensions(), tiling.tile_sizes().at(root))) {
    num_tiles *= CeilOfRatio(dim, tile_size);
  }

  return CreateTiledKernelDefinition(context, fusion, buffer_assignment, name,
                                     num_work_groups, num_tiles,
                                     std::move(module));
}

absl::StatusOr<ge::TiledHloComputation> GetTiledHloComputation(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion) {
  RETURN_IF_ERROR(VerifyTensorRanks(fusion));

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&fusion);
  ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> tiling_space,
                   ge::TilingSpace::Create(*fusion_adaptor, &context));
  llvm::SmallVector<ge::TilingSpace::DimensionInfo, 4> dims =
      tiling_space->dimensions();

  // TODO: b/511084185 - This is a temporary "Single Tile" dummy strategy (tile
  // size = PowerOf2Ceil(dimension size)) to verify the end-to-end MLIR pipeline
  // plumbing first.
  std::vector<int64_t> tile_sizes;
  if (!dims.empty()) {
    tile_sizes.reserve(dims.size());
  }

  for (const auto& dim : dims) {
    int64_t tile_size =
        dim.dimension_size == 0 ? 1 : llvm::PowerOf2Ceil(dim.dimension_size);
    tile_sizes.push_back(tile_size);
  }

  RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
  return ge::TiledHloComputation::Tile(*fusion_adaptor,
                                       std::move(tiling_space));
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitTiledFusionKernelImpl(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups, const ge::TiledHloComputation& tiled_computation) {
  ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      xtile::EmitXTileModule(name, fusion, tiled_computation, context));

  const ge::TiledHloInstruction* root = tiled_computation.roots().front();
  const HloInstruction* root_hlo = root->hlo();
  int64_t num_tiles = 1;
  for (auto [dim, tile_size] :
       llvm::zip(root_hlo->shape().dimensions(), root->tile_sizes())) {
    num_tiles *= CeilOfRatio(dim, tile_size);
  }

  return CreateTiledKernelDefinition(context, fusion, buffer_assignment, name,
                                     num_work_groups, num_tiles,
                                     std::move(module));
}

}  // namespace

// We don't currently support sub-byte types in the tiled CPU emitter.
bool IsSupportedTilingType(PrimitiveType type) {
  if (type == PRED) {
    return true;
  }

  if (primitive_util::BitWidth(type) < 8) {
    return false;
  }

  if (primitive_util::IsUnsignedIntegralType(type)) {
    return false;
  }

  if (primitive_util::IsComplexType(type)) {
    return false;
  }

  // Some f8 types are not supported by the emitter, just don't support any of
  // them for now.
  if (primitive_util::IsF8Type(type) || primitive_util::IsF6Type(type) ||
      primitive_util::IsMXType(type)) {
    return false;
  }

  return true;
}

TiledEmissionResult EmitTiledFusionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups) {
  if (!IsSupportedTiledFusion(fusion).ok()) {
    return {absl::UnimplementedError(
                "Fusion is not supported by the tiled CPU emitter."),
            /*tiling_succeeded=*/false};
  }

  if (options::EnableExperimentalTiling(fusion.GetModule()->config())) {
    absl::StatusOr<ge::TiledHloComputation> tiled_computation =
        GetTiledHloComputation(context, fusion);
    if (!tiled_computation.ok()) {
      return {tiled_computation.status(), /*tiling_succeeded=*/false};
    }

    return {EmitTiledFusionKernelImpl(context, fusion, buffer_assignment, name,
                                      num_work_groups, *tiled_computation),
            /*tiling_succeeded=*/true};
  }

  absl::StatusOr<SymbolicTileAnalysis> symbolic_tile_analysis =
      GetSymbolicTileAnalysis(context, fusion);
  if (!symbolic_tile_analysis.ok()) {
    return {symbolic_tile_analysis.status(), /*tiling_succeeded=*/false};
  }

  absl::StatusOr<Tiling> tiling =
      GetTiling(context, fusion, *symbolic_tile_analysis);
  if (!tiling.ok()) {
    return {tiling.status(), /*tiling_succeeded=*/false};
  }

  return {EmitTiledFusionKernelImpl(context, fusion, buffer_assignment, name,
                                    num_work_groups, *symbolic_tile_analysis,
                                    *tiling),
          /*tiling_succeeded=*/true};
}

}  // namespace xla::cpu
