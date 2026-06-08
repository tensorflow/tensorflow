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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
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

template <typename TiledInstructionT>
int64_t PerTileCacheLinesGeneric(const TiledInstructionT& inst) {
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
template <typename TiledComputationT>
int64_t TotalCacheLineHitsGeneric(
    const TiledComputationT& tiling,
    const absl::flat_hash_set<const HloInstruction*>& operands) {
  int64_t per_tile_cost = 0;
  for (const auto* root : tiling.roots()) {
    per_tile_cost += PerTileCacheLinesGeneric(*root);
  }

  for (const auto* inst : tiling.instructions()) {
    // The tiling computation doesn't explicitly contain the parameter
    // instructions so we instead just check which instructions are operands to
    // the fusion.
    if (operands.contains(inst->hlo())) {
      per_tile_cost += PerTileCacheLinesGeneric(*inst);
    }
  }

  return tiling.num_output_tiles() * per_tile_cost;
}

int64_t EvaluateSymbolicCost(
    const ge::TiledHloComputation& symbolic_computation,
    llvm::ArrayRef<int64_t> candidate_tile_sizes,
    const absl::flat_hash_set<const HloInstruction*>& operands) {
  const ge::TilingSpace& space = symbolic_computation.tiling_space();
  mlir::MLIRContext* mlir_context = space.mlir_context();
  int64_t num_dims = space.num_dimensions();

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replacement_map;
  for (auto [index, dim] : llvm::enumerate(space.dimensions())) {
    int64_t val =
        index < candidate_tile_sizes.size() ? candidate_tile_sizes[index] : 1;
    replacement_map[CreateSymbolExpr(dim.id.value(), num_dims, mlir_context)] =
        CreateSymbolicConstant(val, mlir_context);
  }

  int64_t per_tile_cost = 0;

  auto cost_inst = [&](const ge::TiledHloInstruction& inst) {
    const Shape& shape = inst.hlo()->shape();
    if (ShapeUtil::IsEffectiveScalar(shape)) {
      per_tile_cost += 1;
      return;
    }

    ge::Tile concrete_tile = inst.tile();
    concrete_tile.Replace(replacement_map);
    concrete_tile.Simplify();
    auto static_sizes = concrete_tile.GetStaticTileSizes();
    if (!static_sizes.ok()) {
      return;
    }

    int64_t minor_dim_idx = LayoutUtil::Minor(shape.layout(), 0);
    int64_t tile_minor_size =
        minor_dim_idx < static_sizes->size()
            ? llvm::PowerOf2Ceil((*static_sizes)[minor_dim_idx])
            : 1;

    int64_t non_min_size = 1;
    for (auto [dim_idx, size] : llvm::enumerate(*static_sizes)) {
      if (dim_idx != minor_dim_idx) {
        non_min_size *= llvm::PowerOf2Ceil(size);
      }
    }

    int64_t element_bytes =
        ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    per_tile_cost += non_min_size *
                     CeilOfRatio(tile_minor_size * element_bytes, int64_t{64});
  };

  for (const auto* root : symbolic_computation.roots()) {
    cost_inst(*root);
  }

  for (const auto* inst : symbolic_computation.instructions()) {
    if (operands.contains(inst->hlo())) {
      cost_inst(*inst);
    }
  }

  int64_t num_output_tiles = 1;
  for (auto [index, dim] : llvm::enumerate(space.dimensions())) {
    if (dim.type == ge::TilingSpace::DimensionSemantics::kParallel) {
      int64_t val =
          index < candidate_tile_sizes.size() ? candidate_tile_sizes[index] : 1;
      num_output_tiles *= CeilOfRatio(dim.dimension_size, val);
    }
  }

  return num_output_tiles * per_tile_cost;
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
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  FlatTiling best_tile_sizes;
  absl::flat_hash_set<const HloInstruction*> operands(fusion.operands().begin(),
                                                      fusion.operands().end());
  for (const auto& tiling : valid_tilings) {
    const FlatTiling& tile_sizes = tiling.tile_sizes().at(root_hlo);
    ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                     symbolic_tile_analysis.ComputeTiledComputation(tiling));
    const int64_t cost =
        TotalCacheLineHitsGeneric(tiled_hlo_computation, operands);

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

std::vector<std::vector<int64_t>> GetTilingCandidates(
    llvm::ArrayRef<ge::TilingSpace::DimensionInfo> dimensions) {
  std::vector<std::vector<int64_t>> result = {{}};

  // Compile-time guard. Above this many candidates we fall back to a single
  // heuristic tiling instead of exploding the Cartesian product on high-rank
  // fusions.
  // TODO(b/511084185): replace this flat cap with a per-dim cache-capacity
  // prune once TargetMachineFeatures is plumbed through (getCacheSize).
  constexpr int64_t kMaxTilingCandidates = 256;

  for (const auto& dim : dimensions) {
    int64_t dim_size = dim.dimension_size;

    std::vector<int64_t> cur_dim_candidates;
    if (dim_size <= 0) {
      cur_dim_candidates.push_back(1);
    } else {
      int64_t ceil_size = llvm::PowerOf2Ceil(dim_size);
      for (int64_t size = 1; size <= std::min(kMaxTilingCandidates, ceil_size);
           size *= 2) {
        cur_dim_candidates.push_back(size);
      }
    }

    // Generate the search space via progressive Cartesian product expansion
    // across all dimensions.
    std::vector<std::vector<int64_t>> next_result;
    next_result.reserve(result.size() * cur_dim_candidates.size());
    for (const auto& combination : result) {
      for (const auto& item : cur_dim_candidates) {
        auto next_combination = combination;
        next_combination.push_back(item);
        next_result.push_back(std::move(next_combination));
      }
    }
    result = std::move(next_result);
  }

  return result;
}

absl::StatusOr<ge::TiledHloComputation> GetTiledHloComputation(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion) {
  RETURN_IF_ERROR(VerifyTensorRanks(fusion));

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&fusion);
  // 1. Construct the Symbolic Graph EXACTLY ONCE on the stack/heap.
  ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> tiling_space,
                   ge::TilingSpace::Create(*fusion_adaptor, &context));
  std::vector<std::vector<int64_t>> candidates =
      GetTilingCandidates(tiling_space->dimensions());

  absl::StatusOr<ge::TiledHloComputation> symbolic_computation =
      ge::TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space));
  if (!symbolic_computation.ok()) {
    return symbolic_computation.status();
  }

  absl::flat_hash_set<const HloInstruction*> operands(fusion.operands().begin(),
                                                      fusion.operands().end());

  // 2. Evaluate all 81 candidates using Full Symbolic Graph Substitution on
  // that SINGLE graph.
  std::vector<int64_t> best_tile_sizes;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  for (const auto& tile_sizes : candidates) {
    int64_t cost =
        EvaluateSymbolicCost(*symbolic_computation, tile_sizes, operands);
    if (cost < best_cost) {
      best_cost = cost;
      best_tile_sizes = tile_sizes;
    }
  }

  if (best_tile_sizes.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No valid tiled search candidates found for: ", fusion.name()));
  }

  // 3. Re-instantiate exactly once with concrete winning sizes.
  ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> winning_tiling_space,
                   ge::TilingSpace::Create(*fusion_adaptor, &context));
  RETURN_IF_ERROR(winning_tiling_space->AssignTileSizes(best_tile_sizes));
  return ge::TiledHloComputation::Tile(*fusion_adaptor,
                                       std::move(winning_tiling_space));
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
