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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
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
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

#include "xla/backends/cpu/codegen/target_machine_features.h"

namespace ge = ::xla::gpu::experimental;

namespace {

template <typename TiledInstructionT>
int64_t PerTileCacheLines(
    const TiledInstructionT& inst,
    int64_t cache_line_bytes = TargetMachineFeatures::kDefaultCacheLineBytes) {
  const Shape& shape = inst.hlo()->shape();
  if (ShapeUtil::IsEffectiveScalar(shape)) {
    return 1;
  }

  int64_t minor_dim_idx = LayoutUtil::Minor(shape.layout(), 0);
  // The tiled emitter pads all tile dimensions to the next power of 2, we
  // therefore must take that into account.
  int64_t tile_minor_size = llvm::PowerOf2Ceil(inst.tile_size(minor_dim_idx));
  int64_t element_bytes =
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t tile_minor_bytes = tile_minor_size * element_bytes;

  int64_t non_minor_size = 1;
  for (auto [dim_idx, size] : llvm::enumerate(inst.tile_sizes())) {
    if (dim_idx != minor_dim_idx) {
      // See above comment
      non_minor_size *= llvm::PowerOf2Ceil(size);
    }
  }
  return non_minor_size * CeilOfRatio(tile_minor_bytes, cache_line_bytes);
}

// Super simple cost model that calculates the total number of cache line hits
// for the tiled computation. This doesn't take into account cache re-use
// between tiles or computation overheads, but it is a quick and easy heuristic
// that seems to give ok results.
// TODO(willfroom): Implement a cost model similar to
// GpuPerformanceModelWithIndexingAnalysis.
template <typename TiledComputationT>
int64_t TotalCacheLineHits(
    const TiledComputationT& tiling,
    const absl::flat_hash_set<const HloInstruction*>& operands,
    int64_t cache_line_bytes = TargetMachineFeatures::kDefaultCacheLineBytes) {
  int64_t per_tile_cost = 0;
  for (const auto* root : tiling.roots()) {
    per_tile_cost += PerTileCacheLines(*root, cache_line_bytes);
  }

  for (const auto* inst : tiling.instructions()) {
    // The tiling computation doesn't explicitly contain the parameter
    // instructions so we instead just check which instructions are operands to
    // the fusion.
    if (operands.contains(inst->hlo())) {
      per_tile_cost += PerTileCacheLines(*inst, cache_line_bytes);
    }
  }

  return tiling.num_output_tiles() * per_tile_cost;
}

int64_t EvaluateSymbolicCost(
    const ge::TiledHloComputation& symbolic_computation,
    llvm::ArrayRef<int64_t> candidate_tile_sizes,
    const absl::flat_hash_set<const HloInstruction*>& operands,
    int64_t cache_line_bytes = TargetMachineFeatures::kDefaultCacheLineBytes,
    int64_t max_stack_alloc_bytes =
        TargetMachineFeatures::kDefaultMaxStackAllocBytes) {
  const ge::TilingSpace& space = symbolic_computation.tiling_space();
  int64_t num_dims = space.num_dimensions();

  // Prepare variable values for SymbolicExpr::Evaluate.
  // The first `num_dims` are dimensions (we can default to 0 as they are not
  // used in tile size expressions).
  // The next `num_dims` are symbols representing the tile sizes.
  std::vector<int64_t> var_values(2 * num_dims, 0);
  for (int i = 0; i < num_dims; ++i) {
    int64_t tile_size = candidate_tile_sizes[i];
    int64_t dim_size = space.dimensions()[i].dimension_size;
    if (dim_size > 0) {
      tile_size = llvm::PowerOf2Ceil(std::min(tile_size, dim_size));
    }
    var_values[num_dims + i] = tile_size;
  }

  int64_t per_tile_cost = 0;

  struct TileMetrics {
    int64_t cache_lines_touched = 0;
    int64_t total_tile_bytes = 0;
  };

  auto compute_tile_metrics =
      [&](const ge::TiledHloInstruction& inst) -> TileMetrics {
    const Shape& shape = inst.hlo()->shape();
    int64_t element_bytes =
        ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    if (ShapeUtil::IsEffectiveScalar(shape)) {
      return TileMetrics{/*cache_lines_touched=*/1, element_bytes};
    }

    const ge::Tile& tile = inst.tile();
    int64_t minor_dim_idx = LayoutUtil::Minor(shape.layout(), 0);

    int64_t tile_minor_size = 1;
    int64_t tile_minor_stride = 1;
    int64_t non_minor_size = 1;
    int64_t tile_rank = tile.dim_tiles().size();
    int64_t total_tile_elements = 1;

    for (int i = 0; i < tile_rank; ++i) {
      int64_t size = tile.dim_tiles()[i].size.Evaluate(var_values);
      int64_t stride = tile.dim_tiles()[i].stride.Evaluate(var_values);
      int64_t padded_size = llvm::PowerOf2Ceil(size);
      total_tile_elements *= padded_size;

      if (i == minor_dim_idx) {
        tile_minor_size = padded_size;
        tile_minor_stride = stride;
      } else {
        non_minor_size *= padded_size;
      }
    }

    int64_t minor_cache_lines = CeilOfRatio(
        tile_minor_size * tile_minor_stride * element_bytes, cache_line_bytes);
    minor_cache_lines = std::max(int64_t{1}, minor_cache_lines);
    minor_cache_lines = std::min(tile_minor_size, minor_cache_lines);

    int64_t cache_lines_touched = non_minor_size * minor_cache_lines;
    int64_t total_tile_bytes = total_tile_elements * element_bytes;

    return TileMetrics{cache_lines_touched, total_tile_bytes};
  };

  absl::flat_hash_set<const HloInstruction*> root_hlos;
  for (const auto* root : symbolic_computation.roots()) {
    if (root->hlo()) {
      root_hlos.insert(root->hlo());
    }
    TileMetrics metrics = compute_tile_metrics(*root);
    per_tile_cost += metrics.cache_lines_touched;
  }

  for (const auto* inst : symbolic_computation.instructions()) {
    if (operands.contains(inst->hlo())) {
      TileMetrics metrics = compute_tile_metrics(*inst);
      per_tile_cost += metrics.cache_lines_touched;
    }
  }

  // Incorporate producer-consumer edge materialization decision costs
  // for intermediate fused operations (evaluating recomputation cost vs
  // stack memref.alloca cost vs scratch buffer memref.alloc cost).
  for (const auto* inst : symbolic_computation.instructions()) {
    const HloInstruction* hlo = inst->hlo();
    if (hlo == nullptr || operands.contains(hlo) || root_hlos.contains(hlo)) {
      continue;
    }

    TileMetrics metrics = compute_tile_metrics(*inst);
    int64_t tile_bytes = metrics.total_tile_bytes;
    int64_t cache_lines_touched = metrics.cache_lines_touched;
    int64_t user_count = hlo->user_count();

    if (tile_bytes <= max_stack_alloc_bytes) {
      // Buffer fits in max stack allocation budget (memref.alloca).
      int64_t stack_alloca_cost = cache_lines_touched;
      int64_t recomputation_cost = user_count * cache_lines_touched;
      per_tile_cost += std::min(stack_alloca_cost, recomputation_cost);
    } else {
      // Exceeds stack allocation limit -> requires dynamic scratch buffer
      // (memref.alloc).
      constexpr int64_t kScratchAllocPenaltyFactor = 100;
      int64_t scratch_alloc_cost =
          kScratchAllocPenaltyFactor * cache_lines_touched;
      int64_t recomputation_cost = user_count * cache_lines_touched;
      per_tile_cost += std::min(scratch_alloc_cost, recomputation_cost);
    }
  }

  int64_t num_output_tiles = 1;
  for (auto [index, dim] : llvm::enumerate(space.dimensions())) {
    if (dim.type == ge::TilingSpace::DimensionSemantics::kParallel) {
      int64_t val = candidate_tile_sizes[index];
      num_output_tiles *= CeilOfRatio(dim.dimension_size, val);
    }
  }

  return per_tile_cost * num_output_tiles;
}

absl::StatusOr<Tiling> GetTiling(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const TargetMachineFeatures* target_machine_features = nullptr) {
  int64_t cache_line_bytes =
      target_machine_features ? target_machine_features->cache_line_bytes()
                              : TargetMachineFeatures::kDefaultCacheLineBytes;
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
  const HloInstruction* best_root = nullptr;
  for (const auto& tiling : valid_tilings) {
    const HloInstruction* cur_root = root_hlo;
    if (!tiling.tile_sizes().contains(cur_root)) {
      for (const auto& [inst, sizes] : tiling.tile_sizes()) {
        if (inst != nullptr && inst->shape().IsArray()) {
          cur_root = inst;
          break;
        }
      }
    }
    if (!tiling.tile_sizes().contains(cur_root)) {
      continue;
    }
    const FlatTiling& tile_sizes = tiling.tile_sizes().at(cur_root);
    ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                     symbolic_tile_analysis.ComputeTiledComputation(tiling));
    const int64_t cost =
        TotalCacheLineHits(tiled_hlo_computation, operands, cache_line_bytes);

    if (cost < best_cost) {
      best_cost = cost;
      best_root = cur_root;
      best_tile_sizes.assign(tile_sizes.begin(), tile_sizes.end());
    }
  }

  const HloInstruction* actual_root = best_root ? best_root : root_hlo;
  if (actual_root != nullptr && actual_root->shape().IsArray()) {
    const Shape& shape = actual_root->shape();
    for (int i = 0; i < best_tile_sizes.size() && i < shape.dimensions_size();
         ++i) {
      int64_t dim_size = shape.dimensions(i);
      if (dim_size > 0) {
        best_tile_sizes[i] =
            llvm::PowerOf2Ceil(std::min(best_tile_sizes[i], dim_size));
      }
    }
  }

  std::vector<FlatTiling> result{best_tile_sizes};
  Tiling::TileMapping tile_mapping{{actual_root, best_tile_sizes}};
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
    case HloOpcode::kConvert: {
      PrimitiveType operand_type = inst.operand(0)->shape().element_type();
      PrimitiveType result_type = inst.shape().element_type();
      // TODO(b/480995909): Remove this once JAX does not rely on convert from
      // PRED to U8 not clamping the value to the [0, 1] range. This lowering
      // would actually do (correct) clamping, but JAX has a test that
      // essentially checks that PRED storage is 8 bit, and it uses (broken)
      // Convert semantics instead of BitcastConvert, because BitcastConvert
      // with PRED types (assuming 8 bit storage for PRED) is not completely
      // supported on all backends yet.
      if (operand_type == PRED && result_type == U8) {
        return false;
      }
      return true;
    }
    case HloOpcode::kBitcast: {
      if (ShapeUtil::ElementsIn(inst.operand(0)->shape()) !=
          ShapeUtil::ElementsIn(inst.shape())) {
        return false;
      }
      PrimitiveType operand_type = inst.operand(0)->shape().element_type();
      PrimitiveType result_type = inst.shape().element_type();
      // TiledFusionEmitter uses i1 type for PRED, whereas the BitcastConvert
      // semantics for PRED types require 8 bit storage.
      if (result_type != operand_type &&
          (result_type == PRED || operand_type == PRED)) {
        return false;
      }
      return true;
    }
    case HloOpcode::kIota:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kParameter:
      return true;
    case HloOpcode::kGather: {
      const auto* gather = DynCast<const HloGatherInstruction>(&inst);
      if (gather == nullptr || !GatherSimplifier::IsSimplifiedGather(gather)) {
        return false;
      }
      const auto& start_index_map =
          gather->gather_dimension_numbers().start_index_map();
      for (int64_t i = 0; i < start_index_map.size(); ++i) {
        if (start_index_map[i] != i) {
          return false;
        }
      }
      return true;
    }
    case HloOpcode::kDot:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReverse:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
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
    case HloOpcode::kMulhi:
      return false;
      break;
    default:
      return inst.IsElementwise();
  }
}

absl::Status VerifyTensorRanks(const HloFusionInstruction& fusion) {
  constexpr int kMaxRank = 8;
  auto check_shape = [](const Shape& shape) -> absl::Status {
    if (shape.IsArray()) {
      if (shape.dimensions().size() > kMaxRank) {
        return Internal(
            "Unsupported fusion in EmitGeneric: tensor rank too large");
      }
    } else if (shape.IsTuple()) {
      for (const Shape& subshape : shape.tuple_shapes()) {
        if (subshape.IsArray() && subshape.dimensions().size() > kMaxRank) {
          return Internal(
              "Unsupported fusion in EmitGeneric: tensor rank too large");
        }
      }
    }
    return absl::OkStatus();
  };

  for (const xla::HloInstruction* instruction : fusion.fused_instructions()) {
    RETURN_IF_ERROR(check_shape(instruction->shape()));
    for (const xla::HloInstruction* operand : instruction->operands()) {
      RETURN_IF_ERROR(check_shape(operand->shape()));
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
  if (!IsSupportedShape(fusion.shape())) {
    return Internal("Fusion shape is not supported by the tiled CPU emitter.");
  }

  for (const HloInstruction* operand : fusion.operands()) {
    if (!IsSupportedShape(operand->shape())) {
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
  KernelSpec kernel_spec(name, work_dimensions, KernelSpec::Buffers(),
                         KernelSpec::Buffers(), absl::flat_hash_set<int64_t>());
  auto kernel_spec_or =
      emitters::GetKernelSpec(name, fusion, buffer_assignment, work_dimensions);
  if (kernel_spec_or.ok()) {
    kernel_spec = std::move(*kernel_spec_or);
  }
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

  const HloInstruction* root =
      symbolic_tile_analysis.GetRoot(symbolic_tile_analysis.real_root_index());
  if (!tiling.tile_sizes().contains(root)) {
    for (const auto& [inst, sizes] : tiling.tile_sizes()) {
      if (inst != nullptr && inst->shape().IsArray()) {
        root = inst;
        break;
      }
    }
  }
  int64_t num_tiles = 1;
  if (root != nullptr && tiling.tile_sizes().contains(root) &&
      root->shape().IsArray()) {
    for (auto [dim, tile_size] :
         llvm::zip(root->shape().dimensions(), tiling.tile_sizes().at(root))) {
      num_tiles *= CeilOfRatio(dim, tile_size);
    }
  }

  return CreateTiledKernelDefinition(context, fusion, buffer_assignment, name,
                                     num_work_groups, num_tiles,
                                     std::move(module));
}

absl::StatusOr<ge::TiledHloComputation> GetTiledHloComputation(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const TargetMachineFeatures* target_machine_features = nullptr) {
  RETURN_IF_ERROR(VerifyTensorRanks(fusion));

  int64_t cache_line_bytes =
      target_machine_features ? target_machine_features->cache_line_bytes()
                              : TargetMachineFeatures::kDefaultCacheLineBytes;
  int64_t max_stack_alloc_bytes =
      target_machine_features
          ? target_machine_features->max_stack_alloc_bytes()
          : TargetMachineFeatures::kDefaultMaxStackAllocBytes;

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&fusion);
  ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> tiling_space,
                   ge::TilingSpace::Create(*fusion_adaptor, &context));
  using ValidTilings = std::vector<llvm::SmallVector<int64_t, 4>>;
  ASSIGN_OR_RETURN(ValidTilings candidates, tiling_space->GetValidTilings());

  // 1. Construct the Symbolic Graph EXACTLY ONCE on the stack/heap.
  ASSIGN_OR_RETURN(
      ge::TiledHloComputation symbolic_computation,
      ge::TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)));

  absl::flat_hash_set<const HloInstruction*> operands(fusion.operands().begin(),
                                                      fusion.operands().end());

  // 2. Evaluate all candidates by substituting concrete tile sizes into the
  // symbolic tiles of roots and operands.
  struct Candidate {
    llvm::SmallVector<int64_t, 4> padded_tile_sizes;
    int64_t cost;
  };
  std::vector<Candidate> evaluated_candidates;
  evaluated_candidates.reserve(candidates.size());
  for (const auto& tile_sizes : candidates) {
    auto padded_tile_sizes = xla::xtile::GetPaddedTileSizes(tile_sizes);
    for (int i = 0; i < padded_tile_sizes.size() &&
                    i < symbolic_computation.tiling_space().num_dimensions();
         ++i) {
      int64_t dim_size =
          symbolic_computation.tiling_space().dimensions()[i].dimension_size;
      if (dim_size > 0) {
        padded_tile_sizes[i] =
            llvm::PowerOf2Ceil(std::min(padded_tile_sizes[i], dim_size));
      }
    }
    int64_t cost =
        EvaluateSymbolicCost(symbolic_computation, padded_tile_sizes, operands,
                             cache_line_bytes, max_stack_alloc_bytes);
    VLOG(2) << "Candidate: {" << absl::StrJoin(tile_sizes, ", ")
            << "} (padded: {" << absl::StrJoin(padded_tile_sizes, ", ")
            << "}) cost: " << cost;
    evaluated_candidates.push_back({std::move(padded_tile_sizes), cost});
  }
  std::sort(evaluated_candidates.begin(), evaluated_candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.cost != b.cost) {
                return a.cost < b.cost;
              }
              return a.padded_tile_sizes < b.padded_tile_sizes;
            });

  // 3. Try to tile candidates in order of increasing cost.
  for (int i = 0; i < evaluated_candidates.size(); ++i) {
    const auto& candidate = evaluated_candidates[i];
    VLOG(2) << "Trying candidate " << i << ": {"
            << absl::StrJoin(candidate.padded_tile_sizes, ", ")
            << "} cost: " << candidate.cost;
    ASSIGN_OR_RETURN(std::unique_ptr<ge::TilingSpace> winning_tiling_space,
                     ge::TilingSpace::Create(*fusion_adaptor, &context));
    if (const absl::Status status =
            winning_tiling_space->AssignTileSizes(candidate.padded_tile_sizes);
        !status.ok()) {
      VLOG(2) << "  AssignTileSizes failed: " << status;
      continue;
    }
    auto tiled_computation = ge::TiledHloComputation::Tile(
        *fusion_adaptor, std::move(winning_tiling_space));
    if (tiled_computation.ok()) {
      tiled_computation->Simplify();
      tiled_computation->SortInstructionsPostOrder();
      VLOG(2) << "  Tiling succeeded! Winner picked.";
      return std::move(*tiled_computation);
    }
    VLOG(2) << "  Tiling failed: " << tiled_computation.status();
  }

  return absl::NotFoundError(absl::StrCat(
      "No valid tiled search candidates found for: ", fusion.name()));
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitTiledFusionKernelImpl(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups, const ge::TiledHloComputation& tiled_computation) {
  ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      xtile::EmitXTileModule(name, fusion, tiled_computation, context));

  const ge::TiledHloInstruction* root = tiled_computation.roots().front();
  for (const ge::TiledHloInstruction* r : tiled_computation.roots()) {
    if (r->hlo() != nullptr && r->hlo()->shape().IsArray()) {
      root = r;
      break;
    }
  }
  const HloInstruction* root_hlo = root->hlo();
  int64_t num_tiles = 1;
  if (root_hlo != nullptr && root_hlo->shape().IsArray()) {
    for (auto [dim, tile_size] :
         llvm::zip(root_hlo->shape().dimensions(), root->tile_sizes())) {
      num_tiles *= CeilOfRatio(dim, tile_size);
    }
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
    int64_t num_work_groups,
    const TargetMachineFeatures* target_machine_features) {
  VLOG(2) << "EmitTiledFusionKernel called for fusion: " << fusion.name();
  auto supported_status = IsSupportedTiledFusion(fusion);
  VLOG(2) << "  IsSupportedTiledFusion: " << supported_status;
  if (!supported_status.ok()) {
    return {absl::UnimplementedError(
                "Fusion is not supported by the tiled CPU emitter."),
            /*tiling_succeeded=*/false};
  }

  if (options::EnableExperimentalTiling(fusion.GetModule()->config())) {
    VLOG(2) << "  EnableExperimentalTiling: true";
    absl::StatusOr<ge::TiledHloComputation> tiled_computation =
        GetTiledHloComputation(context, fusion, target_machine_features);
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

  absl::StatusOr<Tiling> tiling = GetTiling(
      context, fusion, *symbolic_tile_analysis, target_machine_features);
  if (!tiling.ok()) {
    return {tiling.status(), /*tiling_succeeded=*/false};
  }

  return {EmitTiledFusionKernelImpl(context, fusion, buffer_assignment, name,
                                    num_work_groups, *symbolic_tile_analysis,
                                    *tiling),
          /*tiling_succeeded=*/true};
}

}  // namespace xla::cpu
