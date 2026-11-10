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

#include "xla/backends/gpu/transforms/fusion_block_level_rewriter.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/autotuning/gpu_default_backend_policy.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<bool> ProcessFusionInstruction(
    HloFusionInstruction* fusion_instruction,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    mlir::MLIRContext* mlir_context,
    tsl::thread::ThreadPool* thread_pool = nullptr,
    MlirContextPool* mlir_context_pool = nullptr) {
  bool dump_fusion_visualization = fusion_instruction->GetModule()
                                       ->config()
                                       .debug_options()
                                       .xla_dump_fusion_visualization();

  ABSL_ASSIGN_OR_RETURN(auto backend_config,
                   fusion_instruction->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_backend_config =
      backend_config.fusion_backend_config();
  if (fusion_backend_config.kind() == kTensorIrFusionKind) {
    // The fusion was explicitly assigned to the TensorIR emitter (by the
    // autotuner or by hand). Rewriting it into a Triton block-level fusion
    // would silently discard that assignment.
    VLOG(2) << "Not rewriting fusion " << fusion_instruction->name()
            << " because it is assigned to the TensorIR emitter.";
    return false;
  }
  if (fusion_backend_config.kind() == kPhasedBlockClusterFusionKind ||
      (fusion_instruction->parent()->IsFusionComputation() &&
       IsGpuFusionKind(*fusion_instruction->parent()->FusionInstruction(),
                       kPhasedBlockClusterFusionKind))) {
    return false;
  }
  if (fusion_backend_config.has_block_level_fusion_config()) {
    // Fusion is already block-level! Skip.
    return false;
  }

  if (GetDefaultBackend(*fusion_instruction) !=
      autotuner::Backend::BLOCK_LEVEL_EMITTER) {
    VLOG(2) << "Not rewriting fusion " << fusion_instruction->ToString()
            << " because it is not supported.";
    return false;
  }

  const HloComputation* fusion_computation =
      fusion_instruction->fused_instructions_computation();
  if (CodegenDecision can_codegen = IsTritonSupportedComputation(
          *fusion_computation, device_info.gpu_compute_capability());
      !can_codegen) {
    VLOG(2) << "Can't rewrite fusion " << fusion_instruction->ToString()
            << " because one or more instructions is not supported by Triton: "
            << can_codegen.Explain();
    if (dump_fusion_visualization) {
      RegisterFusionState(
          *fusion_instruction->parent(),
          absl::StrCat("Can't rewrite |", fusion_instruction->name(),
                       "|: not supported by Triton: ", can_codegen.Explain()),
          *fusion_instruction);
    }
    return false;
  }

  HloFusionAnalysisCache fusion_analysis_cache(device_info);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info, &fusion_analysis_cache, shape_size, mlir_context,
      fusion_instruction->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion(),
      mlir_context_pool);

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      Cast<HloFusionInstruction>(fusion_instruction));

  ABSL_ASSIGN_OR_RETURN(TiledRunTimeDataOrError tiled_runtime_data_or_error,
                   indexing_performance_model
                       .TryFindBestTilingForFusionAsync(
                           *fusion_adaptor,
                           thread_pool ? thread_pool->AsExecutor() : nullptr)
                       .Await());

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&tiled_runtime_data_or_error)) {
    // Can't rewrite this fusion because we can't tile it, skip!
    VLOG(2) << "Can't rewrite fusion " << fusion_instruction->ToString()
            << " because tiling search failed: " << fusion_decision->Explain();
    if (dump_fusion_visualization) {
      RegisterFusionState(
          *fusion_instruction->parent(),
          absl::StrCat("Can't rewrite |", fusion_instruction->name(),
                       "|: tiling search failed: ", fusion_decision->Explain()),
          *fusion_instruction);
    }
    return false;
  }

  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or_error));
  VLOG(1)
      << "Found parameters "
      << absl::StrCat(
             "sizes=[",
             absl::StrJoin(
                 tiled_runtime_data.block_level_parameters.output_tile_sizes[0],
                 ", "),
             "], num_warps=",
             tiled_runtime_data.block_level_parameters.num_warps)
      << " for fusion computation " << fusion_computation->ToString();

  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  backend_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
  ABSL_RETURN_IF_ERROR(fusion_instruction->set_backend_config(backend_config));
  fusion_instruction->set_fusion_kind(HloInstruction::FusionKind::kCustom);

  if (dump_fusion_visualization) {
    RegisterFusionState(*fusion_instruction->parent(),
                        absl::StrCat("Rewrote |", fusion_instruction->name(),
                                     "| to Triton block-level fusion"),
                        *fusion_instruction);
  }
  return true;
}

}  // anonymous namespace

absl::StatusOr<bool> FusionBlockLevelRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));

  bool has_changed = false;

  for (HloComputation* computation :
       module->MakeComputationSorted(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    HloFusionInstruction* fusion_instruction =
        ::xla::Cast<HloFusionInstruction>(computation->FusionInstruction());
    ABSL_ASSIGN_OR_RETURN(bool changed,
                     ProcessFusionInstruction(
                         fusion_instruction, device_info_, shape_size_,
                         mlir_context_, thread_pool_, mlir_context_pool_));

    has_changed |= changed;
  }

  return has_changed;
}

}  // namespace gpu
}  // namespace xla
