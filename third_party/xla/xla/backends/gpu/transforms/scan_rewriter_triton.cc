/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/scan_rewriter_triton.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

namespace {

bool IsEligibleScan(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kScan || instr->IsRoot() ||
      !instr->shape().IsTuple()) {
    return false;
  }
  const auto* scan = Cast<HloScanInstruction>(instr);
  if (scan->num_carries() != 1 || scan->operand_count() != 2) {
    return false;
  }

  int64_t num_outputs = scan->shape().tuple_shapes_size() - scan->num_carries();
  bool has_output_gte = false;
  for (const HloInstruction* user : scan->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    if (user->tuple_index() < num_outputs) {
      has_output_gte = true;
    } else if (user->user_count() > 0 || user->IsRoot()) {
      return false;
    }
  }
  return has_output_gte;
}

absl::StatusOr<bool> RewriteScanToTritonFusion(
    HloInstruction* scan,
    GpuPerformanceModelWithIndexingAnalysis& indexing_performance_model) {
  HloComputation* parent = scan->parent();
  HloModule* module = parent->parent();
  const Shape& output_shape = scan->shape().tuple_shapes(0);

  HloComputation::Builder builder("triton_scan_computation");
  std::vector<HloInstruction*> parameters;
  parameters.reserve(scan->operand_count());
  for (int i = 0; i < scan->operand_count(); ++i) {
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, scan->operand(i)->shape(), absl::StrCat("param_", i))));
  }

  HloInstruction* cloned_scan = builder.AddInstruction(
      scan->CloneWithNewOperands(scan->shape(), parameters));
  HloInstruction* gte_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(output_shape, cloned_scan, 0));
  HloComputation* fusion_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(gte_0),
                                                /*is_entry=*/false);

  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      output_shape, HloInstruction::FusionKind::kCustom, scan->operands(),
      fusion_computation));
  fusion->GetModule()->SetAndUniquifyInstrName(fusion, "triton_scan");

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(fusion);
  ABSL_ASSIGN_OR_RETURN(
      TiledRunTimeDataOrError tiled_runtime_data_or,
      indexing_performance_model.TryFindBestTilingForFusion(*fusion_adaptor));

  if (std::holds_alternative<FusionDecision>(tiled_runtime_data_or)) {
    ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(fusion));
    ABSL_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(fusion_computation));
    return false;
  }

  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or));
  *gpu_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  int64_t num_outputs = scan->shape().tuple_shapes_size() -
                        Cast<HloScanInstruction>(scan)->num_carries();
  for (HloInstruction* user : std::vector<HloInstruction*>(scan->users())) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    if (user->IsRoot()) {
      parent->set_root_instruction(fusion);
    }
    ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(user));
    ABSL_RETURN_IF_ERROR(user->DropAllControlDeps());
    if (user->tuple_index() < num_outputs) {
      ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(fusion));
    }
    ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(user));
  }

  ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(scan));
  ABSL_RETURN_IF_ERROR(scan->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(scan));
  return true;
}

}  // namespace

absl::StatusOr<bool> ScanRewriterTriton::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloInstruction*> scans_to_rewrite;

  HloFusionAnalysisCache fusion_analysis_cache(device_info_);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info_, &fusion_analysis_cache, shape_size_, mlir_context_,
      /*use_experimental_tiling=*/true,
      module->config()
          .debug_options()
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion());

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsEligibleScan(instr) &&
          IsTritonSupportedInstruction(*instr,
                                       device_info_.gpu_compute_capability())) {
        scans_to_rewrite.push_back(instr);
      }
    }
  }

  for (HloInstruction* scan : scans_to_rewrite) {
    ABSL_ASSIGN_OR_RETURN(bool rewritten, RewriteScanToTritonFusion(
                                         scan, indexing_performance_model));
    changed |= rewritten;
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
