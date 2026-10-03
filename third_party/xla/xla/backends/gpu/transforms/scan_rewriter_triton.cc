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

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
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
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

namespace {

// Complements `IsTritonSupportedInstruction` with graph-level checks (non-root,
// only GTE 0 users).
bool IsEligibleScan(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kScan) {
    return false;
  }
  if (instr->IsRoot()) {
    return false;
  }
  if (!instr->shape().IsTuple()) {
    return false;
  }
  if (instr->shape().tuple_shapes().size() != 2) {
    return false;
  }

  const auto* scan = Cast<HloScanInstruction>(instr);
  for (const HloInstruction* user : scan->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    if (user->tuple_index() != 0) {
      return false;
    }
  }
  return true;
}

// Wraps the scan instruction and its output get-tuple-element into a custom
// Triton fusion computation (kind=__triton) and replaces all original uses.
absl::Status RewriteScanToTritonFusion(HloInstruction* scan) {
  HloComputation* parent = scan->parent();
  HloModule* module = parent->parent();
  const Shape& output_shape = scan->shape().tuple_shapes(0);

  HloComputation::Builder builder("triton_scan_computation");
  std::vector<HloInstruction*> parameters;
  parameters.reserve(scan->operand_count());
  for (int64_t i = 0; i < scan->operand_count(); ++i) {
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
  module->SetAndUniquifyInstrName(fusion, "triton_scan");

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  std::vector<HloInstruction*> users = scan->users();
  for (HloInstruction* user : users) {
    if (user->IsRoot()) {
      parent->set_root_instruction(fusion);
    }
    ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(user));
    ABSL_RETURN_IF_ERROR(user->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(fusion));
    ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(user));
  }

  ABSL_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(scan));
  ABSL_RETURN_IF_ERROR(scan->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(scan));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ScanRewriterTriton::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> scans_to_rewrite;
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
    ABSL_RETURN_IF_ERROR(RewriteScanToTritonFusion(scan));
  }
  return !scans_to_rewrite.empty();
}

}  // namespace gpu
}  // namespace xla
