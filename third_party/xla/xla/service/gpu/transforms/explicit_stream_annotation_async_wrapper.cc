/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

void ClearSchedulingAnnotations(HloInstruction* instr) {
  // These attributes are only valid on the async pairs.
  instr->erase_frontend_attribute(kXlaSchedulingGroupIdAttr);
  instr->erase_frontend_attribute(kXlaStreamAnnotationAttr);
}

static absl::StatusOr<bool> AsynchronizeInstruction(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCall ||
      !instr->frontend_attributes().map().contains(kXlaStreamAnnotationAttr)) {
    return false;
  }
  HloComputation* computation = instr->parent();
  auto original_attributes = instr->frontend_attributes();

  // These annotations are only legal on the async instructions and
  // can cause issues if the annotations remain on the inner operations,
  // so we clear them before creating the async pair.
  for (auto* inner_instr : instr->called_computations()[0]->instructions()) {
    ClearSchedulingAnnotations(inner_instr);
  }
  ClearSchedulingAnnotations(instr);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * done,
      computation->CreateAsyncInstructions(
          instr, {},
          ExplicitStreamAnnotationAsyncWrapper::kExplicitExecutionThread,
          /*replace=*/true));
  // Replace the original attributes after creating the async pair.
  done->set_frontend_attributes(original_attributes);
  done->mutable_operand(0)->set_frontend_attributes(original_attributes);
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      done->backend_config<GpuBackendConfig>());
  // Set earliest schedule of done op to be false so it can be scheduled
  // far apart from start.
  gpu_config.set_force_earliest_schedule(false);
  TF_RETURN_IF_ERROR(done->set_backend_config(gpu_config));
  VLOG(5) << "Created async instruction: " << done->ToString();
  return true;
}
}  // namespace

absl::StatusOr<bool> ExplicitStreamAnnotationAsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool result, AsynchronizeInstruction(instr));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace xla::gpu
