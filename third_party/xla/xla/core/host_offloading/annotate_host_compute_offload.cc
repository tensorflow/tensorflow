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

#include "xla/core/host_offloading/annotate_host_compute_offload.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/host_offload_utils.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
void SetHostComputeFrontendAttribute(HloInstruction& host_instruction) {
  FrontendAttributes frontend_attributes =
      host_instruction.frontend_attributes();
  frontend_attributes.mutable_map()->insert(
      {kXlaComputeTypeAttr, kXlaComputeTypeHost});
  host_instruction.set_frontend_attributes(frontend_attributes);
}

void AnnotateComputationHostOffload(HloComputation& computation) {
  for (HloInstruction* host_instruction : computation.instructions()) {
    if (host_instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    SetHostComputeFrontendAttribute(*host_instruction);
    for (HloComputation* called_computation :
         host_instruction->called_computations()) {
      AnnotateComputationHostOffload(*called_computation);
    }
  }
}
}  // namespace

absl::StatusOr<bool> AnnotateHostComputeOffload::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall &&
          host_offload_utils::ComputeTypeIsHost(instruction)) {
        AnnotateComputationHostOffload(*instruction->to_apply());
        modified = true;
      }
    }
  }
  return modified;
}

}  // namespace xla
