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

#include "xla/backends/gpu/transforms/scan_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

absl::StatusOr<bool> ScanRewriter::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> scans;
  for (HloInstruction* inst : computation->instructions()) {
    if (inst->opcode() == HloOpcode::kScan) {
      scans.push_back(inst);
    }
  }

  for (HloInstruction* scan_instr : scans) {
    // Only handles inclusive prefix sums for now.
    if (scan_instr->operand_count() != 2) {
      continue;
    }
    auto* scan = xla::Cast<HloScanInstruction>(scan_instr);
    HloInstruction* root = scan->to_apply()->root_instruction();
    if (root->opcode() != HloOpcode::kTuple ||   //
        root->operand_count() != 2 ||            //
        root->operand(0) != root->operand(1) ||  //
        root->operand(0)->opcode() != HloOpcode::kAdd) {
      continue;
    }

    // Create the custom call.
    Shape scratch_shape =
        ShapeUtil::MakeShape(U8, {0});  // Empty shape, assigned later.
    Shape new_result_shape =
        ShapeUtil::MakeTupleShape({scan_instr->shape(), scratch_shape});

    HloInstruction* custom_call =
        computation->AddInstruction(HloInstruction::CreateCustomCall(
            new_result_shape, absl::MakeSpan(scan_instr->operands()),
            kCubDeviceScanUnassignedScratchSizeTarget));

    HloInstruction* get_tuple_element =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            scan_instr->shape(), custom_call, 0));

    TF_RETURN_IF_ERROR(scan_instr->ReplaceAllUsesWith(get_tuple_element));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(scan_instr));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> ScanRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace xla::gpu
