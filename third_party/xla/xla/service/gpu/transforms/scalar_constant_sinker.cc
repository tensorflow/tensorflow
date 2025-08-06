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

#include "xla/service/gpu/transforms/scalar_constant_sinker.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> ScalarConstantSinker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    std::optional<HloInstruction*> maybe_fusion =
        computation->GetUniqueCaller(HloOpcode::kFusion);
    if (!maybe_fusion) {
      continue;
    }

    const HloInstruction* fusion = *maybe_fusion;
    if (fusion->IsCustomFusion()) {
      continue;
    }

    for (int i = computation->num_parameters() - 1; i >= 0; --i) {
      HloInstruction* param = computation->parameter_instruction(i);
      if (!ShapeUtil::IsEffectiveScalar(param->shape())) {
        continue;
      }

      const HloInstruction* operand = fusion->operand(i);
      if (operand->opcode() != HloOpcode::kConstant) {
        continue;
      }

      // Clone the constant into the fusion, replace all uses of the parameter
      // and remove the parameter and the operand.
      TF_RETURN_IF_ERROR(
          computation->ReplaceWithNewInstruction(param, operand->Clone()));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
