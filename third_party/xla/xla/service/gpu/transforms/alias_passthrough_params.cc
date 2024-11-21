/* Copyright 2019 The OpenXLA Authors.

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
#include "xla/service/gpu/transforms/alias_passthrough_params.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> AliasPassthroughParams::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloInstruction* root = module->entry_computation()->root_instruction();
  if (module->entry_computation()->num_parameters() == 0 ||
      HloPredicateIsNotOp<HloOpcode::kTuple>(root)) {
    return false;
  }
  bool changed = false;
  absl::flat_hash_set<int64_t> used_params;
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    if (root->operand(i)->opcode() == HloOpcode::kParameter &&
        used_params.count(root->operand(i)->parameter_number()) == 0) {
      VLOG(2) << "Parameter " << root->operand(i)->parameter_number()
              << " with shape " << root->operand(i)->shape().ToString()
              << " in module " << module->name()
              << " is passed-through to root tuple element " << i << ": "
              << root->shape().ToString();

      if (module->input_output_alias_config().OutputHasAlias({i}) ||
          module->input_output_alias_config().ParameterHasAlias(
              root->operand(i)->parameter_number(), /*param_index=*/{})) {
        VLOG(2) << "Skip setting the above pass-through alias as an alias may"
                << " have been set up for alising resource update.";
        continue;
      }

      TF_RETURN_IF_ERROR(module->input_output_alias_config().SetUpAlias(
          /*output_index=*/{i},
          /*param_number=*/root->operand(i)->parameter_number(),
          /*param_index=*/{}));
      used_params.insert(root->operand(i)->parameter_number());
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
