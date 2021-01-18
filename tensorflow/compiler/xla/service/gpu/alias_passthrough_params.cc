/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

StatusOr<bool> AliasPassthroughParams::Run(HloModule* module) {
  const HloInstruction* root = module->entry_computation()->root_instruction();
  if (module->entry_computation()->num_parameters() == 0 ||
      root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  bool changed = false;
  absl::flat_hash_set<int64> used_params;
  for (int64 i = 0; i < root->operand_count(); ++i) {
    if (root->operand(i)->opcode() == HloOpcode::kParameter &&
        used_params.count(root->operand(i)->parameter_number()) == 0) {
      VLOG(2) << "Parameter " << root->operand(i)->parameter_number()
              << " with shape " << root->operand(i)->shape().ToString()
              << " in module " << module->name()
              << " is passed-through to root tuple element " << i << ": "
              << root->shape().ToString();
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
