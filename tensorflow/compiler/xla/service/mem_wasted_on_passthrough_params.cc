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
#include "tensorflow/compiler/xla/service/mem_wasted_on_passthrough_params.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

StatusOr<bool> MemWastedOnPassthroughParams::Run(HloModule* module) {
  const HloInstruction* root = module->entry_computation()->root_instruction();
  if (module->entry_computation()->num_parameters() == 0 ||
      root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  int64 pass_through_params_size = 0;
  for (int64 i = 0; i < root->operand_count(); ++i) {
    if (root->operand(i)->opcode() == HloOpcode::kParameter) {
      pass_through_params_size +=
          ShapeUtil::ByteSizeOf(root->operand(i)->shape());
      VLOG(2) << "Parameter " << root->operand(i)->parameter_number()
              << "in module " << module->name()
              << " is passed-through to root tuple element " << i << ".";
    }
  }
  if (pass_through_params_size > 0) {
    LOG(WARNING)
        << "Pass-through params incur a memory overhead of "
        << tensorflow::strings::HumanReadableNumBytes(pass_through_params_size)
        << ". Please refer to b/133276457 if you want to see this fixed.";
  }
  return false;
}

}  // namespace xla
