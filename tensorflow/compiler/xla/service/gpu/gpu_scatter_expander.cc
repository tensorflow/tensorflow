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

#include "tensorflow/compiler/xla/service/gpu/gpu_scatter_expander.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

StatusOr<bool> GpuScatterExpander::Run(HloModule* module) {
  auto is_nontrivial_scatter = [](HloInstruction* inst) {
    // TODO(b/129698548): Scattering elements larger than 64 bits is not
    // supported by XLA:GPU.
    return inst->opcode() == HloOpcode::kScatter &&
           inst->shape().element_type() == C128;
  };

  std::vector<HloInstruction*> scatter_instrs;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    absl::c_copy_if(computation->instructions(),
                    std::back_inserter(scatter_instrs), is_nontrivial_scatter);
  }

  for (HloInstruction* inst : scatter_instrs) {
    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root, ExpandScatter(inst));
    TF_RETURN_IF_ERROR(inst->parent()->ReplaceInstruction(inst, expanded_root));
  }

  return !scatter_instrs.empty();
}

}  // namespace xla
