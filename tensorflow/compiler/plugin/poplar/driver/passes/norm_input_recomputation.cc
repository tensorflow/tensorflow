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

#include "tensorflow/compiler/plugin/poplar/driver/passes/norm_input_recomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

NormInputRecomputation::NormInputRecomputation(bool recompute_norm_inputs)
    : recompute_norm_inputs_(recompute_norm_inputs) {}

StatusOr<bool> NormInputRecomputation::Run(HloModule* module) {
  if (!recompute_norm_inputs_) {
    return false;
  }

  bool changed = false;
  /* We look for the following patterns:
  ** clang-format off
  **  ---------------
  ** |      IN       |
  **  ---------------
  **        ||
  **        ||
  **        \/
  **  ---------------
  ** |       X       |
  **  ---------------                                      ....
  **        ||                                              /\
  **        ||==========================                    ||
  **        \/                        ||                    ||
  **  ---------------                 ||           ---------------
  ** |     NORM      |                ||          |   NORM-GRAD   |
  **  ---------------                 ||           ---------------
  **        ||                        ||             /\    /\
  **        ||                        ||=============||    ||
  **        \/                                             ||
  **        ..                                             ..
  ** clang-format on
  **
  ** When this pattern is matched we transform it into following:
  ** clang-format off
  **  ---------------
  ** |      IN       |
  **  ---------------
  **        ||
  **        ||========================||
  **        \/                        \/
  **  ---------------           ---------------
  ** |       X       |         |       X       |
  **  ---------------           ---------------            ....
  **        ||                        ||                    /\
  **        ||                        ||                    ||
  **        \/                        ||                    ||
  **  ---------------                 ||           ---------------
  ** |     NORM      |                ||          |   NORM-GRAD   |
  **  ---------------                 ||           ---------------
  **        ||                        ||             /\    /\
  **        ||                        ||=============||    ||
  **        \/                                             ||
  **        ..                                             ..
  ** clang-format on
  ** By recomputing the norm input for the norm grad we might be able to reduce
  ** the number of activations being always live, as long as calculating
  ** gradient of X requires IN (which is usually the case).
  */

  struct RecompInfo {
    HloInstruction* norm_input;
    HloInstruction* norm_training;
    HloInstruction* norm_grad;

    RecompInfo(HloInstruction* norm_input, HloInstruction* norm_training,
               HloInstruction* norm_grad)
        : norm_input(norm_input),
          norm_training(norm_training),
          norm_grad(norm_grad) {}
  };
  std::string before = module->ToString();
  std::vector<RecompInfo> recomp_infos;
  int64 norm_count = 0;

  for (HloComputation* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      // Find all the norm training ops.
      if (IsNormTraining(inst)) {
        norm_count++;
        HloInstruction* norm_training = inst;

        // Check that the norm training input is only used by the norm training
        // and norm grad op.
        HloInstruction* norm_input = norm_training->mutable_operand(0);
        if (norm_input->user_count() != 2) {
          continue;
        }
        HloInstruction* norm_grad = norm_input->users()[0] == norm_training
                                        ? norm_input->users()[1]
                                        : norm_input->users()[0];
        if (!IsNormGradient(norm_grad)) {
          continue;
        }

        recomp_infos.push_back(
            RecompInfo(norm_input, norm_training, norm_grad));
      }
    }
  }

  for (RecompInfo recomp_info : recomp_infos) {
    // We clone the norm input, use it as the input to the norm grad and then
    // add a control dependency such that the clone of the norm input is
    // executed after the gradients flowing into the norm grad operation.
    HloComputation* comp = recomp_info.norm_input->parent();
    HloInstruction* norm_input_clone =
        comp->AddInstruction(recomp_info.norm_input->Clone());
    TF_RETURN_IF_ERROR(
        recomp_info.norm_grad->ReplaceOperandWith(0, norm_input_clone));
    TF_RETURN_IF_ERROR(
        recomp_info.norm_grad->mutable_operand(4)->AddControlDependencyTo(
            norm_input_clone));

    changed = true;
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
