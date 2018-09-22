/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module_dce.h"

#include <deque>
#include <unordered_set>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_liveness_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

StatusOr<bool> RunWhileDCE(HloModule* module, HloLivenessAnalysis* liveness) {
  bool changed = false;
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile) {
        continue;
      }

      const auto* xla_while = instruction;
      auto* while_body_comp = xla_while->while_body();
      auto* while_body_param = while_body_comp->parameter_instruction(0);
      auto* while_body_root = while_body_comp->root_instruction();

      if (!ShapeUtil::IsTuple(xla_while->shape()) ||
          while_body_root->opcode() != HloOpcode::kTuple) {
        // Only run DCE on tuple-shaped while loops where body root is Tuple,
        // with no I/O instructions.
        VLOG(1) << "WhileDCE SKIP while: " << xla_while->ToString();
        continue;
      }

      // Remove dead tuple elements.
      const int64 tuple_element_count =
          ShapeUtil::TupleElementCount(xla_while->shape());
      for (int64 i = 0; i < tuple_element_count; ++i) {
        if (liveness->IsLive(xla_while, {i})) {
          continue;
        }
        VLOG(1) << "WhileDCE Dead while tuple element."
                << " while: " << xla_while->name() << " tuple_index: " << i;
        // Transform while.body computation to make tuple element at
        // 'shape_index' as simple pass-through parameter (which candidate
        // be removed later by simplification pass).
        HloInstruction* pass_thru_gte = while_body_comp->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                while_body_param->shape().tuple_shapes(i), while_body_param,
                i));
        // Replace while.body.root Tuple operand at 'tuple_index' with
        // 'pass_thru_gte', making prior operand a dead root (to be cleaned
        // up with a subsequent DCE pass).
        TF_RETURN_IF_ERROR(
            while_body_root->ReplaceOperandWith(i, pass_thru_gte));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> HloModuleDCE::Run(HloModule* module) {
  VLOG(2) << "Before HloModuleDCE:";
  XLA_VLOG_LINES(3, module->ToString());

  std::unique_ptr<HloLivenessAnalysis> liveness;
  TF_ASSIGN_OR_RETURN(liveness, HloLivenessAnalysis::Run(*module));

  // Sweep through while instructions, transforming dead while tuple element
  // computations to pass through tuple values (creating dead roots in while
  // body computation in the process).
  TF_ASSIGN_OR_RETURN(bool hlo_module_dce_changed,
                      RunWhileDCE(module, liveness.get()));

  // Run HloDCE to clean up any dead code created during HloModuleDCE.
  HloDCE hlo_dce;
  TF_ASSIGN_OR_RETURN(bool hlo_dce_changed, hlo_dce.Run(module));

  VLOG(2) << "After HloModuleDCE:";
  XLA_VLOG_LINES(3, module->ToString());

  return hlo_module_dce_changed | hlo_dce_changed;
}

}  // namespace xla
