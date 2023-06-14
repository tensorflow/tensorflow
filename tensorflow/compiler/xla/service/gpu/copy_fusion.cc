/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/copy_fusion.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

StatusOr<bool> CopyFusion::DoCopyFusion(HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();

  for (HloInstruction* hlo : defs_before_uses) {
    // TODO(b/285561974): Make this work with MultiOutputFusion.
    if (hlo->opcode() != HloOpcode::kFusion || hlo->IsMultiOutputFusion()) {
      continue;
    }
    std::vector<HloInstruction*> copies;
    std::vector<HloInstruction*> other_users;
    for (auto user : hlo->users()) {
      if (user->opcode() == HloOpcode::kCopy && user->shape() == hlo->shape() &&
          !user->HasControlDependencies()) {
        copies.push_back(user);
      } else {
        other_users.push_back(user);
      }
    }
    if (copies.size() <= 1) {
      continue;
    }
    HloComputation* fused_computation = hlo->fused_instructions_computation();
    HloInstruction* root = fused_computation->root_instruction();
    if (root->opcode() == HloOpcode::kDynamicUpdateSlice ||
        root->opcode() == HloOpcode::kReduce) {
      continue;
    }
    changed = true;

    HloInstruction::InstructionVector tuple_elements;
    tuple_elements.reserve(copies.size() + 1);
    tuple_elements.push_back(root);
    int64_t offset = tuple_elements.size();

    for (auto copy : copies) {
      HloInstruction* clone = fused_computation->AddInstruction(
          HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, root));
      tuple_elements.push_back(clone);
    }

    HloInstruction* new_root = fused_computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_computation->set_root_instruction(new_root,
                                            /*accept_different_shape=*/true);
    *hlo->mutable_shape() = new_root->shape();

    auto get_tuple_element_root = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(hlo, 0));
    TF_CHECK_OK(hlo->ReplaceAllUsesWithDifferentShape(other_users,
                                                      get_tuple_element_root));
    for (int64_t i = 0; i < copies.size(); ++i) {
      auto get_tuple_element = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(hlo, offset + i));
      TF_CHECK_OK(
          computation->ReplaceInstruction(copies[i], get_tuple_element));
    }
  }
  return changed;
}

StatusOr<bool> CopyFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool fusion_changed, DoCopyFusion(computation));
    changed |= fusion_changed;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
