/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/copy_fusion.h"

#include <cstdint>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

bool OnlyElementwiseOpsReachableFromParams(HloComputation* fused_computation) {
  std::queue<const HloInstruction*> q;
  absl::flat_hash_set<const HloInstruction*> visited;
  for (auto param : fused_computation->parameter_instructions()) {
    q.push(param);
    visited.insert(param);
  }
  while (!q.empty()) {
    const HloInstruction* hlo = q.front();
    q.pop();
    for (auto user : hlo->users()) {
      if ((!user->IsElementwiseOnOperand(user->operand_index(hlo)) ||
           user->opcode() == HloOpcode::kCopy) &&
          user->opcode() != HloOpcode::kBitcast &&
          user->opcode() != HloOpcode::kTuple) {
        return false;
      }
      if (visited.insert(user).second) {
        q.push(user);
      }
    }
  }
  return true;
}

absl::StatusOr<bool> CopyFusion::DoCopyFusion(HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();

  for (HloInstruction* hlo : defs_before_uses) {
    if (hlo->opcode() != HloOpcode::kFusion) {
      continue;
    }
    std::vector<HloInstruction*> copies;
    std::vector<HloInstruction*> other_users;
    HloComputation* fused_computation = hlo->fused_instructions_computation();
    if (!OnlyElementwiseOpsReachableFromParams(fused_computation)) {
      continue;
    }
    HloInstruction* root = fused_computation->root_instruction();
    if (IsReductionFromOrToContiguousDimensions(*root, device_description_) ||
        root->opcode() == HloOpcode::kScatter ||
        (hlo->IsMultiOutputFusion() &&
         absl::c_all_of(root->operands(), [](const HloInstruction* slice) {
           return slice->opcode() == HloOpcode::kSlice;
         }))) {
      continue;
    }
    for (auto user : hlo->users()) {
      HloInstruction* copy_user = user;
      // Skip get-tuple-element ops.
      if (copy_user->opcode() == HloOpcode::kGetTupleElement &&
          copy_user->user_count() == 1) {
        if (IsReductionFromOrToContiguousDimensions(
                *(root->operand(copy_user->tuple_index())),
                device_description_)) {
          other_users.push_back(user);
          continue;
        }
        copy_user = copy_user->users()[0];
      }
      // Skip bitcast ops.
      if (copy_user->opcode() == HloOpcode::kBitcast &&
          copy_user->user_count() == 1) {
        copy_user = copy_user->users()[0];
      }
      if (copy_user->opcode() == HloOpcode::kCopy &&
          copy_user->shape() == copy_user->operand(0)->shape() &&
          !copy_user->shape().IsTuple() &&
          !copy_user->HasControlDependencies()) {
        copies.push_back(copy_user);
      } else {
        other_users.push_back(user);
      }
    }
    if (copies.empty()) {
      continue;
    }
    auto fusion_adaptor = HloFusionAdaptor::ForComputation(fused_computation);
    auto dynamic_update_slices =
        GetOutputDefiningDynamicUpdateSlices(fusion_adaptor->GetRoots());
    // Skip dynamic update slice fusions which might be emitted in-place.
    if (!dynamic_update_slices.empty() &&
        (root->opcode() != HloOpcode::kTuple ||
         dynamic_update_slices.size() == root->shape().tuple_shapes_size())) {
      continue;
    }
    changed = true;

    HloInstruction::InstructionVector tuple_elements;
    int64_t num_outputs =
        hlo->IsMultiOutputFusion() ? root->operand_count() : int64_t{1};
    tuple_elements.reserve(copies.size() + num_outputs);
    if (hlo->IsMultiOutputFusion()) {
      for (HloInstruction* operand : root->operands()) {
        tuple_elements.push_back(operand);
      }
    } else {
      tuple_elements.push_back(root);
    }

    for (auto copy : copies) {
      HloInstruction* user = copy;
      std::vector<HloInstruction*> operand_chain;
      operand_chain.push_back(user);
      while (user->operand(0) != hlo) {
        user = user->mutable_operand(0);
        operand_chain.push_back(user);
      }
      HloInstruction* clone_operand = root;
      if (hlo->IsMultiOutputFusion()) {
        clone_operand = root->mutable_operand(user->tuple_index());
        CHECK_EQ(operand_chain.back()->opcode(), HloOpcode::kGetTupleElement);
        operand_chain.pop_back();
      }
      for (int64_t i = operand_chain.size() - 1; i >= 0; --i) {
        HloInstruction* user = operand_chain[i];
        clone_operand = fused_computation->AddInstruction(
            user->CloneWithNewOperands(user->shape(), {clone_operand}));
      }
      tuple_elements.push_back(clone_operand);
    }

    HloInstruction* new_root = fused_computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_computation->set_root_instruction(new_root,
                                            /*accept_different_shape=*/true);
    *hlo->mutable_shape() = new_root->shape();

    if (root->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(fused_computation->RemoveInstruction(root));
    } else {
      auto get_tuple_element_root = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(hlo, 0));
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWithDifferentShape(
          other_users, get_tuple_element_root));
    }
    for (int64_t i = 0; i < copies.size(); ++i) {
      auto get_tuple_element = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(hlo, num_outputs + i));
      TF_RETURN_IF_ERROR(
          computation->ReplaceInstruction(copies[i], get_tuple_element));
    }
  }
  return changed;
}

absl::StatusOr<bool> CopyFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Only for the entry computation we can be sure that the copies do not share
  // a buffer with a parameter of the fusion that it will be fused with. For
  // example while loop computations have tuple parameters that need to share
  // the buffers with the output tuples, and copies inserted by the
  // CopyInsertion pass will share a buffer with the tuple output (and thus
  // with the tuple input as well).
  return DoCopyFusion(module->entry_computation());
}

}  // namespace gpu
}  // namespace xla
