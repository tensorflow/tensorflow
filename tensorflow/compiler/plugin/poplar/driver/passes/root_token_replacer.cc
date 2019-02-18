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

#include "tensorflow/compiler/plugin/poplar/driver/passes/root_token_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

RootTokenReplacer::RootTokenReplacer() {}

void UpdateLayout(HloModule* module, HloComputation* comp,
                  HloInstruction* new_root) {
  if (comp == module->entry_computation()) {
    auto* computation_layout = module->mutable_entry_computation_layout();
    auto* result_layout = computation_layout->mutable_result_layout();
    *result_layout = ShapeLayout(new_root->shape());
  }
}

std::vector<int> GetTokenIndices(int64 num_elements, const Shape& tuple) {
  std::vector<int> token_indices;
  for (int i = 0; i < num_elements; ++i) {
    const auto& element_shape = ShapeUtil::GetTupleElementShape(tuple, i);
    if (element_shape.IsToken()) {
      token_indices.push_back(i);
    }
  }

  return token_indices;
}

StatusOr<HloInstruction*> UpdateRootInstruction(
    std::unique_ptr<HloInstruction> new_root, HloComputation* comp,
    HloModule* module, HloInstruction* original_root) {
  HloInstruction* new_root_instr = comp->AddInstruction(std::move(new_root));
  TF_CHECK_OK(original_root->AddControlDependencyTo(new_root_instr));
  comp->set_root_instruction(new_root_instr, true);

  UpdateLayout(module, comp, new_root_instr);

  if (original_root->has_sharding()) {
    new_root_instr->set_sharding(original_root->sharding());
  }
  return new_root_instr;
}

StatusOr<bool> RootTokenReplacer::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    HloInstruction* root = comp->root_instruction();
    const Shape& root_shape = root->shape();

    if (root_shape.IsTuple()) {
      if (ShapeUtil::IsNestedTuple(root_shape)) {
        auto flattened_shapes = FlattenedXlaShape(root_shape);
        for (const auto& shape : flattened_shapes) {
          if (shape.IsToken()) {
            return Unimplemented("Tokens in nested tuples are not supported");
          }
        }
        return changed;
      }

      // check if any of the tuple elments are of token shape
      // create empty tuple element and add token as control dependency
      // create new tuple with token swapped out with newly created empty tuple
      const auto num_elements = ShapeUtil::TupleElementCount(root_shape);
      std::vector<int> token_indices =
          GetTokenIndices(num_elements, root_shape);
      if (token_indices.empty()) {
        continue;
      }

      const bool all_elements_are_tokens = token_indices.size() == num_elements;

      if (all_elements_are_tokens) {
        // Output single empty tuple with control dependency added to
        // original root instruction
        TF_ASSIGN_OR_RETURN(auto new_root, UpdateRootInstruction(
                                               HloInstruction::CreateTuple({}),
                                               comp, module, root));
        changed = true;
      } else if (token_indices.size() > 0) {
        // Some tuple elements are tokens, remove tokens from root tuple
        // set control dependency on new root from the removed tokens
        const auto num_elements_new_root = num_elements - token_indices.size();
        std::vector<HloInstruction*> new_root_instructions;
        new_root_instructions.reserve(num_elements_new_root);
        for (int i = 0; i < num_elements; ++i) {
          const auto& element_shape =
              ShapeUtil::GetTupleElementShape(root_shape, i);
          if (element_shape.IsToken()) {
            continue;
          }
          auto gte = comp->AddInstruction(
              HloInstruction::CreateGetTupleElement(element_shape, root, i));
          new_root_instructions.push_back(gte);
        }
        TF_ASSIGN_OR_RETURN(auto new_root,
                            UpdateRootInstruction(HloInstruction::CreateTuple(
                                                      new_root_instructions),
                                                  comp, module, root));
        changed = true;
      }
    } else if (root_shape.IsToken()) {
      TF_ASSIGN_OR_RETURN(auto new_root,
                          UpdateRootInstruction(HloInstruction::CreateTuple({}),
                                                comp, module, root));
      changed = true;
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
