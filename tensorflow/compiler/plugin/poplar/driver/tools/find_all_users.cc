/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include "absl/container/flat_hash_set.h"

namespace xla {
namespace poplarplugin {

namespace {

// Find the index of a tensor after extracting it (or a tuple containing it)
// from a tuple. tuple_index is the index of one of the elements of the tuple,
// and original_index is the tensor position within the original tuple.
int64 ExtractFromTuple(const Shape& tuple, int64 tuple_index,
                       int64 original_index) {
  int64 index = original_index;
  for (int64 i = 0; i < tuple_index; i++) {
    index -= CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  int64 n = CountShapes(ShapeUtil::GetTupleElementShape(tuple, tuple_index));
  if (index < 0 || index >= n) {
    return -1;
  }
  return index;
}

}  // namespace

void FindAllUsers::FindUsers(HloInstruction* tgt, const InstructionList& stack,
                             int64 index) {
  if (tgt->parent()->root_instruction() == tgt) {
    if (stack.size() > 0) {
      HloInstruction* caller = stack.back();
      InstructionList new_stack(stack);
      new_stack.pop_back();
      FindUsers(caller, new_stack, index);
    }
  } else {
    for (auto* user : tgt->users()) {
      for (int op_index = 0; op_index < user->operand_count(); op_index++) {
        if (user->operand(op_index) == tgt) {
          path.push_back(user);
          switch (user->opcode()) {
            case HloOpcode::kCall: {
              // This also handles repeat loops which are represented as a Call
              // operation.
              HloComputation* comp = user->to_apply();
              HloInstruction* param = comp->parameter_instruction(op_index);

              InstructionList new_stack(stack);
              new_stack.push_back(user);
              FindUsers(param, new_stack, index);
              break;
            }
            case HloOpcode::kFusion: {
              if (IsPopOpsFusion(user)) {
                paths.insert(path);
              }
              break;
            }
            case HloOpcode::kWhile: {
              HloComputation* comp = user->while_body();
              HloInstruction* param = comp->parameter_instruction(op_index);

              InstructionList new_stack(stack);
              new_stack.push_back(user);
              FindUsers(param, new_stack, index);
              break;
            }
            case HloOpcode::kTuple: {
              int64 new_index = InsertIntoTuple(user->shape(), op_index, index);
              FindUsers(user, stack, new_index);
              break;
            }
            case HloOpcode::kGetTupleElement: {
              int64 tuple_index = user->tuple_index();
              int64 new_index =
                  ExtractFromTuple(tgt->shape(), tuple_index, index);
              if (new_index != -1) {
                FindUsers(user, stack, new_index);
              }
              break;
            }
            default: {
              paths.insert(path);
              break;
            }
          }
          path.pop_back();
        }
      }
    }
  }

  return;
}

void FindAllUsers::Find(HloInstruction* inst) {
  path.clear();
  paths.clear();
  FindUsers(inst, {}, 0);
}

std::set<HloInstruction*> FindAllUsers::Users() const {
  std::set<HloInstruction*> users;
  for (auto& p : paths) {
    users.insert(p.back());
  }
  return users;
}

const std::set<InstructionList>& FindAllUsers::Paths() const { return paths; }

const InstructionList& FindAllUsers::PathFor(HloInstruction* target) const {
  for (auto& p : paths) {
    if (p.back() == target) {
      return p;
    }
  }

  static InstructionList empty = {};
  return empty;
}

}  // namespace poplarplugin
}  // namespace xla
