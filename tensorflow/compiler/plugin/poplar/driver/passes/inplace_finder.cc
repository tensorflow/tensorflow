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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include <deque>

namespace xla {
namespace poplarplugin {
void InplaceFinder::RouteFinder(HloInstruction* inst,
                                const std::vector<int64>& stack) {
  std::vector<int64> new_stack;
  bool tuple_stack_modified = false;

  switch (inst->opcode()) {
    case HloOpcode::kParameter: {
      if (inst->shape().IsTuple()) {
        new_stack = stack;
        tuple_stack_modified = true;
        new_stack.push_back(-1);
      }
      break;
    }
    case HloOpcode::kDynamicUpdateSlice: {
      if (inst->operand(0) != current_route.back()) {
        return;
      }
      break;
    }
    case HloOpcode::kAddDependency: {
      if (inst->operand(0) != current_route.back()) {
        return;
      }
      break;
    }
    case HloOpcode::kFusion:
      if (IsPopOpsFusion(inst, "conv_scaled_inplace")) {
        // This is always acceptable on a variable update inplace route
        break;
      }
      if (!IsPopOpsFusion(inst, "scaled_inplace")) {
        return;
      }
    // Fall through since inplace subgraphs have to pass all the same
    // criteria
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply: {
      // Operation must be part of an TF core update
      const OpMetadata& md(inst->metadata());
      const std::string& tf_op(md.op_type());
      if (!(tf_op == "AssignAddVariableOp" || tf_op == "AssignSubVariableOp" ||
            tf_op == "ResourceApplyGradientDescent" ||
            tf_op == "ResourceApplyMomentum" ||
            tf_op == "ResourceApplyAdagrad" ||
            tf_op == "ResourceApplyRMSProp")) {
        return;
      }
      if (inst->operand(0) != current_route.back()) {
        return;
      }
      if (!ShapeUtil::Equal(inst->operand(0)->shape(), inst->shape())) {
        return;
      }
      break;
    }
    case HloOpcode::kTuple: {
      new_stack = stack;
      tuple_stack_modified = true;
      new_stack.push_back(inst->operand_index(current_route.back()));
      break;
    }
    case HloOpcode::kGetTupleElement: {
      if (!stack.empty()) {
        if (inst->tuple_index() == stack.back() || stack.back() == -1) {
          new_stack = stack;
          tuple_stack_modified = true;
          new_stack.pop_back();
          break;
        }
      }
      return;
    }
    default:
      return;
  }
  current_route.push_back(inst);

  if (inst->user_count() == 0) {
    routes.insert(std::make_pair(current_route[0], current_route));
  } else {
    for (auto& user : inst->users()) {
      RouteFinder(user, tuple_stack_modified ? new_stack : stack);
    }
  }

  current_route.pop_back();
}

StatusOr<bool> InplaceFinder::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // The reachability map is used for adding and finding control dependencies
    // in order to allow for inplace ops to be executed after other instructions
    // which are using the inplace input.
    std::unique_ptr<HloReachabilityMap> reachability_map =
        HloReachabilityMap::Build(comp);

    // For each input
    const auto& params = comp->parameter_instructions();
    for (auto& p : params) {
      current_route.clear();
      RouteFinder(p, {});
    }

    // For each route in map
    for (auto& r : routes) {
      for (auto& inst : r.second) {
        switch (inst->opcode()) {
          case HloOpcode::kAdd:
          case HloOpcode::kAddDependency:
          case HloOpcode::kFusion:
          case HloOpcode::kDynamicUpdateSlice:
          case HloOpcode::kGetTupleElement:
          case HloOpcode::kMultiply:
          case HloOpcode::kSubtract:
            if (InplaceUtil::IsInPlace(inst, reachability_map.get())) {
              annotations_.inplace_instructions.insert(inst);
              changed = true;
            }
          default:
            break;
        }
      }
    }
    // Get all possible remaining inplace instructions, giving higher priority
    // to outlined poplibs calls.
    std::deque<HloInstruction*> inplace_instructions_queue;
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      // Skip instructions already inplace.
      if (annotations_.inplace_instructions.count(inst)) {
        continue;
      }

      switch (inst->opcode()) {
        case HloOpcode::kAddDependency:
        case HloOpcode::kBitcast:
        case HloOpcode::kBroadcast:
        case HloOpcode::kCall:
        case HloOpcode::kConcatenate:
        case HloOpcode::kCustomCall:
        case HloOpcode::kDynamicUpdateSlice:
        case HloOpcode::kFusion:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kMap:
        case HloOpcode::kReshape:
        case HloOpcode::kSlice:
        case HloOpcode::kSort:
        case HloOpcode::kTranspose:
        case HloOpcode::kTuple:
        case HloOpcode::kPad:
        case HloOpcode::kWhile:
          inplace_instructions_queue.push_front(inst);
          break;
        case HloOpcode::kAdd:
        case HloOpcode::kSubtract:
          inplace_instructions_queue.push_back(inst);
          break;
        default:
          break;
      }
    }

    for (auto* inst : inplace_instructions_queue) {
      if (InplaceUtil::IsInPlace(inst, reachability_map.get())) {
        annotations_.inplace_instructions.insert(inst);
        changed = true;
      }
    }
    routes.clear();
    current_route.clear();
  }

  return changed;
}

InplaceFinder::InplaceFinder(CompilerAnnotations& annotations)
    : annotations_(annotations) {}

}  // namespace poplarplugin
}  // namespace xla
