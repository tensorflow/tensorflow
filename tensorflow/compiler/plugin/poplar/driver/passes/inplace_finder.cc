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

namespace xla {
namespace poplarplugin {
namespace {
enum class InplacePriority {
  kHigh = 0,
  kMedium,
  kLow,
};

using InplaceCandidates =
    std::map<InplacePriority, std::vector<HloInstruction*>>;
}  // namespace

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
  auto& inplace_instructions = annotations_.inplace_instructions;
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

    std::map<HloInstructionType, InplaceCandidates> inplace_candidates;

    InplaceCandidates& inplace_gte_candidates =
        inplace_candidates[HloInstructionType::kInplaceGetTupleElement];
    InplaceCandidates& inplace_read_write_candidates =
        inplace_candidates[HloInstructionType::kInplaceReadWrite];
    InplaceCandidates& inplace_read_only_candidates =
        inplace_candidates[HloInstructionType::kInplaceReadOnly];

    // For each route in map mark inplace ops as high priority inplace
    // candidates.
    for (auto& r : routes) {
      for (auto& inst : r.second) {
        switch (inst->opcode()) {
          case HloOpcode::kAdd:
          case HloOpcode::kFusion:
          case HloOpcode::kDynamicUpdateSlice:
          case HloOpcode::kMultiply:
          case HloOpcode::kSubtract: {
            inplace_read_write_candidates[InplacePriority::kHigh].push_back(
                inst);
            break;
          }
          case HloOpcode::kAddDependency: {
            inplace_read_only_candidates[InplacePriority::kHigh].push_back(
                inst);
            break;
          }
          case HloOpcode::kGetTupleElement: {
            inplace_gte_candidates[InplacePriority::kHigh].push_back(inst);
            break;
          }
          default:
            break;
        }
      }
    }
    // Get all possible remaining inplace instructions.
    // Give medium priority to outlined poplibs calls.
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      switch (inst->opcode()) {
        case HloOpcode::kCustomCall:
        case HloOpcode::kFusion: {
          inplace_read_write_candidates[InplacePriority::kMedium].push_back(
              inst);
          break;
        }
        default: {
          auto inst_description = HloInstructionDescription(inst);
          switch (inst_description.GetType()) {
            case HloInstructionType::kInplaceGetTupleElement: {
              inplace_gte_candidates[InplacePriority::kLow].push_back(inst);
              break;
            }
            case HloInstructionType::kInplaceReadWrite: {
              inplace_read_write_candidates[InplacePriority::kLow].push_back(
                  inst);
              break;
            }
            case HloInstructionType::kInplaceReadOnly: {
              inplace_read_only_candidates[InplacePriority::kLow].push_back(
                  inst);
              break;
            }
            default:
              break;
          }
          break;
        }
      }
    }

    // Because we are using a map, we first inplace GTEs, then Read/Write and
    // then Read-Only.
    for (auto type_candidates_pair : inplace_candidates) {
      auto& inplace_candidates_queues = type_candidates_pair.second;
      // Because we are using a map, all the candidate queues are sorted from
      // High to Low priority.
      for (auto inplace_priority_candidates_pair : inplace_candidates_queues) {
        auto& inplace_instruction_candidates =
            inplace_priority_candidates_pair.second;
        for (auto* inst : inplace_instruction_candidates) {
          if (HloInstructionDescription::IsInplace(inst, reachability_map.get(),
                                                   worklist_,
                                                   inplace_instructions)) {
            inplace_instructions.insert(inst);
            changed = true;
          }
        }
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
