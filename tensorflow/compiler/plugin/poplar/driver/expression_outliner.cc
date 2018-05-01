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

#include "tensorflow/compiler/plugin/poplar/driver/expression_outliner.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

bool IsPopopsElementwise(const HloInstruction *inst) {
  switch (inst->opcode()) {
    // Unary
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTanh:
      // Binary
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      // Ternary
    case HloOpcode::kSelect:
    case HloOpcode::kClamp:
      return true;
    default:
      return false;
  }
}

}

ExpressionOutliner::ExpressionOutliner() : HloMatcher({}, true) {}

ReplacedInstructions
ExpressionOutliner::ReplaceNodes(int, const HloMatcherMatched&) {
  return {};
}

StatusOr<bool> ExpressionOutliner::Run(HloModule *module) {

  HloComputation* comp = module->entry_computation();

  std::list<HloInstruction*> all_ops;
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (IsPopopsElementwise(inst)) {
      all_ops.push_front(inst);
    }
  }

  while (all_ops.size() > 0) {
    HloInstruction* root = all_ops.front();
    all_ops.pop_front();

    HloMatcherMatched match;
    match.computation = comp;
    match.ok = true;

    std::list<HloInstruction*> potential_list;
    std::set<HloInstruction*> potential_set;
    std::map<HloInstruction*, HloInstruction*> parameter_map;

    std::set<HloInstruction*> outlined;

    potential_list.push_back(root);

    while (potential_list.size() > 0) {
      HloInstruction* inst = potential_list.front();
      potential_list.pop_front();
      potential_set.erase(inst);
      auto current = std::find(match.instructions.begin(),
                               match.instructions.end(), inst);
      if (current != match.instructions.end()) {
        match.instructions.erase(current);
      }
      match.instructions.push_back(inst);
      outlined.insert(inst);

      for (auto* op : inst->operands()) {
        bool ok_to_outline =
            (std::find(all_ops.begin(), all_ops.end(), op) != all_ops.end());

        bool all_users_ok=true;
        for (auto* user : op->users()) {
          all_users_ok &= ((potential_set.count(user) > 0) ||
                           (outlined.count(user) > 0));
        }
        if (ok_to_outline && all_users_ok) {
          if (potential_set.count(op) == 0) {
            potential_list.push_back(op);
            potential_set.insert(op);
            parameter_map.erase(op);
          }
        } else {
          parameter_map[op] = inst;
        }
      }
    }

    for (auto* inst : match.instructions) {
      all_ops.remove(inst);
    }

    if (match.instructions.size() > 1) {
      for (auto&& p : parameter_map) {
        auto pr = std::make_pair(p.second, p.second->operand_index(p.first));
        match.parameters.push_back(pr);
      }

      OutlineExpressionFromComputation(match, "arithmetic", 0);
    }
  }

  return true;
}

}
}
