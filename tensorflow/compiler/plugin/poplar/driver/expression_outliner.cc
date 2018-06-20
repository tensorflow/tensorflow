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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/bcast.h"

namespace xla {
namespace poplarplugin {

using InplaceSet = std::set<const HloInstruction*>;

namespace {

bool IsPopopsElementwise(const HloInstruction* inst) {
  switch (inst->opcode()) {
    // Unary
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTanh:
      // Binary
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
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
      return !ShapeUtil::IsTuple(inst->shape());
    case HloOpcode::kClamp:
      return true;
    // Ops not supported in Expressions
    // Unary
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kImag:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    // Binary
    case HloOpcode::kComplex:
      return false;
    default:
      return false;
  }
}

}  // namespace

ExpressionOutliner::ExpressionOutliner(CompilerAnnotations& annotations)
    : HloMatcher({}, annotations, true),
      inplace_instructions(annotations.inplace_instructions) {}

ReplacedInstructions ExpressionOutliner::ReplaceNodes(
    int, const HloMatcherMatched&) {
  return {};
}

StatusOr<bool> ExpressionOutliner::Run(HloModule* module) {
  HloComputation* comp = module->entry_computation();
  XLA_VLOG_LINES(1, module->entry_computation()->ToString());

  for (auto* i:inplace_instructions) {
    LOG(INFO) << "INPLACE INSTRUCTION " << i->name();
  }

  std::list<HloInstruction*> all_ops;
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (IsPopopsElementwise(inst) && inst->user_count() == 1 &&
        inplace_instructions.count(inst) == 0) {
      bool add_op = true;
      if (inst->IsElementwiseBinary()) {
        // for BinaryOps check the shapes of inputs match
        const HloInstruction* in0 = inst->operand(0);
        const HloInstruction* in1 = inst->operand(1);
        const bool input_shapes_match =
            ShapeUtil::Equal(in0->shape(), in1->shape());
        if (!input_shapes_match) {
          // if shapes don't match check that they can be broadcasted to the
          // same shape
          tensorflow::BCast::Vec shape0 =
              convert_array<tensorflow::BCast::Vec>(in0->shape().dimensions());
          tensorflow::BCast::Vec shape1 =
              convert_array<tensorflow::BCast::Vec>(in1->shape().dimensions());

          const bool valid_bcast = tensorflow::BCast(shape0, shape1).IsValid();
          if (!valid_bcast) {
            add_op = false;
          }
        }
      } else if (inst->opcode() == HloOpcode::kClamp) {
        // don't add ClampOps for which inputs don't have the same shape as
        // output
        const bool shapes_match =
            ShapeUtil::Equal(inst->shape(), inst->operand(0)->shape()) &&
            ShapeUtil::Equal(inst->shape(), inst->operand(1)->shape()) &&
            ShapeUtil::Equal(inst->shape(), inst->operand(2)->shape());
        if (!shapes_match) {
          add_op = false;
        }
      } else if (inst->opcode() == HloOpcode::kSelect) {
        const HloInstruction* pred = inst->operand(0);
        const HloInstruction* in0 = inst->operand(1);
        const HloInstruction* in1 = inst->operand(2);
        // for Elementwise Select, predicate has to be scalar
        const bool pred_scalar = ShapeUtil::ElementsIn(pred->shape()) == 1;
        // or match the shape with the inputs
        const bool shapes_match =
            ShapeUtil::Equal(pred->shape(), in0->shape()) &&
            ShapeUtil::Equal(pred->shape(), in1->shape());
        if (!(pred_scalar || shapes_match)) {
          add_op = false;
        }
      }
      if (add_op) {
        all_ops.push_front(inst);
      }
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
      auto current =
          std::find(match.instructions.begin(), match.instructions.end(), inst);
      if (current != match.instructions.end()) {
        match.instructions.erase(current);
      }
      match.instructions.push_back(inst);
      outlined.insert(inst);

      for (auto* op : inst->operands()) {
        bool ok_to_outline =
            (std::find(all_ops.begin(), all_ops.end(), op) != all_ops.end());

        if (inplace_instructions.count(op) > 0) {
          ok_to_outline = false;
        }

        bool all_users_ok = true;
        for (auto* user : op->users()) {
          all_users_ok &=
              ((potential_set.count(user) > 0) || (outlined.count(user) > 0));
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

      OutlineExpressionFromComputation(match, "__arithmetic_expression", 0);
    }
  }

  return true;
}  // namespace poplarplugin

}  // namespace poplarplugin
}  // namespace xla
