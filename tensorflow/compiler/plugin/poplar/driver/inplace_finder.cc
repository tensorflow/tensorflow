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

#include "tensorflow/compiler/plugin/poplar/driver/inplace_finder.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

void InplaceFinder::RouteFinder(HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kParameter:
    {
      if (ShapeUtil::IsTuple(inst->shape())) {
        tuple_stack.push_back(-1);
      }
      break;
    }
    case HloOpcode::kDynamicUpdateSlice:
    {
      if (inst->operand(0) != current_route.back()) {
        return;
      }
      break;
    }
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    {
      // Operation must be part of an TF core update
      const OpMetadata& md(inst->metadata());
      const std::string& tf_op(md.op_type());
      if (!(tf_op == "AssignAddVariableOp" ||
            tf_op == "AssignSubVariableOp" ||
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
    case HloOpcode::kTuple:
    {
      tuple_stack.push_back(inst->operand_index(current_route.back()));
      break;
    }
    case HloOpcode::kGetTupleElement:
    {
      if (inst->tuple_index() != tuple_stack.back()) {
        return;
      }
      tuple_stack.pop_back();
      break;
    }
    default:
      return;
  }
  current_route.push_back(inst);

  if (inst->user_count() == 0) {
    routes.insert(std::make_pair(current_route[0], current_route));
  } else {
    for (auto& user : inst->users()) {
      RouteFinder(user);
    }
  }

  current_route.pop_back();
}

Status InplaceFinder::FindInplaceInstructions(HloModule* module) {
  HloComputation* comp = module->entry_computation();

  // For each input
  const auto& params = comp->parameter_instructions();
  for (auto& p : params) {
    current_route.clear();
    RouteFinder(p);
  }

  // For each route in map
  for (auto& r : routes) {
    if (routes.count(r.first) == 1) {
      for (auto& inst : r.second) {
        inplace_instructions.insert(inst);
      }
    }
  }

  routes.clear();
  current_route.clear();

  return Status::OK();
}

}
}
