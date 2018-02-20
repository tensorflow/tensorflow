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

#include "tensorflow/compiler/xla/service/liveness_util.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                             const ShapeIndex& index,
                             const HloInstruction* user,
                             const TuplePointsToAnalysis& points_to_analysis) {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
    // GetTupleElement instructions only access the top-level buffer of their
    // operand.
    return true;
  } else if (user->opcode() == HloOpcode::kFusion &&
             user->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    // Find fusion parameter associated with 'operand'.
    auto it = std::find_if(
        user->fused_parameters().begin(), user->fused_parameters().end(),
        [=](HloInstruction* fused_param) {
          return user->operand(fused_param->parameter_number()) == operand;
        });
    CHECK(it != user->fused_parameters().end());
    // Iterate through all users of all buffer aliases of the buffer in the
    // points-to set of fusion parameter at 'index'.
    // Return false if any uses are detected at 'index', returns true otherwise.
    const LogicalBuffer* buffer =
        points_to_analysis.GetBufferDefinedAt(*it, index).ValueOrDie();
    for (const BufferAlias& alias :
         points_to_analysis.GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user, points_to_analysis)) {
          continue;
        }
        // Return false: use detected at 'buffer' -> 'alias' -> 'alias_user'.
        return false;
      }
    }
    // Return true: found no uses of 'operand' at 'index' in 'user'.
    return true;
  }
  return false;
}

bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                             const ShapeIndex& index,
                             const HloInstruction* user,
                             const HloDataflowAnalysis& dataflow) {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (user->opcode() == HloOpcode::kFusion &&
      user->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    // Find fusion parameter associated with 'operand'.
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    // Iterate through all users of all uses of the fusion parameter value.
    // Return false if any uses are detected, returns true otherwise.
    const HloValue& value = dataflow.GetValueDefinedAt(fusion_param, index);
    return value.uses().empty();
  } else {
    // Return false if no value at 'operand' and 'index' is used at 'user'.
    for (const HloValue* value :
         dataflow.GetValueSet(operand, index).values()) {
      for (const HloUse& use : value->uses()) {
        if (use.instruction == user) {
          return false;
        }
      }
    }
  }

  return true;
}

namespace {

// Returns all uses of all aliases of 'instruction' at 'index' in 'uses'.
// Each use in 'uses' is a pair (HloInstruction* user, int64 operand_index)
// where 'user' is a user of an alias of 'instruction' at 'index', and
// 'operand_index' is the operand index at which the alias appears in the
// operand list of 'user'.
std::vector<std::pair<HloInstruction*, int64>> GetAllUsesOfInstructionAtIndex(
    HloInstruction* instruction, const ShapeIndex& index,
    const TuplePointsToAnalysis& points_to_analysis) {
  std::vector<std::pair<HloInstruction*, int64>> uses;
  const PointsToSet::BufferList& points_to =
      points_to_analysis.GetPointsToSet(instruction).element(index);
  for (const LogicalBuffer* buffer : points_to) {
    for (const BufferAlias& alias :
         points_to_analysis.GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user, points_to_analysis)) {
          continue;
        }
        for (int64 op_idx : alias_user->OperandIndices(alias.instruction())) {
          uses.emplace_back(alias_user, op_idx);
        }
      }
    }
  }
  return uses;
}

// Returns true if there is exactly one use of 'operand' at 'operand_index'
// in 'fusion.fused_instructions', where the singleton use is the fused
// root at operand index 'use_operand_index'. Returns false otherwise.
//
// REQUIRES: 'fusion' opcode is a kFusion instruction.
bool HasUniqueFusedUseOfOperandAt(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* fusion, const int64 use_operand_index,
    const TuplePointsToAnalysis& points_to_analysis) {
  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  // Check that 'operand' is unique in the operand list of 'fusion'.
  if (fusion->OperandIndices(operand).size() > 1) {
    return false;
  }
  // Find fusion parameter associated with 'operand'.
  const auto& fused_params = fusion->fused_parameters();
  auto fused_param_it = std::find_if(
      fused_params.begin(), fused_params.end(),
      [&](HloInstruction* fused_param) {
        return fusion->operand(fused_param->parameter_number()) == operand;
      });
  if (fused_param_it == fused_params.end()) {
    return false;
  }
  auto* fused_param = *fused_param_it;
  // Get all uses of 'operand' at 'index' from 'fusion.fused_instructions'.
  auto fused_param_uses = GetAllUsesOfInstructionAtIndex(
      fused_param, operand_index, points_to_analysis);
  // Return true iff there is exactly one use of 'operand' at 'index', and
  // this singleton use is the fused root (at index in 'use_operand_indices').
  return fused_param_uses.size() == 1 &&
         fused_param_uses[0].first == fusion->fused_expression_root() &&
         fused_param_uses[0].second == use_operand_index;
}

}  // namespace

// User and operand can share buffers iff both instructions emit the same shape
// and layout, and 'user' meets one of the following qualifications:
//
// (1) Is element-wise. Or...
// (2) Is a loop fusion instruction where the only use of 'operand' at 'index'
//     in the set 'user.fused_instructions' is a DynamicUpdateSlice fused root
//     at operand 0. Or...
// (3) Is a kDot -> kAdd (or fused kTransposeDot -> kAdd) output fusion
//     instruction where the only use of 'operand' at 'index' in the set
//     'user.fused_instructions' is a kAdd fused root at operand 0 or 1. Or...
// (4) The 'user' of 'operand' is DynamicUpdateSlice or While at operand index
//     0.
//
// (2) and (3) can only be determined if points-to analysis is available.
bool CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index,
    const TuplePointsToAnalysis& points_to_analysis) {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);
  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
    return false;
  }
  if (user->opcode() == HloOpcode::kFusion) {
    if (user->fusion_kind() == HloInstruction::FusionKind::kLoop &&
        user->fused_expression_root()->opcode() ==
            HloOpcode::kDynamicUpdateSlice) {
      // Loop fusion with kDynamicUpdateSlice fused root.
      //
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root at operand
      // index 0.
      return HasUniqueFusedUseOfOperandAt(operand, operand_index, user, 0,
                                          points_to_analysis);
    } else if (user->fusion_kind() == HloInstruction::FusionKind::kOutput &&
               user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      // Output fusion with kAdd fused root.

      // Check if one operand of kAdd fused root is either kDot, or nested
      // kFusion of kind kTransposeDot.
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          std::find_if(add->operands().begin(), add->operands().end(),
                       [&](HloInstruction* operand) {
                         return operand->opcode() == HloOpcode::kConvolution ||
                                operand->opcode() == HloOpcode::kDot ||
                                (operand->opcode() == HloOpcode::kFusion &&
                                 operand->fusion_kind() ==
                                     HloInstruction::FusionKind::kTransposeDot);
                       });
      if (add_operand_it == add->operands().end()) {
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      // Calculate operand index of 'add' operand which was not matched above.
      const int64 other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root (at operand
      // index 'other_add_operand_index').
      return HasUniqueFusedUseOfOperandAt(operand, operand_index, user,
                                          other_add_operand_index,
                                          points_to_analysis);
    }
  }
  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in BufferLiveness::live_range_strictly_before,
    // so here we just need to check that the use is at operand index 0.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && operand_indices[0] == 0;
  }
  if (user->opcode() == HloOpcode::kCall) {
    // TODO(b/62548313): Remove when buffer assignment is module scoped and
    // does not assign buffers to calls.
    // Find called computation parameter associated with 'operand'.
    const std::vector<int64> operand_indices = user->OperandIndices(operand);
    if (operand_indices.size() > 1) {
      return false;
    }
    CHECK_EQ(1, operand_indices.size());
    auto* param = user->to_apply()->parameter_instruction(operand_indices[0]);
    // Get all uses of 'operand' at 'index' in called computation.
    auto param_uses = GetAllUsesOfInstructionAtIndex(param, operand_index,
                                                     points_to_analysis);

    // Return true iff:
    // *) There exists exactly one use of 'operand' in called computation.
    // *) The unique use is by the root instruction of called computation.
    //    (Note: we check the root of the called computation, because the
    //     root result buffer is required to alias with the Call result buffer).
    // *) The root instruction of the called computation is element-wise on
    //    'operand'.
    auto* callee_root = user->to_apply()->root_instruction();
    return param_uses.size() == 1 && param_uses[0].first == callee_root &&
           callee_root->IsElementwiseOnOperand(param_uses[0].second);
  }
  // Check if 'user' is element-wise.
  return user->IsElementwise();
}

bool CanShareOperandBufferWithUser(HloInstruction* operand,
                                   const ShapeIndex& operand_index,
                                   HloInstruction* user,
                                   const ShapeIndex& user_index,
                                   const HloDataflowAnalysis& dataflow) {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);
  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
    return false;
  }

  if (user->opcode() == HloOpcode::kFusion) {
    // Get the parameter associated with 'operand';
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));

    const HloValue& value =
        dataflow.GetValueDefinedAt(fusion_param, operand_index);
    if (value.uses().size() != 1) {
      return false;
    }
    const HloUse& use = value.uses()[0];

    if (user->fusion_kind() == HloInstruction::FusionKind::kLoop &&
        user->fused_expression_root()->opcode() ==
            HloOpcode::kDynamicUpdateSlice) {
      // Loop fusion with kDynamicUpdateSlice fused root.
      //
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root at operand
      // index 0.
      return use.instruction == user->fused_expression_root() &&
             use.operand_number == 0;
    } else if (user->fusion_kind() == HloInstruction::FusionKind::kOutput &&
               user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      // Output fusion with kAdd fused root.

      // Check if one operand of kAdd fused root is either kDot, or nested
      // kFusion of kind kTransposeDot.
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          std::find_if(add->operands().begin(), add->operands().end(),
                       [&](HloInstruction* operand) {
                         return operand->opcode() == HloOpcode::kConvolution ||
                                operand->opcode() == HloOpcode::kDot ||
                                (operand->opcode() == HloOpcode::kFusion &&
                                 operand->fusion_kind() ==
                                     HloInstruction::FusionKind::kTransposeDot);
                       });
      if (add_operand_it == add->operands().end()) {
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      // Calculate operand index of 'add' operand which was not matched above.
      const int64 other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root (at operand
      // index 'other_add_operand_index').
      return use.instruction == user->fused_expression_root() &&
             use.operand_number == other_add_operand_index;
    }
  }
  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in BufferLiveness::live_range_strictly_before,
    // so here we just need to check that the use is at operand index 0.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && operand_indices[0] == 0;
  }
  if (user->opcode() == HloOpcode::kCall) {
    // Get all uses of value defined by 'operand' at 'operand_index'.
    const auto& uses =
        dataflow.GetValueDefinedAt(operand, operand_index).uses();
    // Return true iff:
    // *) There exists two uses of 'operand'.
    // *) One use is by 'user' (caller).
    // *) One use is by root instruction of called computation (callee root).
    //    (Note: we check the root of the called computation, because the
    //     root result buffer is required to alias with the Call result buffer).
    // *) The root instruction of the called computation is element-wise on
    //    'operand'.
    const bool found_caller_use =
        std::find_if(uses.begin(), uses.end(), [user](const HloUse& use) {
          return use.instruction == user;
        }) != uses.end();
    auto* callee_root = user->to_apply()->root_instruction();
    const bool found_elementwise_callee_use =
        std::find_if(
            uses.begin(), uses.end(), [callee_root](const HloUse& use) {
              return use.instruction == callee_root &&
                     callee_root->IsElementwiseOnOperand(use.operand_number);
            }) != uses.end();
    return uses.size() == 2 && found_caller_use && found_elementwise_callee_use;
  }
  // Check if 'user' is element-wise.
  return user->IsElementwise();
}

}  // namespace xla
