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

bool DoesNotUseOperandBuffer(HloInstruction* operand, const ShapeIndex& index,
                             HloInstruction* user,
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

namespace {

// Returns all uses of all aliases of 'instruction' at 'index' in 'uses'.
// Each use in 'uses' is a pair (HloInstruction* user, int64 operand_index)
// where 'user' is a user of an alias of 'intruction' at 'index', and
// 'operand_index' is the operand index at which the alias appears in the
// operand list of 'user'.
std::vector<std::pair<HloInstruction*, int64>> GetAllUsesOfInstructionAtIndex(
    HloInstruction* instruction, const ShapeIndex& index,
    const TuplePointsToAnalysis& points_to_analysis) {
  std::vector<std::pair<HloInstruction*, int64>> uses;
  const std::vector<const LogicalBuffer*>& points_to =
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

}  // namespace

// User and operand can share buffers iff both instructions emit the same shape
// and layout, and 'user' meets one of the following qualifications:
// *) Is element-wise. Or...
// *) Is a loop fusion instruction where the only use of 'operand' at 'index'
//    in the set 'user.fused_instructions' is a DynamicUpdateSlice fused root
//    at operand 0. Or...
// *) The 'user' of 'operand' is DynamicUpdateSlice or While at operand index 0.
bool CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index,
    const TuplePointsToAnalysis& points_to_analysis) {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  Shape operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  Shape user_subshape = ShapeUtil::GetSubshape(user->shape(), user_index);
  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
    return false;
  }
  // Copy instructions are explicitly added by CopyInsertion to prevent liveness
  // issues, so they should never re-use their operand buffer.
  if (user->opcode() == HloOpcode::kCopy) {
    return false;
  }
  // Check if 'user' is a loop fusion instruction with a kDynamicUpdateSlice
  // fused root instruction.
  if (user->opcode() == HloOpcode::kFusion &&
      user->fusion_kind() == HloInstruction::FusionKind::kLoop &&
      user->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice) {
    for (auto& fused_param : user->fused_parameters()) {
      // Find fusion parameter associated with 'operand'.
      if (user->operand(fused_param->parameter_number()) != operand) {
        continue;
      }
      // Get all uses of 'operand' at 'index' from 'user.fused_instructions'.
      auto fused_param_uses = GetAllUsesOfInstructionAtIndex(
          fused_param, operand_index, points_to_analysis);
      // Return true iff there is exactly one use of 'operand' at 'index', and
      // this singleton use is the fused root at operand index 0.
      if (fused_param_uses.size() == 1 &&
          fused_param_uses[0].first == user->fused_expression_root() &&
          fused_param_uses[0].second == 0) {
        return true;
      }
      break;
    }
    return false;
  }
  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in BufferLiveness::live_range_strictly_before,
    // so here we just need to check that the use is at operand index 0.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && operand_indices[0] == 0;
  }
  // Check if 'user' is element-wise.
  return user->IsElementwise();
}

}  // namespace xla
