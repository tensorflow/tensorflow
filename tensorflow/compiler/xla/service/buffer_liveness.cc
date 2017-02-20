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

// Defines the data returned by the XLA buffer assignment packages.

#include "tensorflow/compiler/xla/service/buffer_liveness.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */
StatusOr<std::unique_ptr<BufferLiveness>> BufferLiveness::Run(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering) {
  std::unique_ptr<BufferLiveness> liveness(
      new BufferLiveness(module, std::move(hlo_ordering)));
  TF_RETURN_IF_ERROR(liveness->Analyze());
  return std::move(liveness);
}

tensorflow::Status BufferLiveness::Analyze() {
  TF_ASSIGN_OR_RETURN(points_to_analysis_,
                      TuplePointsToAnalysis::Run(
                          module_, /*include_loop_fusion_instructions=*/true));
  for (auto& computation : module_->computations()) {
    // Gather all instructions whose buffers might alias other instructions into
    // the set aliased_buffers_.  This includes those contained as a tuple
    // element in other instruction's output.
    for (const auto& instruction : computation->instructions()) {
      for (const LogicalBuffer* aliased_buffer :
           points_to_analysis_->GetPointsToSet(instruction.get())
               .CreateFlattenedSet()) {
        if (aliased_buffer->instruction() != instruction.get()) {
          aliased_buffers_.insert(aliased_buffer);
        }
      }
    }

    if (computation.get() == module_->entry_computation()) {
      for (const LogicalBuffer* live_out_buffer :
           points_to_analysis_->GetPointsToSet(computation->root_instruction())
               .CreateFlattenedSet()) {
        maybe_live_out_buffers_.insert(live_out_buffer);
      }
    }
  }

  XLA_VLOG_LINES(3, ToString());
  return tensorflow::Status::OK();
}

string BufferLiveness::ToString() const {
  std::vector<string> pieces;
  pieces.push_back(tensorflow::strings::Printf("BufferLiveness(module=%s):",
                                               module_->name().c_str()));
  pieces.push_back("HloOrdering:");
  pieces.push_back(hlo_ordering_->ToString());
  pieces.push_back(tensorflow::strings::Printf("Aliased buffers:"));
  for (const LogicalBuffer* buffer : aliased_buffers_) {
    pieces.push_back(
        tensorflow::strings::Printf("  %s", buffer->ToString().c_str()));
  }
  pieces.push_back(tensorflow::strings::Printf("Live out buffers:"));
  for (const LogicalBuffer* buffer : maybe_live_out_buffers_) {
    pieces.push_back(
        tensorflow::strings::Printf("  %s", buffer->ToString().c_str()));
  }
  return tensorflow::str_util::Join(pieces, "\n");
}

namespace {

// Returns false if 'user' cannot possibly use the buffer at 'index' in
// 'operand'. Returns true otherwise.
// Precondition: 'operand' is an operand of 'user'.
bool MayUseBufferInOperand(HloInstruction* operand, const ShapeIndex& index,
                           HloInstruction* user,
                           const TuplePointsToAnalysis& points_to_analysis) {
  if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
    // GetTupleElement instructions only access the top-level buffer of their
    // operand.
    return false;
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
    // Return true if any uses are detected at 'index', returns false otherwise.
    const LogicalBuffer* buffer =
        points_to_analysis.GetBufferDefinedAt(*it, index).ValueOrDie();
    for (const BufferAlias& alias :
         points_to_analysis.GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (!MayUseBufferInOperand(alias.instruction(), alias.index(),
                                   alias_user, points_to_analysis)) {
          continue;
        }
        // Return true: use detected at 'buffer' -> 'alias' -> 'alias_user'.
        return true;
      }
    }
    // Return false: found no uses of 'operand' at 'index' in 'user'.
    return false;
  }
  return true;
}

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
        if (!MayUseBufferInOperand(alias.instruction(), alias.index(),
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

// Returns true if 'user' (at 'user_index') can share a buffer with its operand
// 'operand' (at 'operand_index').
// Returns false otherwise.
// User and operand can share buffers iff both instructions emit the same shape
// and layout, and 'user' meets one of the following two qualifications:
// *) Is element-wise.
// *) Is a loop fusion instruction where the only use of 'operand' at 'index'
//    in the set 'user.fused_instructions' is a DynamicUpdateSlice fused root
//    at operand 0.
bool CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index,
    const TuplePointsToAnalysis& points_to_analysis) {
  Shape operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  Shape user_subshape = ShapeUtil::GetSubshape(user->shape(), user_index);
  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
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
  // Check if 'user' is element-wise.
  return user->IsElementwise();
}

}  // anonymous namespace

bool BufferLiveness::live_range_strictly_before(const LogicalBuffer& a,
                                                const LogicalBuffer& b) const {
  TF_CHECK_OK(points_to_analysis_->VerifyBuffer(a));
  TF_CHECK_OK(points_to_analysis_->VerifyBuffer(b));

  if (!hlo_ordering_->ExecutesBefore(a.instruction(), b.instruction())) {
    return false;
  }

  // Every user of 'a' must be a predecessor of 'b' or 'b' itself.
  for (const BufferAlias& alias : points_to_analysis_->GetBufferAliases(a)) {
    for (auto user : alias.instruction()->users()) {
      if (!MayUseBufferInOperand(alias.instruction(), alias.index(), user,
                                 points_to_analysis())) {
        continue;
      }
      if (user != b.instruction() &&
          !hlo_ordering_->ExecutesBefore(user, b.instruction())) {
        return false;
      }
    }
  }

  // If 'b' is a user of 'a' then the buffers interfere unless 'a.instruction'
  // and 'b.instruction' emit the same shape/layout, and 'b.instruction' meets
  // one of following qualifications:
  // *) Is element-wise.
  // *) Is a loop fusion instruction (with DynamicUpdateSlice fused root) where
  //    the singleton use of 'a' at 'a.index' is the fused root at operand 0.
  for (const BufferAlias& alias : points_to_analysis_->GetBufferAliases(a)) {
    if (alias.instruction()->users().count(b.instruction()) > 0 &&
        !CanShareOperandBufferWithUser(alias.instruction(), alias.index(),
                                       b.instruction(), b.index(),
                                       points_to_analysis())) {
      return false;
    }
  }
  return true;
}

bool BufferLiveness::MayInterfere(const LogicalBuffer& a,
                                  const LogicalBuffer& b) const {
  return (!live_range_strictly_before(a, b) &&
          !live_range_strictly_before(b, a));
}

bool BufferLiveness::MaybeLiveOut(const LogicalBuffer& buffer) const {
  // Verify that a buffer is actually defined at the given instruction/index
  // (eg, its not an alias of another buffer such as occurs with a bitcast).
  TF_CHECK_OK(points_to_analysis_->VerifyBuffer(buffer));
  return maybe_live_out_buffers_.count(&buffer);
}

}  // namespace xla
