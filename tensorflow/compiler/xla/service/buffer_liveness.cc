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

#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/liveness_util.h"
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
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module_));
  for (auto* computation : module_->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    // Gather all instructions whose buffers might alias other instructions into
    // the set aliased_buffers_.  This includes those contained as a tuple
    // element in other instruction's output.
    for (const auto& instruction : computation->instructions()) {
      for (const LogicalBuffer* aliased_buffer :
           points_to_analysis_->GetPointsToSet(instruction)
               .CreateFlattenedSet()) {
        if (aliased_buffer->instruction() != instruction) {
          aliased_buffers_.insert(aliased_buffer);
        }
      }
    }

    if (computation == module_->entry_computation()) {
      const HloInstruction* root = computation->root_instruction();
      maybe_live_out_buffers_ =
          points_to_analysis_->GetPointsToSet(root).CreateFlattenedSet();
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
      if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(), user,
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
  // the qualifications specified in CanShareOperandBufferWithUser.
  for (const BufferAlias& alias : points_to_analysis_->GetBufferAliases(a)) {
    if (b.instruction()->IsUserOf(alias.instruction()) &&
        !CanShareOperandBufferWithUser(alias.instruction(), alias.index(),
                                       b.instruction(), b.index(),
                                       points_to_analysis())) {
      return false;
    }
  }
  return true;
}

namespace {
bool IsEntryParameter(const HloInstruction* instruction) {
  const HloComputation* computation = instruction->parent();
  return instruction->opcode() == HloOpcode::kParameter &&
         computation == computation->parent()->entry_computation();
}
}  // namespace

bool BufferLiveness::MayInterfere(const LogicalBuffer& a,
                                  const LogicalBuffer& b) const {
  // Entry parameters live at the entry of the execution, thus always interfere
  // with all other instructions executing before them in the ordering.
  const HloInstruction* a_instruction = a.instruction();
  const HloInstruction* b_instruction = b.instruction();
  if (IsEntryParameter(a_instruction) &&
      hlo_ordering_->ExecutesBefore(b_instruction, a_instruction)) {
    return true;
  }
  if (IsEntryParameter(b_instruction) &&
      hlo_ordering_->ExecutesBefore(a_instruction, b_instruction)) {
    return true;
  }
  // Buffers without disjoint liveness may interfere.
  return !live_range_strictly_before(a, b) && !live_range_strictly_before(b, a);
}

bool BufferLiveness::MaybeLiveOut(const LogicalBuffer& buffer) const {
  // Verify that a buffer is actually defined at the given instruction/index
  // (eg, its not an alias of another buffer such as occurs with a bitcast).
  TF_CHECK_OK(points_to_analysis_->VerifyBuffer(buffer));
  return maybe_live_out_buffers_.count(&buffer);
}

}  // namespace xla
