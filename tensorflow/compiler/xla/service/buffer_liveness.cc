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
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module_));
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

// Returns false if 'user' cannot possibly use the buffer at 'index' in
// 'operand'. Returns true otherwise.
// Precondition: 'operand' is an operand of 'user'.
bool MayUseBufferInOperand(HloInstruction* operand, const ShapeIndex& index,
                           HloInstruction* user) {
  if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
    // GetTupleElement instructions only access the top-level buffer of their
    // operand.
    return false;
  }
  return true;
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
      if (!MayUseBufferInOperand(alias.instruction(), alias.index(), user)) {
        continue;
      }
      if (user != b.instruction() &&
          !hlo_ordering_->ExecutesBefore(user, b.instruction())) {
        return false;
      }
    }
  }

  // If 'b' is a user of 'a' then the buffers interfere if b is not an
  // elementwise operation emitting the same shape/layout as 'a'.
  for (const BufferAlias& alias : points_to_analysis_->GetBufferAliases(a)) {
    if (alias.instruction()->users().count(b.instruction()) > 0 &&
        (!ShapeUtil::Equal(alias.instruction()->shape(),
                           b.instruction()->shape()) ||
         !b.instruction()->IsElementwise())) {
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
