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

PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : module_(module) {}

bool PredecessorHloOrdering::ExecutesBefore(const HloInstruction* a,
                                            const HloInstruction* b) const {
  // Instructions in different computations are unordered.
  if (a->parent() != b->parent()) {
    return false;
  }
  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b'.
  return strict_predecessors_.at(b->parent())->IsReachable(b, a);
}

string PredecessorHloOrdering::ToStringHelper(const string& name) const {
  std::vector<string> pieces;
  pieces.push_back(name);
  for (auto& computation : module_->computations()) {
    pieces.push_back(tensorflow::strings::Printf("computation %s:",
                                                 computation->name().c_str()));
    const auto all = computation->MakeInstructionPostOrder();
    for (auto instruction : all) {
      pieces.push_back(tensorflow::strings::Printf(
          "  %s strict predecessors:", instruction->name().c_str()));
      for (auto predecessor : all) {
        if (strict_predecessors_.at(computation.get())
                ->IsReachable(instruction, predecessor)) {
          pieces.push_back(
              tensorflow::strings::Printf("  %s", predecessor->name().c_str()));
        }
      }
    }
  }
  return tensorflow::str_util::Join(pieces, "\n");
}

DependencyHloOrdering::DependencyHloOrdering(const HloModule* module)
    : PredecessorHloOrdering(module) {
  // Compute predecessor relationships between all instructions to determine
  // ordering based on dependencies. ExecutesBefore will return true iff there
  // exists a path in the HLO computation graph from 'a' to 'b'.
  for (auto& computation : module->computations()) {
    strict_predecessors_.emplace(computation.get(),
                                 computation->ComputeTransitiveOperands());
  }
}

string DependencyHloOrdering::ToString() const {
  return ToStringHelper("DependencyHloOrdering");
}

SequentialHloOrdering::SequentialHloOrdering(
    const HloModule* module, const HloModuleSequence& module_sequence)
    : module_(module) {
  // Create a map from instruction to its order position.
  for (auto computation_order : module_sequence) {
    const std::vector<const HloInstruction*>& order = computation_order.second;
    for (int i = 0; i < order.size(); ++i) {
      DCHECK_EQ(0, order_position_.count(order[i]));
      order_position_.emplace(order[i], i);
    }
  }
}

bool SequentialHloOrdering::ExecutesBefore(const HloInstruction* a,
                                           const HloInstruction* b) const {
  // Instructions in different computations are unordered.
  if (a->parent() != b->parent()) {
    return false;
  }
  // If either instruction is not in the order, then 'a' and 'b' are unordered.
  if (order_position_.count(a) == 0 || order_position_.count(b) == 0) {
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}

string SequentialHloOrdering::ToString() const {
  std::vector<string> pieces;
  pieces.push_back("SequentialHloOrdering");
  for (auto& computation : module_->computations()) {
    pieces.push_back(tensorflow::strings::Printf("computation %s order:",
                                                 computation->name().c_str()));
    // Gather all instructions in the module sequence for this computation and
    // sort them by their position.
    std::vector<const HloInstruction*> instructions;
    for (auto& instruction_position : order_position_) {
      const HloInstruction* instruction = instruction_position.first;
      if (instruction->parent() == computation.get()) {
        instructions.push_back(instruction);
      }
    }
    std::sort(instructions.begin(), instructions.end(),
              [this](const HloInstruction* a, const HloInstruction* b) {
                return order_position_.at(a) < order_position_.at(b);
              });
    for (auto instruction : instructions) {
      pieces.push_back(
          tensorflow::strings::Printf("  %s", instruction->name().c_str()));
    }
  }
  return tensorflow::str_util::Join(pieces, "\n");
}

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
