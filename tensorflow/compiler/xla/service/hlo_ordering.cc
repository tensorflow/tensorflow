/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
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

namespace {

// Returns the nearest call graph ancestors of instructions 'a' and 'b' for
// which the ancestors are in the same computation. An instruction is an call
// graph ancestor of 'a' if the instruction calls the computation containing 'a'
// either directly or transitively. Degeneratively an instruction is an ancestor
// of itself. nullptr is returned if there is no common ancestor or if the
// caller chain of 'a' or 'b' diverges (has multiple callers) before the nearest
// common ancestor.
//
// Example:
//
// Entry computation:
//   %x = Call(A, {Constant(42.0)})
//   %y = Call(B, {%x})
//
// Computation A:
//   %a = Negate(Param())
//
// Computation B:
//   %b = Exp(Param());
//
// If called with %a and %b, this function would return (%x, %y). %x is an
// ancestor of %a, and %y is an ancestor of %b, and %x and %y are in the same
// computation.
std::pair<const HloInstruction*, const HloInstruction*>
GetNearestCallGraphAncestorsInSameComputation(const HloInstruction* a,
                                              const HloInstruction* b,
                                              const CallGraph& call_graph) {
  // Lambda which returns the next instruction in the callee->caller chain in
  // the call graph. This is the unique instruction which calls the computation
  // containing 'instruction'. If more than one instruction calls the
  // computation containing 'instruction' or no instructions call the
  // computation then nullptr is returned.
  auto next_caller =
      [&call_graph](
          const HloInstruction* instruction) -> const HloInstruction* {
    const CallGraphNode& node = call_graph.GetNode(instruction->parent());
    if (node.caller_callsites().size() != 1) {
      return nullptr;
    }
    return node.caller_callsites()[0].instruction();
  };

  // Iterate through the callee->caller chains and find the earliest common
  // element.
  for (const HloInstruction* a_ancestor = a; a_ancestor != nullptr;
       a_ancestor = next_caller(a_ancestor)) {
    for (const HloInstruction* b_ancestor = b; b_ancestor != nullptr;
         b_ancestor = next_caller(b_ancestor)) {
      if (a_ancestor->parent() == b_ancestor->parent()) {
        return {a_ancestor, b_ancestor};
      }
    }
  }
  return {nullptr, nullptr};
}

}  // namespace

bool HloOrdering::ExecutesBefore(const HloInstruction* a,
                                 const HloInstruction* b) const {
  // 'a' and 'b' may be in different computations. In this case, find the
  // callgraph ancestor instructions which call (potentially transitively) the
  // computations containing 'a' and 'b' and use these ancestor instructions to
  // compare order.
  const HloInstruction* a_ancestor;
  const HloInstruction* b_ancestor;
  std::tie(a_ancestor, b_ancestor) =
      GetNearestCallGraphAncestorsInSameComputation(a, b, *call_graph_);

  if (a_ancestor == nullptr) {
    // Ancestors in a common computation could not be found so consider the
    // instructions 'a' and 'b' to be unordered.
    return false;
  }
  // a_ancestor and b_ancestor must be either both null or both non-null.
  CHECK_NE(b_ancestor, nullptr);
  CHECK_EQ(a_ancestor->parent(), b_ancestor->parent());

  // If the common ancestor is a while instruction there is an additional
  // ordering criteria which may apply. The condition computation is considered
  // to execute before the body computation so if 'a' is in the condition and
  // 'b' is in the body, then 'a' executes before 'b'.
  if (a_ancestor == b_ancestor && a_ancestor->opcode() == HloOpcode::kWhile) {
    const HloComputation* body = a_ancestor->while_body();
    const HloComputation* condition = a_ancestor->while_condition();
    if (call_graph_->InstructionIsNestedIn(a, condition) &&
        call_graph_->InstructionIsNestedIn(b, body)) {
      return true;
    }
  }

  return ExecutesBeforeInSameComputation(a_ancestor, b_ancestor);
}

HloOrderingProto HloOrdering::ToProto() const {
  HloOrderingProto proto;
  for (const auto& computation : module_->computations()) {
    const std::vector<const HloInstruction*>* sequence =
        SequentialOrder(*computation);
    if (sequence != nullptr) {
      HloOrderingProto::SequentialComputation* proto_computation =
          proto.add_sequential_computations();
      proto_computation->set_computation_name(computation->name());
      for (const HloInstruction* instruction : *sequence) {
        *proto_computation->add_instruction_names() = instruction->name();
      }
    }
  }
  return proto;
}

PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : HloOrdering(module) {}

bool PredecessorHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());

  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b'.
  return a != b && predecessors_.at(a->parent())->IsReachable(a, b);
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
          "  %s predecessors:", instruction->name().c_str()));
      for (auto predecessor : all) {
        if (predecessors_.at(computation.get())
                ->IsReachable(predecessor, instruction)) {
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
    if (computation->IsFusionComputation()) {
      continue;
    }
    predecessors_.emplace(computation.get(),
                          computation->ComputeReachability());
  }
}

string DependencyHloOrdering::ToString() const {
  return ToStringHelper("DependencyHloOrdering");
}

SequentialHloOrdering::SequentialHloOrdering(
    const HloModule* module, const HloModuleSequence& module_sequence)
    : HloOrdering(module), module_sequence_(module_sequence) {
  // Create a map from instruction to its order position.
  for (auto computation_order : module_sequence_) {
    const std::vector<const HloInstruction*>& order = computation_order.second;
    for (int i = 0; i < order.size(); ++i) {
      DCHECK_EQ(0, order_position_.count(order[i]));
      order_position_.emplace(order[i], i);
    }
  }
}

bool SequentialHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());
  // If either instruction is not in the order, then 'a' and 'b' are unordered.
  if (order_position_.count(a) == 0 || order_position_.count(b) == 0) {
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}

const std::vector<const HloInstruction*>*
SequentialHloOrdering::SequentialOrder(
    const HloComputation& computation) const {
  auto find_it = module_sequence_.find(&computation);
  return find_it == module_sequence_.end() ? nullptr : &find_it->second;
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

std::ostream& operator<<(
    std::ostream& out,
    const SequentialHloOrdering::HloModuleSequence& module_sequence) {
  for (auto computation_pair : module_sequence) {
    const HloComputation* computation = computation_pair.first;
    const std::vector<const HloInstruction*>& computation_sequence =
        computation_pair.second;
    out << "Computation " << computation->name() << ":\n";
    for (auto* instruction : computation_sequence) {
      out << "  " << instruction->name() << "\n";
    }
  }
  return out;
}

}  // namespace xla
