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

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/heap_simulator.h"
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

StatusOr<int64> MinimumMemoryForSequence(
    const SequentialHloOrdering::HloModuleSequence& module_sequence,
    const LogicalBuffer::SizeFunction& size_function) {
  if (module_sequence.empty()) {
    return 0;
  }

  const HloModule* module = module_sequence.begin()->first->parent();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(module));

  // The absolute minimum memory required for a given sequence of instructions
  // is determined by the sequence of Alloc and Free calls on a simulated heap,
  // ignoring fragmentation. We run the heap simulation on the whole module,
  // rather than summing each computation, since it gives us a better lower
  // bound, by minimizing the liveness of sub-computations.
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result result,
      HeapSimulator::Run(MakeUnique<NoFragmentationStatsHeap>(), *module,
                         module_sequence, *points_to_analysis, size_function));
  return result.heap_size;
}

namespace {

// Class implementing a list scheduler of HLO instructions which produces a
// sequence which minimizes memory usage.
class ListScheduler {
 public:
  // Construct and return a memory-minimizing sequence of HLO instructions
  // containing the given HLO computation.
  static StatusOr<std::vector<const HloInstruction*>> Run(
      const HloComputation& computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const LogicalBuffer::SizeFunction& size_function) {
    ListScheduler scheduler(computation, points_to_analysis, size_function);
    return scheduler.CreateSchedule();
  }

 private:
  // The scheduling priority of an instruction is first the number of bytes
  // freed by scheduling the instruction, and second (tie-breaker) by the number
  // of users. This is represented as a std::pair containing these two values
  // (first element is the bytes freed). std::pair provides the necessary
  // comparison operators.
  using Priority = std::pair<int64, int64>;

  ListScheduler(const HloComputation& computation,
                const TuplePointsToAnalysis& points_to_analysis,
                const LogicalBuffer::SizeFunction& size_function)
      : computation_(computation),
        points_to_analysis_(points_to_analysis),
        size_function_(size_function) {
    // Create a map containing the LogicalBuffer uses for each HLO
    // instruction. An HLO instruction "uses" a LogicalBuffer if the
    // LogicalBuffer is in an operand of the instruction as indicated by
    // points-to analysis.
    for (auto& instruction : computation.instructions()) {
      buffer_uses_.insert(
          {instruction.get(), std::unordered_set<const LogicalBuffer*>()});
      for (auto* operand : instruction->operands()) {
        for (const LogicalBuffer* buffer :
             points_to_analysis.GetBuffersDefinedByInstruction(operand)) {
          buffer_uses_[instruction.get()].insert(buffer);
        }
      }
    }

    // Create map containing the number of unscheduled uses (hlo instructions)
    // of each logical buffer.
    for (auto& instruction : computation.instructions()) {
      for (auto* buffer : points_to_analysis.GetBuffersDefinedByInstruction(
               instruction.get())) {
        unscheduled_use_count_[buffer] = 0;
      }
    }
    for (auto& instruction : computation.instructions()) {
      for (const LogicalBuffer* buffer : buffer_uses_.at(instruction.get())) {
        ++unscheduled_use_count_[buffer];
      }
    }

    // Buffers live out of the computation have an implicit use at the end of
    // the computation.
    for (const LogicalBuffer* live_out_buffer :
         points_to_analysis.GetPointsToSet(computation.root_instruction())
             .CreateFlattenedSet()) {
      ++unscheduled_use_count_[live_out_buffer];
    }
  }

  // Returns whether the memory used by the given buffer should be ignored by
  // the scheduling heuristic.
  bool IgnoreBuffer(const LogicalBuffer& buffer) {
    return buffer.instruction()->opcode() == HloOpcode::kParameter ||
           buffer.instruction()->opcode() == HloOpcode::kConstant;
  }

  // Return the number of bytes freed if the HLO instruction is scheduled.
  int64 BytesFreedIfScheduled(const HloInstruction* instruction) {
    int64 freed_bytes = 0;
    // Sum the total size of the values last used by this instruction.
    for (auto* buffer : buffer_uses_.at(instruction)) {
      if (IgnoreBuffer(*buffer)) {
        continue;
      }
      CHECK_GE(unscheduled_use_count_.at(buffer), 1);
      if (unscheduled_use_count_.at(buffer) == 1) {
        // This is the last use of the logical buffer.
        freed_bytes += size_function_(*buffer);
      }
    }
    // Then subtract the size of the value(s) defined by this instruction.
    for (auto* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (!IgnoreBuffer(*buffer)) {
        freed_bytes -= size_function_(*buffer);
      }
    }
    return freed_bytes;
  }

  // Construct the scheduling priority of the given instruction.
  Priority GetPriority(const HloInstruction* instruction) {
    return {BytesFreedIfScheduled(instruction), instruction->user_count()};
  }

  std::vector<const HloInstruction*> CreateSchedule() {
    std::vector<const HloInstruction*> schedule;

    // Populate the ready list with instructions which have no operands or
    // control predecessors.
    std::unordered_map<const HloInstruction*, int64> unscheduled_pred_count;
    std::list<const HloInstruction*> ready_list;
    for (auto& instruction : computation_.instructions()) {
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (const HloInstruction* user : instruction->users()) {
        unscheduled_pred_count[user]++;
      }
      for (const HloInstruction* succ : instruction->control_successors()) {
        unscheduled_pred_count[succ]++;
      }
    }
    for (auto& instruction : computation_.instructions()) {
      // Instruction with no operands or control predecessors will
      // not be in the map.
      if (unscheduled_pred_count.count(instruction.get()) == 0) {
        ready_list.push_back(instruction.get());
      }
    }

    while (!ready_list.empty()) {
      // Select the highest priority HLO instruction from the ready list.
      auto best_it = ready_list.begin();
      Priority best_priority = GetPriority(*best_it);
      for (auto ready_it = std::next(ready_list.begin());
           ready_it != ready_list.end(); ++ready_it) {
        Priority priority = GetPriority(*ready_it);
        if (priority > best_priority) {
          best_it = ready_it;
          best_priority = priority;
        }
      }

      // Remove the selected instruction from the ready list and add it to the
      // schedule.
      const HloInstruction* best = *best_it;
      ready_list.erase(best_it);
      schedule.push_back(best);
      scheduled_instructions_.insert(best);

      // Update the unscheduled uses of the logical buffers.
      for (const LogicalBuffer* buffer : buffer_uses_.at(best)) {
        CHECK_GT(unscheduled_use_count_.at(buffer), 0);
        --unscheduled_use_count_[buffer];
      }

      // Add new instructions to ready list.
      auto update_pred_count = [&unscheduled_pred_count,
                                &ready_list](HloInstruction* inst) {
        int64 pred_count = --unscheduled_pred_count.at(inst);
        CHECK_GE(pred_count, 0);
        if (pred_count == 0) {
          ready_list.push_back(inst);
        }
      };
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (HloInstruction* user : best->users()) {
        update_pred_count(user);
      }
      for (HloInstruction* succ : best->control_successors()) {
        update_pred_count(succ);
      }
    }
    CHECK_EQ(schedule.size(), computation_.instructions().size());
    CHECK_EQ(scheduled_instructions_.size(),
             computation_.instructions().size());

    return schedule;
  }

  const HloComputation& computation_;
  const TuplePointsToAnalysis& points_to_analysis_;
  const LogicalBuffer::SizeFunction& size_function_;

  // A map containing the LogicalBuffers that each instruction uses.
  std::unordered_map<const HloInstruction*,
                     std::unordered_set<const LogicalBuffer*>>
      buffer_uses_;

  // A map containing the count of unscheduled HLOs which using a particular
  // LogicalBuffer.
  std::unordered_map<const LogicalBuffer*, int64> unscheduled_use_count_;

  // Set of instructions which have been scheduled.
  std::unordered_set<const HloInstruction*> scheduled_instructions_;
};

int64 SumLogicalBufferSizes(const std::vector<const LogicalBuffer*>& buffers,
                            const LogicalBuffer::SizeFunction& size_function) {
  int64 size = 0;
  for (const LogicalBuffer* buffer : buffers) {
    size += size_function(*buffer);
  }
  return size;
}

StatusOr<std::vector<const HloInstruction*>> RunDFSMemoryScheduler(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function) {
  // This ordering is based on DFS post-order, with a heuristic to decide which
  // operand to visit first.  The heuristic is based on 'extra_users', which is
  // simply users-1 for each instruction.  By subtracting 1, we're saying that
  // instructions with no users or a single user don't count; instructions with
  // lots of fan-out will be visited earlier.
  tensorflow::gtl::FlatMap<const HloInstruction*, int64> extra_users;
  tensorflow::gtl::FlatMap<const HloInstruction*, int64> total_sizes;
  for (const HloInstruction* hlo : computation.MakeInstructionPostOrder()) {
    extra_users[hlo] = hlo->users().empty() ? 0 : hlo->users().size() - 1;
    total_sizes[hlo] = SumLogicalBufferSizes(
        points_to_analysis.GetBuffersDefinedByInstruction(hlo), size_function);
    tensorflow::gtl::FlatSet<const HloInstruction*> unique_operands(
        hlo->operands().begin(), hlo->operands().end());
    for (const HloInstruction* operand : unique_operands) {
      extra_users[hlo] += extra_users[operand];
      total_sizes[hlo] += total_sizes[operand];
    }
  }
  CHECK_EQ(extra_users.size(), computation.instructions().size());
  CHECK_EQ(total_sizes.size(), computation.instructions().size());

  // Construct a total order based on DFS post-order, visiting operands in
  // decreasing cumulative extra user order, and next by cumulative size, with a
  // tiebreaker by name for determinism.
  std::vector<const HloInstruction*> sequence;
  FunctionVisitor visitor([&sequence](HloInstruction* hlo) {
    sequence.push_back(hlo);
    return Status::OK();
  });
  TF_RETURN_IF_ERROR(computation.AcceptWithOperandOrder(
      &visitor, [&extra_users, &total_sizes](const HloInstruction* a,
                                             const HloInstruction* b) {
        if (extra_users[a] != extra_users[b]) {
          return extra_users[a] > extra_users[b];
        }
        if (total_sizes[a] != total_sizes[b]) {
          return total_sizes[a] > total_sizes[b];
        }
        return a->name() < b->name();
      }));
  CHECK_EQ(sequence.size(), computation.instructions().size());
  return sequence;
}

StatusOr<int64> MinimumMemoryForComputation(
    const HloComputation& computation,
    const std::vector<const HloInstruction*>& sequence,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function) {
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result result,
      HeapSimulator::Run(MakeUnique<NoFragmentationStatsHeap>(), computation,
                         sequence, points_to_analysis, size_function));
  return result.heap_size;
}

StatusOr<std::vector<const HloInstruction*>> CreateMemoryMinimizingSequence(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function) {
  // We try both a list-scheduler based ordering and a DFS based ordering, and
  // choose whichever returns a lower min-memory, not accounting for
  // fragmentation.
  //
  // Note that this is just a heuristic. One obvious inaccuracy is that the
  // memory required for sub-computations might be different when considered
  // within the caller's context. But it's good enough for now.
  TF_ASSIGN_OR_RETURN(
      std::vector<const HloInstruction*> list_sequence,
      ListScheduler::Run(computation, points_to_analysis, size_function));
  TF_ASSIGN_OR_RETURN(
      const int64 list_memory,
      MinimumMemoryForComputation(computation, list_sequence,
                                  points_to_analysis, size_function));
  VLOG(2) << "Min-memory list sequence: " << list_memory << " bytes";

  TF_ASSIGN_OR_RETURN(
      std::vector<const HloInstruction*> dfs_sequence,
      RunDFSMemoryScheduler(computation, points_to_analysis, size_function));
  TF_ASSIGN_OR_RETURN(
      const int64 dfs_memory,
      MinimumMemoryForComputation(computation, dfs_sequence, points_to_analysis,
                                  size_function));
  VLOG(2) << "Min-memory dfs sequence: " << dfs_memory << " bytes";

  if (list_memory <= dfs_memory) {
    VLOG(2) << "Chose min-memory list sequence: " << list_memory << " bytes";
    return list_sequence;
  } else {
    VLOG(2) << "Chose min-memory dfs sequence: " << dfs_memory << " bytes";
    return dfs_sequence;
  }
}

}  // namespace

StatusOr<SequentialHloOrdering::HloModuleSequence>
CreateMemoryMinimizingSequence(
    const HloModule& module, const LogicalBuffer::SizeFunction& size_function) {
  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(&module));
  for (const auto& computation : module.computations()) {
    TF_ASSIGN_OR_RETURN(sequence[computation.get()],
                        CreateMemoryMinimizingSequence(
                            *computation, *points_to_analysis, size_function));
  }
  return sequence;
}

StatusOr<std::vector<const HloInstruction*>> CreateMemoryMinimizingSequence(
    const HloComputation& computation,
    const LogicalBuffer::SizeFunction& size_function) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(computation.parent()));
  return CreateMemoryMinimizingSequence(computation, *points_to_analysis,
                                        size_function);
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
