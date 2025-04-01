/* Copyright 2016 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_HLO_ORDERING_H_
#define XLA_HLO_ANALYSIS_HLO_ORDERING_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/types.h"

namespace xla {

// Base class for describing a partial ordering of HLO instructions. Used to
// determine live range overlap of HLO instruction output buffers.
class HloOrdering {
 public:
  explicit HloOrdering(const HloModule* module)
      : module_(module), call_graph_(CallGraph::Build(module)) {}
  virtual ~HloOrdering() = default;

  // Specify the ordering constraints between a pair of instructions a and b.
  enum class ExecutionConstraint {
    // Indicate a and b are the same instruction;
    kIsSame,
    // Indicate a runs before b starts;
    kRunBeforeStart,
    // Indicate a runs before b ends but after b starts, e.g., when b is a
    // conditional or while loop;
    kRunBeforeEnd,
    // Only one of a or b runs each time their common ancestor is evaluated,
    // and a is in an earlier branch than b.
    kRunExclusiveBefore,
    // Only one of a or b runs each time, and a is in a later branch than b.
    kRunExclusiveAfter,
    // Indicate a runs after b ends.
    kRunAfter,
    // An order cannot be detrermined as a and b do not have a common ancestor.
    kUnordered,
  };
  // Return the execution constraint between a and b.
  HloOrdering::ExecutionConstraint GetExecutionConstraint(
      const HloInstruction* a, const HloInstruction* b) const;

  // Returns true if instruction 'a' executes before instruction 'b'. This is
  // not reflexive, that is, an instruction does not execute before itself.
  bool ExecutesBefore(const HloInstruction* a, const HloInstruction* b) const;

  // Returns whether the value 'a' is defined before the value 'b' under the
  // given ordering.
  bool IsDefinedBefore(const HloValue& a, const HloValue& b) const;

  // Returns whether the given use is before the given value definition under
  // the given ordering. Set use_is_always_before_def_in_same_instr to false if
  // you want the analysis to always consider a use at an instruction's operand
  // to be strictly before that instructions definition. The configuration needs
  // to be false when result will be used to remove unnecessary copy
  // instructions, due to additional buffer sharing constraints.
  bool UsesBeforeValueDefinition(
      absl::Span<const HloUse* const> uses, const HloValue& value,
      const HloDataflowAnalysis& dataflow,
      bool use_is_always_before_def_in_same_instr = false) const;
  // Returns whether the given values interfere. Two values interfere if they
  // may both be simultaneously live.
  bool MayInterfere(const HloValue& a, const HloValue& b,
                    const HloDataflowAnalysis& dataflow) const;

  // Returns true if the live range of the given value 'a' is strictly before
  // the live range of value 'b' using the given HLO ordering.
  bool LiveRangeStrictlyBefore(
      const HloValue& a, const HloValue& b, const HloDataflowAnalysis& dataflow,
      bool use_is_always_before_def_in_same_instr = false) const;

  // Returns the sequential instruction order for the given computation, or
  // nullptr if the computation does not have a sequential ordering.
  virtual const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const = 0;

  // Return the call graph of the module used to compute ordering.
  const CallGraph& call_graph() const { return *call_graph_; }

  virtual std::string ToString() const = 0;

 protected:
  // Returns true if instruction 'a' executes before instruction 'b'.
  // Precondition: 'a' and 'b' are in the same computation.
  //
  // Derived classes should implement this method for determining order of
  // instructions in the same computation. ExecutesBefore() analyzes the
  // callgraph and uses this method to determine ordering of instructions in
  // different computations.
  virtual bool ExecutesBeforeInSameComputation(
      const HloInstruction* a, const HloInstruction* b) const = 0;

  const HloModule* module_;

  std::unique_ptr<CallGraph> call_graph_;
};

// Base class for partial orderings implemented by a map of predecessors for
// each instruction. Subclasses should fill in predecessors_.
class PredecessorHloOrdering : public HloOrdering {
 public:
  ~PredecessorHloOrdering() override = default;

  // Returns nullptr indicating the computation does not have a sequential
  // ordering.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override {
    return nullptr;
  }

  HloReachabilityMap& reachability_map(const HloComputation* computation) {
    return *predecessors_.at(computation);
  }
  const HloReachabilityMap& reachability_map(
      const HloComputation* computation) const {
    return *predecessors_.at(computation);
  }

 protected:
  explicit PredecessorHloOrdering(const HloModule* module);
  std::string ToStringHelper(const std::string& name) const;

  bool ExecutesBeforeInSameComputation(const HloInstruction* a,
                                       const HloInstruction* b) const override;

  // For each computation in the module, this is the set of the instruction's
  // predecessors. An instruction is an element of its own predecessor set.
  //
  // Subclasses should fill this in to define the desired ordering.
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloReachabilityMap>>
      predecessors_;
};

// An HLO ordering based on data dependencies in the HLO graph. In this partial
// order, instruction A executes before instruction B only if there is a path
// from A to B in the HLO graph. For example, given the following graph:
/*
          param
         /     \
      negate   exp
          \    /
           add
*/
// DependencyHloOrdering gives the following executes-before relations:
//   param executes before negate, exp, and add
//   negate executes before add
//   exp executes before add
//   add executes before nothing
// negate and exp are not ordered because the dependencies allow either to
// execute before the other (or in parallel). DependencyHloOrdering ordering
// allows maximum parallelism and enables any execution order which satisfies
// data dependencies. This requires pessimistic assumptions about buffer live
// ranges and can result in more memory used than more constrained orderings.
class DependencyHloOrdering : public PredecessorHloOrdering {
 public:
  explicit DependencyHloOrdering(const HloModule* module);
  ~DependencyHloOrdering() override = default;

  std::string ToString() const override;
};

// An HLO ordering based on a total order of instructions in each computation.
// The computation total order is a sequencing of all of its instructions in
// the computation (eg, {inst0, inst1, inst2,...}) as in single-threaded
// execution. For example, given the following HLO graph:
/*
          param
         /     \
      negate   exp
          \    /
           add
*/
// and the following sequence:
//
//  {param, negate, exp, add}
//
// SequentialHloOrdering gives the following executes-before relations:
//   param executes before negate, exp, and add
//   negate executes before exp and add
//   exp executes before add
//   add executes before nothing
// This is more constrained than DependencyHloOrdering in this example because
// negate and exp are ordered (negate before exp). This enables param to share
// the same buffer as exp (param buffer is dead after exp). Generally, this
// ordering enables more buffer sharing (reduced memory usage) because buffer
// interference is reduced relative to DependencyHloOrdering.
class SequentialHloOrdering : public HloOrdering {
 public:
  explicit SequentialHloOrdering(const HloSchedule& schedule);
  explicit SequentialHloOrdering(HloSchedule&& schedule);
  ~SequentialHloOrdering() override = default;

  // Returns the sequential instruction order for the given computation.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override;

  std::string ToString() const override;

 protected:
  void Initialize();

  bool ExecutesBeforeInSameComputation(const HloInstruction* a,
                                       const HloInstruction* b) const override;

  const HloSchedule schedule_;

  // The position of every instruction in the HLO module in its respective
  // computation sequence (a value of zero indicates the instruction is first in
  // the sequence, etc). Instructions from all computations are contained in
  // this map so more than one instruction may have the same position
  // value. This is not a problem because ExecutesBefore also verifies
  // instructions are in the same computation.
  absl::flat_hash_map<const HloInstruction*, int> order_position_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_ORDERING_H_
