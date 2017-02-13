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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {

// Abstract base class for describing a partial ordering of HLO
// instructions. Used to determine live range overlap of HLO instruction output
// buffers.
class HloOrdering {
 public:
  HloOrdering() = default;
  virtual ~HloOrdering() = default;

  // Returns true if instruction 'a' executes before instruction 'b'. This is
  // not reflexive, that is, an instruction does not execute before itself.
  virtual bool ExecutesBefore(const HloInstruction* a,
                              const HloInstruction* b) const = 0;
  virtual string ToString() const = 0;
};

// Base class for partial orderings implemented by a map of strict predecessors
// for each instruction. Subclasses should fill in strict_predecessors_.
class PredecessorHloOrdering : public HloOrdering {
 public:
  ~PredecessorHloOrdering() override = default;

  // Returns true if instruction 'a' executes before instruction 'b'.
  // Instructions in different computations are not ordered.
  bool ExecutesBefore(const HloInstruction* a,
                      const HloInstruction* b) const override;

 protected:
  explicit PredecessorHloOrdering(const HloModule* module);
  string ToStringHelper(const string& name) const;

  const HloModule* module_;

  // For each each computation in the module, this is the set of the
  // instruction's strict predecessors. An instruction is not an element of its
  // own strict predecessor set.
  //
  // Subclasses should fill this in to define the desired ordering.
  tensorflow::gtl::FlatMap<const HloComputation*,
                           std::unique_ptr<HloComputation::ReachabilityMap>>
      strict_predecessors_;
};

// An HLO ordering based on data dependencies in the HLO graph. In this partial
// order, instruction A executes before instruction B only if there is a path
// from A to B in the HLO graph. For example, given the following graph:
//
//        param
//       /     \
//    negate   exp
//        \    /
//         add
//
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

  string ToString() const override;
};

// An HLO ordering based on a total order of instructions in each computation.
// The computation total order is a sequencing of all of its instructions in
// the computation (eg, {inst0, inst1, inst2,...}) as in single-threaded
// execution. For example, given the following HLO graph:
//
//        param
//       /     \
//    negate   exp
//        \    /
//         add
//
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
  // A sequence of instructions for each computation in the module.
  using HloModuleSequence =
      tensorflow::gtl::FlatMap<const HloComputation*,
                               std::vector<const HloInstruction*>>;

  SequentialHloOrdering(const HloModule* module,
                        const HloModuleSequence& module_sequence);
  ~SequentialHloOrdering() override = default;

  // Instruction 'a' executes before 'b' if 'a' appears before 'b' in the
  // instruction sequence for the computation. Instructions in different
  // computations are unordered.
  bool ExecutesBefore(const HloInstruction* a,
                      const HloInstruction* b) const override;
  string ToString() const override;

 protected:
  const HloModule* module_;

  // The position of every instruction in the HLO module in its respective
  // computation sequence (a value of zero indicates the instruction is first in
  // the sequence, etc). Instructions from all computations are contained in
  // this map so more than one instruction may have the same position
  // value. This is not a problem because ExecutesBefore also verifies
  // instructions are in the same computation.
  tensorflow::gtl::FlatMap<const HloInstruction*, int> order_position_;
};

// Returns an HloModuleSequence which seeks to minimize the memory required for
// the computation. size_function is the function returning the number of bytes
// required for a LogicalBuffer.
StatusOr<SequentialHloOrdering::HloModuleSequence>
CreateMemoryMinimizingSequence(
    const HloModule& module, const LogicalBuffer::SizeFunction& size_function);

std::ostream& operator<<(
    std::ostream& out,
    const SequentialHloOrdering::HloModuleSequence& module_sequence);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_
