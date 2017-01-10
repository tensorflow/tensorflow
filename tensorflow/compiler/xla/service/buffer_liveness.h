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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_

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

// Class which computes liveness of the output buffers of HLOs and their
// interference.
class BufferLiveness {
 public:
  // Constructs a buffer liveness object for the given module assuming the given
  // HLO instruction ordering.
  static StatusOr<std::unique_ptr<BufferLiveness>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering);

  // Returns true if the live range of the buffer containing the output of 'a'
  // may overlap with the live range of the buffer of 'b'. If instruction 'a'
  // interferes with instruction 'b' then they cannot share the same buffer.
  bool MayInterfere(const LogicalBuffer& a, const LogicalBuffer& b) const;

  // Returns true if the buffer for the given instruction may be live out of the
  // module. That is, the instruction's buffer may be included in the output of
  // the entry computation.
  bool MaybeLiveOut(const LogicalBuffer& buffer) const;

  // Returns the underlying points-to analysis used for this liveness analysis.
  const TuplePointsToAnalysis& points_to_analysis() const {
    return *points_to_analysis_;
  }

  string ToString() const;

 private:
  explicit BufferLiveness(const HloModule* module,
                          std::unique_ptr<HloOrdering> hlo_ordering)
      : module_(module), hlo_ordering_(std::move(hlo_ordering)) {}

  // Perform buffer liveness analysis. This method must be called prior to
  // MayInterfere or MaybeLiveOut.
  tensorflow::Status Analyze();

  // Returns true if the live range of the buffer of 'a' is strictly before the
  // live range of the buffer of 'b' (they do not overlap).
  bool live_range_strictly_before(const LogicalBuffer& a,
                                  const LogicalBuffer& b) const;

  const HloModule* module_;
  std::unique_ptr<HloOrdering> hlo_ordering_;

  // Set of LogicalBuffers which are aliased in the output of other
  // instructions. For example, a LogicalBuffer which is inserted into a tuple
  // is considered to be aliased and will be in this set.
  tensorflow::gtl::FlatSet<const LogicalBuffer*> aliased_buffers_;

  // LogicalBuffers that may be live out of the entry computation.
  tensorflow::gtl::FlatSet<const LogicalBuffer*> maybe_live_out_buffers_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_LIVENESS_H_
