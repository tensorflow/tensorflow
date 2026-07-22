/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_TUPLE_POINTS_TO_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_TUPLE_POINTS_TO_ANALYSIS_H_

#include <stddef.h>

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/analysis/logical_buffer_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/compactptrset.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A PointsToSet describes the source(s) of the LogicalBuffer(s) contained in
// the output of a particular HLO instruction.
//
// In XLA, HLO instructions do not necessarily allocate new memory; some may
// forward or alias existing buffers (e.g., GetTupleElement, Bitcast). To track
// who defines the buffers, the compiler performs points-to analysis.
//
// Since an instruction's output can have a nested or structured shape (like a
// nested tuple), PointsToSet internally uses a ShapeTree. The structure of this
// tree mirrors the Shape of the instruction.
//
// Each node in the tree (corresponding to a subshape index/path in the output
// Shape) contains:
// 1) A list of LogicalBuffers (`BufferList`) that can potentially define the
//    buffer at that subshape index.
// 2) A set of HLO instructions (`SourceSet`) representing the instructions that
//    define/originate the tuple shape at that index (i.e. 'tuple sources').
//
// Thread Safety: PointsToSet is not thread-safe due to internal mutation of
// the ShapeTree without locking.
class PointsToSet {
 public:
  // Constructs a PointsToSet for the given shape.
  //
  // Note: Accepts a `const Shape*` instead of a copy or reference. This is an
  // optimization in hot execution paths to avoid copying/creating many
  // temporary `Shape` objects.
  explicit PointsToSet(const Shape* shape) : tree_(shape) {}

  // Returns true if any of the points-to sets (at any subshape index) contains
  // more than one LogicalBuffer.
  //
  // An ambiguous set indicates that the compiler cannot statically determine
  // a single unique source buffer for that part of the output.
  bool IsAmbiguous() const;

  // Returns true if no LogicalBuffer appears at more than one subshape index
  // of the instruction's shape.
  //
  // A distinct points-to set guarantees that different components/paths of the
  // output (e.g., elements in a tuple) do not share or alias the same logical
  // buffers.
  bool IsDistinct() const;

  // Returns the total number of unique LogicalBuffers across all subshape
  // indices in this PointsToSet. This is equivalent to
  // `CreateFlattenedSet().size()`.
  size_t size() const;

  // Returns the union of all LogicalBuffers contained in all subshape indices
  // of this PointsToSet.
  using BufferSet = tsl::gtl::CompactPointerSet<const LogicalBuffer*>;
  BufferSet CreateFlattenedSet() const;

  // Returns true if the points-to set at the given `ShapeIndex` contains the
  // specified `LogicalBuffer`.
  bool ContainsBufferAtIndex(const LogicalBuffer& buffer,
                             const ShapeIndex& index) const;

  // Returns true if the specified `LogicalBuffer` is present in the points-to
  // set at any `ShapeIndex` in this PointsToSet.
  bool ContainsBuffer(const LogicalBuffer& buffer) const;

  // Adds the given `LogicalBuffer` to the points-to set at the specified
  // `ShapeIndex`. This is a no-op if the buffer is already present at that
  // index.
  void AddPointedToBuffer(const LogicalBuffer& buffer, const ShapeIndex& index);

  using SourceSet = tsl::gtl::CompactPointerSet<HloInstruction*>;

  // For the subshape at the given index (which must be a tuple shape), returns
  // the set of HLO instructions that may define/originate the tuple structure
  // at that index.
  //
  // This is used to track the allocation/creation points of tuple shapes,
  // which helps analyze how tuple buffers are shared.
  //
  // Example:
  // Given:
  //   %tuple1 = tuple(...)
  //   %tuple2 = tuple(...)
  //   %cond = conditional(...) // selects between %tuple1 and %tuple2
  //   %nested_tuple = tuple(%cond, %tuple1)
  //
  // In `nested_tuple`'s PointsToSet:
  //   tuple_sources({})   -> {%nested_tuple}
  //   tuple_sources({0})  -> {%cond}
  //   tuple_sources({1})  -> {%tuple1}
  //
  // If the subshape at the index is an array (not a tuple), this returns an
  // empty set. The instructions in the set are typically Tuple instructions,
  // constants, parameters, or control flow instructions (e.g., While,
  // Conditional).
  //
  // Note: One cannot assume that `tuple_sources` is a subset of the defining
  // instructions of the `buffers` at the same index. For example, a shallow
  // copy of a tuple `%copy = copy(%original_tuple)` allocates a new top-level
  // buffer defined by `%copy` at index `{}`, which updates `buffers({})` to
  // contain `LogicalBuffer(%copy, {})`. However, the tuple structure origin is
  // unchanged, so `tuple_sources({})` remains `{%original_tuple}`.
  const SourceSet& tuple_sources(const ShapeIndex& index) const;

  // Adds the given HLO instruction as a tuple source for the subshape index.
  void add_tuple_source(const ShapeIndex& index, HloInstruction* tuple);

  using BufferList = absl::InlinedVector<const LogicalBuffer*, 1>;

  // Returns the `BufferList` (list of `LogicalBuffer`s) for the subshape at the
  // given index.
  const BufferList& element(const ShapeIndex& index) const {
    return tree_.element(index).buffers;
  }
  BufferList* mutable_element(const ShapeIndex& index) {
    return &tree_.mutable_element(index)->buffers;
  }

  // Invokes `fn` (a callable with signature compatible with
  // `void(const ShapeIndex&, const BufferList&)`) on each element in the
  // points-to set.
  template <typename Fn>
  void ForEachElement(const Fn& fn) const {
    tree_.ForEachElement([&fn](const ShapeIndex& index, const Elem& elem) {
      fn(index, elem.buffers);
    });
  }
  template <typename Fn>
  void ForEachMutableElement(const Fn& fn) {
    tree_.ForEachMutableElement([&fn](const ShapeIndex& index, Elem* elem) {
      fn(index, &elem->buffers);
    });
  }
  template <typename Fn>
  absl::Status ForEachMutableElementWithStatus(const Fn& fn) {
    return tree_.ForEachMutableElementWithStatus(
        [&fn](const ShapeIndex& index, Elem* elem) {
          return fn(index, &elem->buffers);
        });
  }
  template <typename Fn>
  absl::Status ForEachElementWithStatus(const Fn& fn) const {
    return tree_.ForEachElementWithStatus(
        [&fn](const ShapeIndex& index, const Elem& elem) {
          return fn(index, elem.buffers);
        });
  }

  // Returns a string representation of the PointsToSet for debugging.
  std::string ToString() const;

 private:
  struct Elem {
    // The list of LogicalBuffers that may potentially define/provide the value
    // at this subshape.
    BufferList buffers;
    // The set of HLO instructions that define/originate the tuple shape at
    // this subshape (populated only if this subshape is tuple-shaped). Note
    // that this set is not necessarily a subset of the instructions defining
    // the buffers in `buffers` (e.g., in the case of a shallow copy).
    SourceSet tuple_sources;
  };
  ShapeTree<Elem> tree_;

  // PointsToSet contains references (const LogicalBuffer*) to elements within
  // TuplePointsToAnalysis, so disable copying.
  PointsToSet(const PointsToSet&) = delete;
  PointsToSet& operator=(const PointsToSet&) = delete;
};

// This class describes a particular subshape in a computation (instruction and
// shape index) and the logical buffer which may be a source of the subshape
// value.
class BufferAlias {
 public:
  BufferAlias(HloInstruction* instruction, const ShapeIndex& index)
      : instruction_(instruction), index_(index) {}

  // Return the instruction/index of the subshape.
  HloInstruction* instruction() const { return instruction_; }
  const ShapeIndex& index() const { return index_; }

  bool operator==(const BufferAlias& other) const {
    return instruction_ == other.instruction_ && index_ == other.index_;
  }
  bool operator!=(const BufferAlias& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  HloInstruction* instruction_;
  ShapeIndex index_;
};

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias);

// DFS visitor that performs tuple points-to analysis. This analysis determines
// the potential sources of each buffer in each instruction's output.
class TuplePointsToAnalysis : public DfsHloVisitorWithDefault {
 public:
  // Runs points-to analysis on 'module'.
  static absl::StatusOr<std::unique_ptr<TuplePointsToAnalysis>> Run(
      const HloModule* module);

  // Return the points-to set of an instruction. This describes the potential
  // sources of each buffer in the instruction's output.
  const PointsToSet& GetPointsToSet(
      const HloInstruction* hlo_instruction) const;

  // Returns the logical buffer with the given ID.
  const LogicalBuffer& GetBuffer(LogicalBuffer::Id id) const;

  // Returns the buffer defined at the given instruction and index. An error is
  // returned if no buffer is defined at that point.
  absl::StatusOr<const LogicalBuffer*> GetBufferDefinedAt(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Return a (possibly empty) vector containing all BufferAliases of the given
  // logical buffer The buffer alias set is the inverse of the points-to set.
  // That is, LogicalBuffer B is in the points-to set of instruction I at index
  // N iff instruction I, index N is a BufferAlias of B.
  using BufferAliasVector = absl::InlinedVector<BufferAlias, 1>;
  const BufferAliasVector& GetBufferAliases(const LogicalBuffer& buffer) const;

  // Returns the number of logical buffers in the module
  LogicalBuffer::Id num_logical_buffers() const {
    return logical_buffer_analysis_->num_logical_buffers();
  }

  // Returns a vector of buffers that the instruction produces. Most
  // instructions produce a single buffer (the top-level buffer), some produce
  // no buffers (eg bitcast), and some produce more than one buffer (eg,
  // tuple-shaped parameters).
  using BufferDefinitionVector = absl::InlinedVector<const LogicalBuffer*, 1>;
  const BufferDefinitionVector& GetBuffersDefinedByInstruction(
      const HloInstruction* instruction) const;

  // Returns true if the given instruction defines a buffer at the given index.
  bool InstructionDefinesBufferAtIndex(const HloInstruction* instruction,
                                       const ShapeIndex& index) const;

  // Returns an OK status if the given buffer is defined by instruction
  // 'buffer.instruction()' at index 'buffer.index()' and if the given buffer
  // matches the TuplePointsToAnalysis' LogicalBuffer with 'buffer.id'. Returns
  // an FailedPrecondition error status otherwise. An example of a LogicalBuffer
  // which is not defined is a tuple element in a Tuple instruction. In this
  // case, the Tuple instruction does not define the LogicalBuffer, rather that
  // index aliases one of its operands.
  absl::Status VerifyBuffer(const LogicalBuffer& buffer) const;

  absl::Status DefaultAction(HloInstruction* hlo_instruction) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleAsyncStart(HloInstruction* async_start) override;
  absl::Status HandleAsyncUpdate(HloInstruction* async_update) override;
  absl::Status HandleAsyncDone(HloInstruction* async_done) override;
  absl::Status HandleBitcast(HloInstruction* bitcast) override;
  absl::Status HandleDomain(HloInstruction* domain) override;
  absl::Status HandleCopy(HloInstruction* copy) override;
  absl::Status HandleCopyStart(HloInstruction* copy_start) override;
  absl::Status HandleCopyDone(HloInstruction* copy_done) override;
  absl::Status HandleRecvDone(HloInstruction* recv_done) override;
  absl::Status HandleSend(HloInstruction* send) override;
  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;
  absl::Status HandleCustomCall(HloInstruction* custom_call) override;
  absl::Status HandleFusion(HloInstruction* fusion) override;
  absl::Status HandleOptimizationBarrier(HloInstruction* barrier) override;

  std::string ToString() const;

  // Returns true if 'user' cannot possibly use the buffer at 'index' in
  // 'operand'. Returns false otherwise.
  //
  // REQUIRES: 'operand' is an operand of 'user'.
  bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                               const ShapeIndex& index,
                               const HloInstruction* user) const;

 private:
  explicit TuplePointsToAnalysis(
      const HloModule* module,
      std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis)
      : module_(module),
        logical_buffer_analysis_(std::move(logical_buffer_analysis)) {}

  // Perform the analysis. Should be called immediately after constructing the
  // object and before calling GetPointsToSet.
  absl::Status Analyze();

  // Populates instruction-defined buffers and aliases for each instruction
  // in 'instructions'.
  absl::Status PopulateDefinedBuffersAndAliases(
      const decltype(std::declval<HloComputation>()
                         .instructions())& instructions);

  // Creates an empty PointsToSet in the points_to_ map for the given
  // instruction.
  PointsToSet& CreateEmptyPointsToSet(const HloInstruction* instruction);

  // Creates a PointsToSet in the points_to_ map for 'instruction' which is a
  // copy of the existing PointsToSet for 'src'.
  PointsToSet& CreateCopiedPointsToSet(const HloInstruction* instruction,
                                       const HloInstruction* src);

  // Adds the buffers defined by the given instruction to the given vector.
  absl::Status GatherBuffersDefinedByInstruction(
      const HloInstruction* instruction, BufferDefinitionVector* buffers);

  // Applies deferred aliases from async-start to the current instruction's
  // points-to set.
  void ApplyDeferredAliases(HloInstruction* current_instruction,
                            PointsToSet& points_to_set);

  // Propagates points-to sets for an instruction that aggregates its operands
  // into an output tuple structure.
  absl::Status ConstructPointsToSetByAggregatingOperands(
      HloInstruction* instruction);

  // Print points-to set for 'instruction' to 'output'.
  void InstructionToString(const HloInstruction* instruction,
                           std::string* output) const;

  // Information kept per instruction
  struct PerInstruction {
    std::unique_ptr<PointsToSet> points_to_set;
    // Buffers defined by this instruction.
    // Empirically, ~92% of instructions have 1
    // instruction_defined_buffer, and 99% have 0 or 1
    BufferDefinitionVector instruction_defined_buffers;
  };

  const PerInstruction* PerInst(const HloInstruction* inst) const {
    int64_t id = inst->unique_id();
    DCHECK_GE(id, 0);
    auto iter = per_instruction_.find(id);
    if (iter == per_instruction_.end()) {
      LOG(FATAL) << "Expected per-instruction information to already exist";
    }
    return iter->second.get();
  }
  PerInstruction* PerInst(const HloInstruction* inst) {
    int64_t id = inst->unique_id();
    DCHECK_GE(id, 0);
    auto iter = per_instruction_.find(id);
    if (iter == per_instruction_.end()) {
      return per_instruction_.emplace(id, std::make_unique<PerInstruction>())
          .first->second.get();
    }
    return iter->second.get();
  }

  // The module this analysis is performed on.
  const HloModule* module_;

  // The logical buffers for this module.
  const std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis_;

  // A map from instruction->unique_id() to
  absl::flat_hash_map<int64_t, std::unique_ptr<PerInstruction>>
      per_instruction_;

  // A map from LogicalBuffer->id() to alias information about that logical
  // buffer
  std::vector<BufferAliasVector> logical_buffer_aliases_;

  TuplePointsToAnalysis(const TuplePointsToAnalysis&) = delete;
  TuplePointsToAnalysis& operator=(const TuplePointsToAnalysis&) = delete;
  // Whether to alias buffers connected by dataflow relations. This aliasing
  // relation should not be recognized if copies can be inserted to break up
  // the dataflow relation.
  const bool alias_buffer_across_dataflow_ = false;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_TUPLE_POINTS_TO_ANALYSIS_H_
