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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_

#include <stddef.h>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/logical_buffer_analysis.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/compactptrset.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A class describing the source(s) of the Buffer(s) contained in the output of
// a particular HLO instruction. The structure of PointsToSet mirrors the
// structure of the instruction's shape, which may be an arbitrary tree (eg, a
// nested tuple). Each node in this tree corresponds to a single buffer in the
// instruction's output and contains the set of Buffers which might define
// the corresponding buffer.
class PointsToSet {
 public:
  // Construct our ShapeTree with a pointer rather than a reference to a Shape
  // because this is very hot code, and copying (and then destroying) all these
  // Shapes is slow.
  explicit PointsToSet(const Shape* shape) : tree_(shape) {}

  // Returns true if any points-to sets for any subshape element is not a
  // singleton.
  bool IsAmbiguous() const;

  // Returns true if no LogicalBuffer appears in more than one points-to set of
  // the shape nodes.
  bool IsDistinct() const;

  // Returns the total number of different LogicalBuffers contained in this
  // object. This is equal to CreateFlattenedSet().size().
  size_t size() const;

  // Creates a set containing the union of all LogicalBuffers contained in the
  // PointsToSet.
  using BufferSet = tensorflow::gtl::CompactPointerSet<const LogicalBuffer*>;
  BufferSet CreateFlattenedSet() const;

  // Returns true if the given buffer is in the points-to set at the given
  // index.
  bool ContainsBufferAtIndex(const LogicalBuffer& buffer,
                             const ShapeIndex& index) const;

  // Returns true if the given buffer is in the points-to set at any index.
  bool ContainsBuffer(const LogicalBuffer& buffer) const;

  // Adds the given buffer to the points-to set at the given index. This is a
  // nop if the buffer already is in the set at that index.
  void AddPointedToBuffer(const LogicalBuffer& buffer, const ShapeIndex& index);

  // For the subshape at the given index (where index is defined as in
  // ShapeUtil::GetSubshape) this method returns the set of HLO instructions
  // which may produce the tuple subshape at that index. For example, given:
  //
  // %tuple1 = tuple(...)
  // %tuple2 = tuple(...)
  // %select = select(%tuple1, %tuple2)
  // %nested_tuple = tuple(%select, %tuple1)
  //
  // These are the values for tuple_sources() for the PointsToSet of
  // %nested_tuple:
  //
  // tuple_sources({}) = {%nested_tuple}
  // tuple_sources({0}) = {%tuple1, %tuple2}
  // tuple_sources({1}) = {%tuple1}
  //
  // tuple_sources() at the index of an array shape (not a tuple) returns the
  // empty set. The instructions in the set returned by tuple_sources
  // necessarily are either Tuple instructions, constants, or parameters.
  using SourceSet = tensorflow::gtl::CompactPointerSet<HloInstruction*>;
  const SourceSet& tuple_sources(const ShapeIndex& index) const;

  // Add a tuple source instruction for the given index.
  void add_tuple_source(const ShapeIndex& index, HloInstruction* tuple);

  using BufferList = tensorflow::gtl::InlinedVector<const LogicalBuffer*, 1>;

  // Return the list of logical buffers for the subshape at index.
  const BufferList& element(const ShapeIndex& index) const {
    return tree_.element(index).buffers;
  }
  BufferList* mutable_element(const ShapeIndex& index) {
    return &tree_.mutable_element(index)->buffers;
  }

  // Call fn(index, buflist) for every subshape index.
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
  Status ForEachElementWithStatus(const Fn& fn) const {
    return tree_.ForEachElementWithStatus(
        [&fn](const ShapeIndex& index, const Elem& elem) {
          return fn(index, elem.buffers);
        });
  }

 private:
  struct Elem {
    BufferList buffers;
    SourceSet tuple_sources;
  };
  ShapeTree<Elem> tree_;

  // PointsToSet contains references (const LogicalBuffer*) to elements within
  // TuplePointsToAnalysis, so disable copying.
  TF_DISALLOW_COPY_AND_ASSIGN(PointsToSet);
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

  string ToString() const;

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
  static StatusOr<std::unique_ptr<TuplePointsToAnalysis>> Run(
      const HloModule* module);

  // Return the points-to set of an instruction. This describes the potential
  // sources of each buffer in the instruction's output.
  const PointsToSet& GetPointsToSet(
      const HloInstruction* hlo_instruction) const;

  // Returns the logical buffer with the given ID.
  const LogicalBuffer& GetBuffer(LogicalBuffer::Id id) const;

  // Returns the buffer defined at the given instruction and index. An error is
  // returned if no buffer is defined at that point.
  StatusOr<const LogicalBuffer*> GetBufferDefinedAt(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Return a (possibly empty) vector containing all BufferAliases of the given
  // logical buffer The buffer alias set is the inverse of the points-to set.
  // That is, LogicalBuffer B is in the points-to set of instruction I at index
  // N iff instruction I, index N is a BufferAlias of B.
  using BufferAliasVector = tensorflow::gtl::InlinedVector<BufferAlias, 1>;
  const BufferAliasVector& GetBufferAliases(const LogicalBuffer& buffer) const;

  // Returns the number of logical buffers in the module
  LogicalBuffer::Id num_logical_buffers() const {
    return logical_buffer_analysis_->num_logical_buffers();
  }

  // Return a the logical buffer with id "id" in the module. Iteration
  // over all logical buffers is usually done with something like:
  //
  // for (LogicalBuffer:Id id = 0; id < points_to.num_logical_buffers(); id++){
  //   const auto& buffer = points_to.logical_buffer(id);
  //   ... do something with buffer ...
  // }
  LogicalBuffer& logical_buffer(LogicalBuffer::Id id) const {
    return logical_buffer_analysis_->GetBuffer(id);
  }

  // Returns a vector of buffers that the instruction produces. Most
  // instructions produce a single buffer (the top-level buffer), some produce
  // no buffers (eg bitcast), and some produce more than one buffer (eg,
  // tuple-shaped parameters).
  using BufferDefinitionVector =
      tensorflow::gtl::InlinedVector<const LogicalBuffer*, 1>;
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
  Status VerifyBuffer(const LogicalBuffer& buffer) const;

  Status DefaultAction(HloInstruction* hlo_instruction) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleDomain(HloInstruction* domain) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;

  string ToString() const;

  // Returns true if 'user' cannot possibly use the buffer at 'index' in
  // 'operand'. Returns false otherwise.
  //
  // REQUIRES: 'operand' is an operand of 'user'.
  bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                               const ShapeIndex& index,
                               const HloInstruction* user) const;

  // Returns true if 'user' (at 'user_index') can share a buffer with its
  // operand 'operand' (at 'operand_index'). Returns false otherwise.
  //
  // REQUIRES: 'operand' is an operand of 'user'.
  bool CanShareOperandBufferWithUser(HloInstruction* operand,
                                     const ShapeIndex& operand_index,
                                     HloInstruction* user,
                                     const ShapeIndex& user_index) const;

 private:
  explicit TuplePointsToAnalysis(
      const HloModule* module,
      std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis)
      : module_(module),
        logical_buffer_analysis_(std::move(logical_buffer_analysis)) {}

  // Perform the analysis. Should be called immediately after constructing the
  // object and before calling GetPointsToSet.
  Status Analyze();

  // Populates instruction-defined buffers and aliases for each instruction
  // in 'instructions'.
  Status PopulateDefinedBuffersAndAliases(const decltype(
      std::declval<HloComputation>().instructions())& instructions);

  // Creates an empty PointsToSet in the points_to_ map for the given
  // instruction.
  PointsToSet& CreateEmptyPointsToSet(const HloInstruction* instruction);

  // Creates a PointsToSet in the points_to_ map for 'instruction' which is a
  // copy of the existing PointsToSet for 'src'.
  PointsToSet& CreateCopiedPointsToSet(const HloInstruction* instruction,
                                       const HloInstruction* src);

  // Adds the buffers defined by the given instruction to the given vector.
  Status GatherBuffersDefinedByInstruction(const HloInstruction* instruction,
                                           BufferDefinitionVector* buffers);

  // Print points-to set for 'instruction' to 'output'.
  void InstructionToString(const HloInstruction* instruction,
                           string* output) const;

  // Information kept per instruction
  struct PerInstruction {
    std::unique_ptr<PointsToSet> points_to_set;
    // Empircally, ~92% of instructions have 1
    // instruction_defined_buffer, and 99% have 0 or 1
    BufferDefinitionVector instruction_defined_buffers;
  };

  const PerInstruction* PerInst(const HloInstruction* inst) const {
    int id = inst->unique_id();
    DCHECK_GE(id, 0);
    DCHECK_LT(id, per_instruction_.size());
    return &per_instruction_[id];
  }
  PerInstruction* PerInst(const HloInstruction* inst) {
    int id = inst->unique_id();
    DCHECK_GE(id, 0);
    DCHECK_LT(id, per_instruction_.size());
    return &per_instruction_[id];
  }

  std::vector<std::pair<HloInstruction*, int64>> GetAllUsesOfInstructionAtIndex(
      HloInstruction* instruction, const ShapeIndex& index) const;
  bool HasUniqueFusedUseOfOperandAt(HloInstruction* operand,
                                    const ShapeIndex& operand_index,
                                    HloInstruction* fusion,
                                    const int64 use_operand_index) const;

  // The module this analysis is performed on.
  const HloModule* module_;

  // The logical buffers for this module.
  const std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis_;

  // A map from instruction->unique_id() to
  std::vector<PerInstruction> per_instruction_;

  // A map from LogicalBuffer->id() to alias information about that logical
  // buffer
  std::vector<BufferAliasVector> logical_buffer_aliases_;

  TF_DISALLOW_COPY_AND_ASSIGN(TuplePointsToAnalysis);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
