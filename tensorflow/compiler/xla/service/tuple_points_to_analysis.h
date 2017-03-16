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
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A class describing the source(s) of the Buffer(s) contained in the output of
// a particular HLO instruction. The structure of PointsToSet mirrors the
// structure of the instruction's shape which may be an arbitrary tree (eg, a
// nested tuple). Each node in this tree corresponds to a single buffer in the
// instruction's output and contains the set of Buffers which might define
// the corresponding buffer.
class PointsToSet : public ShapeTree<std::vector<const LogicalBuffer*>> {
 public:
  explicit PointsToSet(const Shape& shape)
      : ShapeTree<std::vector<const LogicalBuffer*>>(shape),
        tuple_sources_(shape) {}

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
  tensorflow::gtl::FlatSet<const LogicalBuffer*> CreateFlattenedSet() const;

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
  const std::set<HloInstruction*>& tuple_sources(const ShapeIndex& index) const;

  // Add a tuple source instruction for the given index.
  void add_tuple_source(const ShapeIndex& index, HloInstruction* tuple);

 private:
  ShapeTree<std::set<HloInstruction*>> tuple_sources_;

  // PointsToSet contains references (const LogicalBuffer*) to elements within
  // TuplePointsToAnalysis so disable copying.
  TF_DISALLOW_COPY_AND_ASSIGN(PointsToSet);
};

// This class describes a particular subshape in a computation (instruction and
// shape index) and the logical buffer which may be a source of the subshape
// value.
class BufferAlias {
 public:
  BufferAlias(const LogicalBuffer& buffer, HloInstruction* instruction,
              const ShapeIndex& index)
      : buffer_(&buffer), instruction_(instruction), index_(index) {}

  // Return the logical buffer aliased at the instruction and index.
  const LogicalBuffer& buffer() const { return *buffer_; }

  // Return the instruction/index of the subshape.
  HloInstruction* instruction() const { return instruction_; }
  const ShapeIndex& index() const { return index_; }

  bool operator==(const BufferAlias& other) const {
    return buffer_ == other.buffer_ && instruction_ == other.instruction_ &&
           index_ == other.index_;
  }
  bool operator!=(const BufferAlias& other) const { return !(*this == other); }

  string ToString() const;

 private:
  const LogicalBuffer* buffer_;
  HloInstruction* instruction_;
  const ShapeIndex index_;
};

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias);

// DFS visitor that performs tuple points-to analysis. This analysis determines
// the potential sources of each buffer in each instruction's output.
class TuplePointsToAnalysis : public DfsHloVisitorWithDefault {
 public:
  // Runs points-to analysis on 'module'. If 'include_loop_fusion_instructions'
  // is true, includes fused instructions from each loop fusion instruction
  // in 'module' in the points-to analysis.
  static StatusOr<std::unique_ptr<TuplePointsToAnalysis>> Run(
      const HloModule* module,
      const bool include_loop_fusion_instructions = false);

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

  // Return a vector containing all BufferAliases of the given logical buffer
  // This trivially includes the BufferAlias with same instruction and index as
  // the logical buffer itself, so the returned vector is never empty.  The
  // buffer alias set is the inverse of the points-to set. That is,
  // LogicalBuffer B is in the points-to set of instruction I at index N iff
  // instruction I, index N is a BufferAlias of B.
  const std::vector<BufferAlias>& GetBufferAliases(
      const LogicalBuffer& buffer) const;

  // Return a vector containing all logical buffers in the module.
  const std::vector<std::unique_ptr<LogicalBuffer>>& logical_buffers() const {
    return logical_buffers_;
  }

  // Returns a vector of buffers that the instruction produces. Most
  // instructions produce a single buffer (the top-level buffer), some produce
  // no buffers (eg bitcast), and some produce more than one buffer (eg,
  // tuple-shaped parameters).
  const std::vector<const LogicalBuffer*>& GetBuffersDefinedByInstruction(
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
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override;

  string ToString() const;

 private:
  explicit TuplePointsToAnalysis(const HloModule* module,
                                 const bool include_loop_fusion_instructions)
      : module_(module),
        include_loop_fusion_instructions_(include_loop_fusion_instructions) {}

  // Perform the analysis. Should be called immediately after constructing the
  // object and before calling GetPointsToSet.
  Status Analyze();

  // Populates instruction-defined buffers and aliases for each instruction
  // in 'instructions'. The parameter 'instructions' is passed in a form
  // common to how both HloComputation, and fusion instructions maintain a
  // list of instructions.
  Status PopulateDefinedBuffersAndAliases(
      const std::list<std::unique_ptr<HloInstruction>>& instructions);

  // Create a new logical buffer and return a reference to it. The newly created
  // buffer is stored in an internal vector of LogicalBuffers and can be
  // accessed with GetBuffer.
  const LogicalBuffer& NewLogicalBuffer(HloInstruction* instruction,
                                        const ShapeIndex& index);

  // Creates an empty PointsToSet in the points_to_ map for the given
  // instruction.
  PointsToSet& CreateEmptyPointsToSet(const HloInstruction* instruction);

  // Creates a PointsToSet in the points_to_ map for 'instruction' which is a
  // copy of the existing PointsToSet for 'src'.
  PointsToSet& CreateCopiedPointsToSet(const HloInstruction* instruction,
                                       const HloInstruction* src);

  // Adds the buffers defined by the given instruction to the given vector.
  Status GatherBuffersDefinedByInstruction(
      const HloInstruction* instruction,
      std::vector<const LogicalBuffer*>* buffers);

  // Print points-to set for 'instruction' to 'output'.
  void InstructionToString(const HloInstruction* instruction,
                           string* output) const;

  // The module this analysis is performed on.
  const HloModule* module_;

  // Whether to run points-to analysis on loop fusion instructions in 'module_'.
  const bool include_loop_fusion_instructions_;

  // A map containing a PointsToSet for every HLO instruction.
  tensorflow::gtl::FlatMap<const HloInstruction*, std::unique_ptr<PointsToSet>>
      points_to_;

  // A map containing the LogicalBuffers defined by each HLO instruction.
  tensorflow::gtl::FlatMap<const HloInstruction*,
                           std::vector<const LogicalBuffer*>>
      instruction_defined_buffers_;

  tensorflow::gtl::FlatMap<const LogicalBuffer*, std::vector<BufferAlias>>
      buffer_aliases_;

  // All logical buffers in the module, indexed by LogicalBuffer::Id. Keep as
  // vector of std::unique_ptr to keep the underlying pointer values stable.
  std::vector<std::unique_ptr<LogicalBuffer>> logical_buffers_;

  // The ID of the next logical buffer created.
  LogicalBuffer::Id next_buffer_id_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(TuplePointsToAnalysis);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
