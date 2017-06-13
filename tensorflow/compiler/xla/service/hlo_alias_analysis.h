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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_

#include <stddef.h>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
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

// A container which can hold one or more HloValues. An HLO buffer abstractly
// represents the allocation which HLO instructions write into and read
// from. Generally there is a one-to-one correspondence between HloBuffers and
// HloValue where each HloValue in the module is held in a unique HloBuffer. An
// exception is the while instruction which updates the loop state in-place. In
// this case, we have a single HloBuffer for each HloLocation in the loop state,
// but multiple HloValues. For example:
//
//   %init = ...
//   %while = While(%init, body, condition)
//
//  body:
//   %body_param = Param(0)
//     ...
//   %body_root = ...
//
//  condition:
//   %cond_param = Param(0)
//     ...
//
// For simplicity, assume that %while is array-shaped. In this case, we have a
// single HloBuffer which holds the following HloValues: HloValue{%init},
// HloValue{%while}, HloValue{%body_param}, HloValue{%body_root}, and
// HloValue{%cond_param}.
//
// HloBuffers may appear at different HloLocations in the module mirroring the
// same propery of HloValues. For example:
//
//   %sub = Sub(...)
//   %add = Add(...)
//   %tuple = Tuple(%add, %sub)
//   %gte = GetTupleElement(%tuple, 0)
//
// In this case, the HloBuffer containing %add appears at the following
// locations: HloLocation{%add, {}}, HloLocation{%tuple, {0}}, and
// HloLocation{%gte, {}}.
//
// Different HloLocations which share the same HloBuffer indicate mandatory
// aliasing in the HLO module. These locations must share the same memory
// allocation for correctness (the backends rely on this property). This differs
// from incidental aliasing introduced by memory reuse in BufferAssignment where
// different instructions may happen to get the same allocation.
class HloBuffer {
 public:
  using Id = int64;

  HloBuffer(int64 id) : id_(id) {}

  // Return the unique identifier for this HloBuffer.
  int64 id() const { return id_; }

  // Add a value to the set of values held by this buffer. Also adds the
  // HloLocations of the value to the locations vector of the buffer. If the
  // buffer already contains this value, then this method is a nop.
  void AddValue(const HloValue& value);

  // Return the IDs of all values contained in this buffer.
  const std::vector<HloValue::Id>& value_ids() const { return value_ids_; }

  // Return the locations (output of which instruction and at what index) where
  // the buffer is used. This is exactly the union of the locations of the
  // HloValues contained by the buffer.
  const std::vector<HloLocation>& locations() const { return locations_; }

  string ToString() const;

  bool operator==(const HloBuffer& other) const;
  bool operator!=(const HloBuffer& other) const { return !(*this == other); }

 private:
  // Unique identifier for this HloBuffer.
  const Id id_;

  // The set of values contained in the this buffer.
  std::vector<HloValue::Id> value_ids_;

  // The set of locations where this buffer is used.
  std::vector<HloLocation> locations_;
};

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer);

// A class representing the set of possible HloBuffers at a particular
// HloLocation (shape index in the output of an instruction) in the XLA
// graph. In most cases, the buffer set will have a single HloBuffer indicating
// that the HloBuffer which appears at that particular location is known
// unambiguously at compile-time.  However, tuple-shaped Select instructions can
// introduce ambiguity as the tuple elements of the operands are passed by
// reference into the output of the Select. For example:
//
//   %pred = ...
//   %tuple0 = Tuple(%a, %b)
//   %tuple1 = Tuple(%x, %y)
//   %select = Select(%pred, %tuple0, %tuple1)
//
// In this case the HloBufferSet at HloLocation{%select, {0}} contains the
// HloBuffer holding %a and the HloBuffer holding %x.
class HloBufferSet {
 public:
  HloBufferSet() = default;

  // Add the given buffer to this buffer set. If the buffer already exists in
  // the set, then this is a NOP.
  void AddBuffer(HloBuffer::Id buffer_id);

  // Removes the given buffer from this buffer set. CHECK fails in the buffer is
  // not contained in this set.
  void RemoveBufferOrDie(HloBuffer::Id buffer_id);

  // Returns the unique buffer in this set. CHECK fails if the set does not
  // contain exactly one buffer.
  HloBuffer::Id GetUniqueBufferId() const {
    CHECK_EQ(buffer_ids().size(), 1);
    return buffer_ids()[0];
  }

  // Returns the IDs of the HloBuffers contained in this buffer set.
  const std::vector<HloBuffer::Id>& buffer_ids() const { return buffer_ids_; }

  string ToString() const;

 private:
  // The IDs of the HloBuffers containted in this buffer set.
  std::vector<HloBuffer::Id> buffer_ids_;
};

std::ostream& operator<<(std::ostream& out, const HloBufferSet& buffer_set);

// A class collecting the HloBuffers in the output of an HLO instruction. For
// array-shaped instructions, an InstructionBufferSet trivially holds a single
// HloBufferSet. Tuple-shaped InstructionBufferSets hold multiple
// HloBufferSets.
class InstructionBufferSet : public ShapeTree<HloBufferSet> {
 public:
  InstructionBufferSet(const Shape& shape) : ShapeTree<HloBufferSet>(shape) {}

  // Returns true if any HloBufferSet contained in this InstructionBufferSet
  // is not a singleton.
  bool IsAmbiguous() const;

  // Returns true if any HloBuffer appears in more than one HloBufferSet
  // contained in this InstructionBufferSet.
  bool IsDistinct() const;

  string ToString() const;
};

std::ostream& operator<<(std::ostream& out,
                         const InstructionBufferSet& buffer_set);

class HloAliasAnalysis {
 public:
  static StatusOr<std::unique_ptr<HloAliasAnalysis>> Run(HloModule* module);

  string ToString() const;

  // Return the InstructionBufferSet for the given instruction.
  const InstructionBufferSet& GetInstructionBufferSet(
      const HloInstruction* instruction) const;
  InstructionBufferSet& GetInstructionBufferSet(
      const HloInstruction* instruction);

  // Return the HloBufferSet for the given location.
  const HloBufferSet& GetBufferSet(const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const;
  HloBufferSet& GetBufferSet(const HloInstruction* instruction,
                             const ShapeIndex& index = {});

  // Return the HloBuffer with the given ID.
  const HloBuffer& GetBuffer(HloBuffer::Id buffer_id) const {
    return buffers_.at(buffer_id);
  }
  HloBuffer& GetBuffer(HloBuffer::Id buffer_id) {
    return buffers_.at(buffer_id);
  }

  // Returns the unique buffer at the given location. CHECK fails if the buffer
  // set at that location does not contain exactly one buffer.
  const HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                                     const ShapeIndex& index = {}) const {
    return GetBuffer(GetBufferSet(instruction, index).GetUniqueBufferId());
  }
  HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                               const ShapeIndex& index = {}) {
    return GetBuffer(GetBufferSet(instruction, index).GetUniqueBufferId());
  }

  // Return a vector of all HloBuffers stabily sorted by HloBuffer::Id. This
  // vector is lazily computed. Mutating operations on HloAliasAnalysis may
  // invalidate the underlying vector requiring recomputation.
  const std::vector<const HloBuffer*>& buffers() const;

  // Returns the underlying dataflow analysis used by this alias analysis.
  const HloDataflowAnalysis& dataflow_analysis() const {
    return *dataflow_analysis_;
  }

 protected:
  HloAliasAnalysis(HloModule* module);

  // Creates a new HloBuffer and returns a reference to it.
  HloBuffer& NewHloBuffer();

  // Construct the initial set of buffer sets where an HloBuffer is created for
  // each HloValue in the module.
  void InitializeBufferSets();

  // Combine the InstructionBufferSets for given instructions. The HloBuffers in
  // the HloBufferSets at each ShapeIndex are combined via CombineBuffers
  // into a single HloBuffer. This single HloBuffer then becomes the only member
  // of these HloBufferSets (ie, they become singletons). The HloBuffers
  // which are removed from the buffer sets are deleted from the analysis. This
  // flattening may change InstructionBufferSets of other instructions not in
  // 'instructions' because the HloBuffers of the InstructionBufferSets of
  // 'instructions' can be used elsewhere in the module.
  //
  // This method is used to enforce the mandatory aliasing of while instructions
  // where the init operand, body parameter, condition parameter, body root
  // instruction, and the while itself must have exactly the same HloBuffer at
  // each ShapeIndex.
  //
  // Precondition: The shapes on the given instructions must be compatible.
  void FlattenInstructionBufferSets(
      tensorflow::gtl::ArraySlice<const HloInstruction*> instructions);

  // Combines the given HloBuffers into a single buffer. One of the given
  // HloBuffers is chosen as the unified buffer, and all other references to the
  // remaining buffers are replaced by this unified buffer. All HloValues
  // contained in the replaced buffers are moved to the unified buffer, and the
  // replaced buffers are deleted from the analysis.
  void CombineBuffers(tensorflow::gtl::ArraySlice<HloBuffer::Id> buffer_ids);

  // Verifies internal state of the analysis.
  Status Verify() const;

  HloModule* module_;

  // The underlying dataflow analysis used by this alias analysis.
  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;

  // The map of all HloBuffers in the module.
  std::unordered_map<HloBuffer::Id, HloBuffer> buffers_;

  // A map from instruction to its InstructionBufferSet.
  std::unordered_map<const HloInstruction*, InstructionBufferSet> buffer_sets_;

  // A lazily constructed vector containing all HloBuffers sorted by
  // HloBuffer::Id.
  mutable std::vector<const HloBuffer*> buffers_vector_;

  // The Id to use for the next HloBuffer.
  int64 next_buffer_id_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
