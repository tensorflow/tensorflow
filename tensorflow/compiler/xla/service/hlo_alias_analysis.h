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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

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
