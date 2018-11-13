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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Analysis which allocates HloBuffers to HloValues.
class HloAliasAnalysis {
 public:
  // The callgraph of the given HloModule must be flattened
  // (xla::FlattenCallGraph) prior to running the analysis.
  static StatusOr<std::unique_ptr<HloAliasAnalysis>> Run(
      HloModule* module,
      const HloDataflowAnalysis::FusionCanShareBufferFunction&
          fusion_can_share_buffer);

  string ToString() const;

  // Return the buffer containing the given value.
  const HloBuffer& GetBufferContainingValue(const HloValue& value) const {
    return *value_to_buffer_.at(&value);
  }
  HloBuffer& GetBufferContainingValue(const HloValue& value) {
    return *value_to_buffer_.at(&value);
  }

  // Return the HloBuffer with the given ID.
  const HloBuffer& GetBuffer(HloBuffer::Id buffer_id) const {
    return buffers_.at(buffer_id);
  }
  HloBuffer& GetBuffer(HloBuffer::Id buffer_id) {
    return buffers_.at(buffer_id);
  }

  // Returns the unique buffer at the given position. CHECK fails if the buffer
  // set at that position does not contain exactly one buffer.
  const HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                                     const ShapeIndex& index = {}) const;
  HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                               const ShapeIndex& index = {});

  // Compute the set of buffers at the given instruction and index and return as
  // a vector. This set is exactly the union of the buffers containing the
  // HloValues at this position.
  std::vector<const HloBuffer*> ComputeBuffersAt(
      const HloInstruction* instruction, const ShapeIndex& index = {}) const;

  // Return a vector of all HloBuffers stabily sorted by HloBuffer::Id. This
  // vector is lazily computed. Mutating operations on HloAliasAnalysis may
  // invalidate the underlying vector requiring recomputation.
  const std::vector<HloBuffer>& buffers() const { return buffers_; }

  // Returns the underlying dataflow analysis used by this alias analysis.
  const HloDataflowAnalysis& dataflow_analysis() const {
    return *dataflow_analysis_;
  }

  // Returns true if any index in the output of the given instruction has more
  // than one buffer. That is, ComputeBuffersAt returns a vector with more than
  // one element.
  bool InstructionBuffersAreAmbiguous(const HloInstruction* instruction) const;

  // Returns true if no HloBuffer appears in more than one shape index in the
  // output of the given instruction.
  bool InstructionBuffersAreDistinct(const HloInstruction* instruction) const;

  // Returns true if any HLO values in the module have interfering live ranges
  // assuming the given ordering.
  bool HasLiveRangeInterference(const HloOrdering& ordering) const;

 protected:
  explicit HloAliasAnalysis(HloModule* module);

  // Verify various invariants of the alias analysis.
  Status Verify() const;

  HloModule* module_;

  // The underlying dataflow analysis used by this alias analysis.
  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;

  // A map indicating which buffer a value is contained in.
  tensorflow::gtl::FlatMap<const HloValue*, HloBuffer*> value_to_buffer_;

  // A lazily constructed vector containing all HloBuffers sorted by
  // HloBuffer::Id.
  std::vector<HloBuffer> buffers_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
