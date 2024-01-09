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

#ifndef XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_
#define XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/logical_buffer.h"
#include "xla/statusor.h"

namespace xla {
// A class to create all the logical buffers defined by the HLO ops in a module.
class LogicalBufferAnalysis : public DfsHloVisitorWithDefault {
 public:
  // Runs points-to analysis on 'module'.
  static StatusOr<std::unique_ptr<LogicalBufferAnalysis>> Run(
      const HloModule* module);

  // Returns the logical buffer with the given ID.
  LogicalBuffer& GetBuffer(LogicalBuffer::Id id) const;

  // Returns the logical buffer that represents the output of a given HLO
  // at a given index.
  LogicalBuffer& GetBuffer(HloInstruction* instruction,
                           const ShapeIndex& index) const;

  const std::vector<std::unique_ptr<LogicalBuffer>>& logical_buffers() const {
    return logical_buffers_;
  }
  size_t num_logical_buffers() const { return logical_buffers_.size(); }

 private:
  explicit LogicalBufferAnalysis(const HloModule* module) : module_(module) {}
  Status Analyze();

  // The module this analysis is performed on.
  const HloModule* module_;

  // Create a new logical buffer and return a reference to it. The newly created
  // buffer is stored in an internal vector of LogicalBuffers and can be
  // accessed with GetBuffer.
  void NewLogicalBuffer(HloInstruction* instruction, const ShapeIndex& index);

  Status DefaultAction(HloInstruction* hlo_instruction) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleDomain(HloInstruction* domain) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleCopyStart(HloInstruction* copy_start) override;
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleFusion(HloInstruction* fusion) override;

  // A map from the buffer ID to the logical buffer
  std::vector<std::unique_ptr<LogicalBuffer>> logical_buffers_;

  // A map from an hlo + shape index to the logical buffer representing
  // the appropriate output.
  absl::flat_hash_map<std::pair<const HloInstruction*, const ShapeIndex>,
                      LogicalBuffer*>
      output_buffers_;
  // Whether to alias buffers defined by dataflow relations. This aliasing
  // relation should not be recognized if copies can be inserted to break up
  // the dataflow relation-induced aliasing.
  const bool alias_buffer_across_dataflow_ = false;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_
