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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/hash/hash.h"

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
  LogicalBuffer::Id num_logical_buffers() const { return next_buffer_id_; }

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
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;

  // A map from the buffer ID to the logical buffer
  std::vector<std::unique_ptr<LogicalBuffer>> logical_buffers_;

  struct Hasher {
    size_t operator()(
        std::pair<const HloInstruction*, const ShapeIndex> p) const {
      size_t inst_hash = tensorflow::hash<const HloInstruction*>()(p.first);
      for (auto index = p.second.begin(); index != p.second.end(); ++index) {
        inst_hash = tensorflow::Hash64Combine(*index, inst_hash);
      }
      return inst_hash;
    }
  };

  // A map from an hlo + shape index to the logical buffer representing
  // the appropriate output.
  std::unordered_map<std::pair<const HloInstruction*, const ShapeIndex>,
                     LogicalBuffer*, Hasher>
      output_buffers_;

  // The ID of the next logical buffer created.
  LogicalBuffer::Id next_buffer_id_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOGICAL_BUFFER_ANALYSIS_H_
