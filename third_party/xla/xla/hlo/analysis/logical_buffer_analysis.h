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

#ifndef XLA_HLO_ANALYSIS_LOGICAL_BUFFER_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_LOGICAL_BUFFER_ANALYSIS_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"

namespace xla {
// Identifies and instantiates all the logical buffers defined by HLO
// instructions in an HLO module.
//
// This class visits HLO instructions in post-order DFS order to determine
// which output shape indices of which HLO instructions define/create a new
// buffer and which ones alias existing buffers. It assigns a unique
// `LogicalBuffer::Id` to each newly defined buffer.
//
// Analysis steps:
// 1. Visit each HLO instruction in DFS order.
// 2. Identify the logical buffers defined/created by the instruction.
// 3. Populate `logical_buffers_` (for retrieval by ID) and `output_buffers_`
//    (for retrieval by instruction and subshape index).
class LogicalBufferAnalysis : public DfsHloVisitorWithDefault {
 public:
  // Runs the logical buffer analysis on the given `HloModule` and returns
  // the completed analysis result.
  static absl::StatusOr<std::unique_ptr<LogicalBufferAnalysis>> Run(
      const HloModule* module);

  // Returns a reference to the `LogicalBuffer` with the given ID.
  //
  // REQUIRES: The ID must be valid (less than `num_logical_buffers()`).
  // Otherwise, this method will result in a crash.
  LogicalBuffer& GetBuffer(LogicalBuffer::Id id) const;

  // Returns a reference to the `LogicalBuffer` defined by the instruction at
  // the specified `ShapeIndex` in its output.
  //
  // REQUIRES: The instruction must define a logical buffer at the given index.
  // Otherwise, this method will result in a crash.
  LogicalBuffer& GetBuffer(HloInstruction* instruction,
                           const ShapeIndex& index) const;

  // Returns a view list of all `LogicalBuffer`s created in the module.
  const std::vector<std::unique_ptr<LogicalBuffer>>& logical_buffers() const {
    return logical_buffers_;
  }

  // Returns the total number of unique `LogicalBuffer`s created.
  size_t num_logical_buffers() const { return logical_buffers_.size(); }

 private:
  explicit LogicalBufferAnalysis(const HloModule* module) : module_(module) {}

  // Performs the DFS analysis over all non-fusion and fusion computations
  // of the module.
  absl::Status Analyze();

  // The module this analysis is performed on.
  const HloModule* module_;

  // Instantiates a new `LogicalBuffer` associated with `instruction` at
  // `index` and adds it to internal collections.
  void NewLogicalBuffer(HloInstruction* instruction, const ShapeIndex& index);

  // Visitor overrides that identify buffer creation points for each
  // instruction type.
  absl::Status DefaultAction(HloInstruction* hlo_instruction) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
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
  absl::Status HandleAsyncStart(HloInstruction* async_start) override;

  // A collection of all logical buffers, indexed by their ID.
  std::vector<std::unique_ptr<LogicalBuffer>> logical_buffers_;

  // A map mapping an `(HloInstruction, ShapeIndex)` pair to the `LogicalBuffer`
  // defined at that position.
  absl::flat_hash_map<std::pair<const HloInstruction*, const ShapeIndex>,
                      LogicalBuffer*>
      output_buffers_;

  // Flag indicating whether to respect output-to-operand aliasing annotations
  // on custom calls (only).
  //
  // If true, outputs explicitly annotated to alias an operand do not define
  // new logical buffers (the operand's buffer is forwarded). If false, new
  // logical buffers are created for all outputs, ignoring those annotations.
  //
  // For example, given:
  //   %ccall = custom-call(%param), output_to_operand_aliasing={ {}: (0, {}) }
  //
  // - If true: `%ccall` does not define a new buffer; it reuses `%param`'s.
  // - If false: `%ccall` defines a new `LogicalBuffer(%ccall, {})`.
  //
  // Note: This does not affect Call, Fusion, AsyncStart, Parameter,
  // or While instructions, which always define new buffers.
  const bool alias_buffer_across_dataflow_ = false;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_LOGICAL_BUFFER_ANALYSIS_H_
