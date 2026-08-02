/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CALL_OUTLINER_H_
#define XLA_SERVICE_CALL_OUTLINER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class HloInstruction;

// This pass uses information from the call-marker pass to outline previously
// inlined computations.
class CallOutliner : public HloModulePass {
 public:
  absl::string_view name() const override { return "call-outliner"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  static constexpr absl::string_view kDefaultOutlinedComputationName =
      "outlined_computation";

  // Represents a block of instructions bounded by call markers (`_before` and
  // `_after` custom-call instructions) that is a candidate for being outlined
  // into a new computation.
  struct OutlineBlock {
    HloInstruction* before;
    HloInstruction* after;
    std::vector<HloInstruction*> body;
  };

  // Creates parameters for the new computation corresponding to the operands
  // of the `_before` marker.
  void InitializeParameters(const OutlineBlock& block,
                            HloComputation::Builder& builder,
                            std::vector<HloInstruction*>& new_parameters,
                            std::vector<HloInstruction*>& old_operands);

  // Resolves an operand from the original computation to its counterpart in the
  // new outlined computation.
  // - If the operand was already cloned, returns the clone.
  // - If it is a GTE on the `_before` marker, returns the corresponding
  // parameter.
  // - If it is an external value, creates a new parameter for it in the new
  //   computation and records the mapping.
  HloInstruction* GetOrCreateMappedOperand(
      HloInstruction* operand, const OutlineBlock& block,
      HloComputation::Builder& builder,
      std::vector<HloInstruction*>& new_parameters,
      std::vector<HloInstruction*>& old_operands);

  // Clones `inst` into the new computation after remapping its operands.
  void ProcessInstruction(HloInstruction* instruction,
                          const OutlineBlock& block,
                          HloComputation::Builder& builder,
                          std::vector<HloInstruction*>& new_parameters,
                          std::vector<HloInstruction*>& old_operands);

  bool IsBeforeMarker(const HloInstruction* instruction);
  bool IsAfterMarker(const HloInstruction* instruction);

  // Pushes a new block to the stack and records its index for fast lookup.
  void HandleBeforeMarker(HloInstruction* instruction);

  // Accumulates non-marker instructions into the body of the current block.
  void HandleOtherInstruction(HloInstruction* instruction);

  // Constructs the new computation from the block body, handling parameter
  // mapping and external captures.
  absl::StatusOr<HloComputation*> BuildOutlinedComputation(
      HloModule* module, const OutlineBlock& block,
      std::vector<HloInstruction*>& old_operands);

  // Coordinates the outlining process for a block: builds the computation,
  // creates the call, and cleans up markers.
  absl::StatusOr<HloInstruction*> OutlineAndReplaceBlock(
      HloModule* module, HloComputation* computation,
      const OutlineBlock& block);

  // Resolves abandoned blocks and triggers outlining when an `_after` marker is
  // seen.
  absl::StatusOr<bool> HandleAfterMarker(HloModule* module,
                                         HloComputation* computation,
                                         HloInstruction* instruction);

  // Main loop for processing a computation in post-order to find and outline
  // blocks.
  absl::StatusOr<bool> OutlineComputation(
      HloModule* module, HloComputation* computation,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Maps instructions from the original computation to their cloned
  // counterparts in the newly built outlined computation.
  absl::flat_hash_map<HloInstruction*, HloInstruction*>
      original_to_outlined_map_;

  // Stack to track active outline blocks during post-order traversal of a
  // computation. Pushed when a `_before` marker is seen, and popped when a
  // matching `_after` marker is seen (or when abandoned due to unbalanced
  // markers).
  std::vector<OutlineBlock> stack_;

  // Maps computation names (from frontend attributes of markers) to their
  // indices in `stack_`. This enables fast lookup of the matching `_before`
  // marker when an `_after` marker is encountered. A vector of indices is
  // used to handle multiple markers with the same name correctly.
  absl::flat_hash_map<std::string, std::vector<int>> name_to_stack_indices_;
};
}  // namespace xla

#endif  // XLA_SERVICE_CALL_OUTLINER_H_
