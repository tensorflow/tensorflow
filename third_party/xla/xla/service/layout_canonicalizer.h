/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LAYOUT_CANONICALIZER_H_
#define XLA_SERVICE_LAYOUT_CANONICALIZER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// HLO pass that canonicalizes all layouts (except input and output of module)
// to have descending layout by default. This is done by applying the layout
// order to the logical dimension ordering and transform each operation
// attributes according to the new logical shape.
class LayoutCanonicalizer : public HloModulePass {
 public:
  explicit LayoutCanonicalizer() = default;
  ~LayoutCanonicalizer() override = default;
  absl::string_view name() const override { return "cononicalize_layout"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Given an instruction (with non-tuple output shape), if the instruction has
  // non canonical output, this function updates the output shape such that the
  // layout will be descending. It adds the instruction and its majpr-to-minor
  // map to the map of all the canonicalized instructions. We lookup this map
  // later when canonicalizng operands of an instruction.
  bool CanonicalizeOutputShape(HloInstruction* instr);

  // Given an instruction, this function canonicalizes all non-parameter
  // operands. It also updates the canonicalized_instr map and adds the updated
  // operand.
  bool CanonicalizeOperands(HloInstruction* instr);

  // Given an instruction, this function canonicalizes the subgraph rooted at
  // instr. It looks at the type of the instruction and calls the appropriate
  // handler for the instruction.
  bool CanonicalizeInstructionLayout(HloInstruction* instr, bool is_entry_root);

 private:
  // Holds the mapping from instructions to major-to-minor layout ordering
  // pre-canonicalization.
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>
      canonicalized_instrs_;
};
}  // namespace xla

#endif  // XLA_SERVICE_LAYOUT_CANONICALIZER_H_
