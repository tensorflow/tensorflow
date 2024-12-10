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

#ifndef XLA_SERVICE_COPY_INSERTION_H_
#define XLA_SERVICE_COPY_INSERTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"

namespace xla {

// Copy insertion is a legalization HLO pass which inserts copies (kCopy
// instructions) to eliminate several kinds of problems in the HLO module.
//
//   (1) Entry parameter or a constant live out of the entry computation.  Entry
//       computation arguments and constants have different lifetimes than the
//       computation result and cannot share the same allocation. Parameters and
//       constants live out of non-entry computations do not need copies.
//
//   (2) Different values which are simultaneously live and which must be held
//       in the same buffer. This can occur in while bodies. Specifically, the
//       while loop state (the arguments to the while instruction) is updated
//       in-place and the update may clobber the value from the previous
//       iteration before the previous value is dead. Computations called from
//       kCall instructions do not need such copies because kCall has no update
//       in-place semantics.
//
//   (3) The buffer set of the root instruction of the entry computation must be
//       unambiguous and distinct. That is, InstructionAliasSet::IsAmbiguous and
//       InstructionAliasSet::IsDistinct return true.
class CopyInsertion : public HloModulePass {
 public:
  absl::string_view name() const override { return "copy-insertion"; }
  static constexpr int64_t kUseRegionAnalysisLimit = 0;

  // backend specific function that decides whether an instruction
  // can share buffer with its operand.
  //
  // TODO(b/80315712): Find a better way to tell whether a fusion can share
  // buffer.
  explicit CopyInsertion(
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr,
      int64_t use_region_based_live_range_analysis = kUseRegionAnalysisLimit)
      : can_share_buffer_(can_share_buffer),
        use_region_based_live_range_analysis_(
            use_region_based_live_range_analysis) {}

  // Run the pass on the given module. Returns whether the module was changed
  // (copies were inserted).
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Try to remove as many copies from the module as possible without
  // introducing live range interference. Only copy instructions that are
  // eligible for copy elision are considered for removal.
  // If check_live_range_ordering is true, check that live ranges are ordered
  // in all the existing aliased buffers.
  absl::Status RemoveUnnecessaryCopies(
      HloModule* module, bool check_live_range_ordering = false,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Add copies to address special constraints on the roots of computations not
  // related to live range interference:
  //
  //    (1) Entry computation root must be unambiguous and distinct.
  //
  //    (2) Any computation called by a kCall instruction must have an
  //        unambiguous root.
  //
  //    (3) Constants and parameters cannot be live out of the entry computation
  //
  absl::Status AddSpecialCaseCopies(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

 protected:
  // Override which requires the caller to pass in a call graph.
  virtual absl::Status AddSpecialCaseCopies(
      const CallGraph& call_graph,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      HloModule* module);

  // Add copies for conditional instructions.
  virtual absl::Status AddCopiesForConditional(
      const HloAliasAnalysis& alias_analysis, HloInstruction* conditional);

  // Add copies for async send/recv instructions.
  absl::Status AddCopiesForAsyncSendRecv(const HloAliasAnalysis& alias_analysis,
                                         HloInstruction* async);

  // Backend specific function that decides whether an instruction can share
  // buffer with its operand.
  HloDataflowAnalysis::CanShareBuffer can_share_buffer_;

 private:
  absl::Status AddCopiesToResolveInterference(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
  int64_t use_region_based_live_range_analysis_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COPY_INSERTION_H_
