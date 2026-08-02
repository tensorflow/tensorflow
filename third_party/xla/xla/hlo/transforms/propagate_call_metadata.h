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

#ifndef XLA_HLO_TRANSFORMS_PROPAGATE_CALL_METADATA_H_
#define XLA_HLO_TRANSFORMS_PROPAGATE_CALL_METADATA_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Propagates metadata (op_name prefix and stack_frame_id) from kCall
// instructions into their called computations, recursing through nested
// calls and control-flow structures. This pass should run late in the pipeline
// to annotate all non-inlined calls left in the module.
class PropagateCallMetadata : public HloModulePass {
 public:
  static constexpr int kMaxOpNameSize = 1024;

  PropagateCallMetadata() = default;
  ~PropagateCallMetadata() override = default;

  absl::string_view name() const override { return "propagate-call-metadata"; }

  // Updates the op_name in `metadata` by prepending `prefix`. Returns true if
  // the metadata was modified.
  static bool UpdateOpName(OpMetadata& metadata, absl::string_view prefix);

  // Updates the stack frame of `hlo` by concatenating `parent_frame_id` as
  // ancestor. Returns true if the instruction was modified.
  static bool UpdateStackFrame(HloInstruction* hlo,
                               StackFrameId parent_frame_id);

  // Propagates metadata (op_name prefix and stack_frame_id) into a single
  // instruction and recursively into its control-flow sub-computations (while,
  // conditional). Does NOT recurse into kCall or embedded computations.
  static void PropagateMetadataToInstruction(HloInstruction* hlo,
                                             absl::string_view prefix,
                                             StackFrameId parent_frame_id);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_PROPAGATE_CALL_METADATA_H_
