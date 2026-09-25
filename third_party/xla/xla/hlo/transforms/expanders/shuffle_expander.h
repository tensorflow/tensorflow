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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_SHUFFLE_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_SHUFFLE_EXPANDER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// HLO pass which rewrites Shuffle operations into equivalent sequences of
// supported operations.
//
// - rotate mode: rewritten into a Concat of Slices.
// - permute mode: rewritten into a Gather of one element of the operand per
//   element of the result.
// - multi_rotate mode: rewritten into the Gather of the permute that moves the
//   elements the same way.
class ShuffleExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "shuffle-expander"; }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_SHUFFLE_EXPANDER_H_
