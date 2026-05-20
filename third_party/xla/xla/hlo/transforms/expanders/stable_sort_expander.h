/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_STABLE_SORT_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_STABLE_SORT_EXPANDER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// HLO pass which expands Sort ops that have the is_stable field set to true
// into equivalent Sort ops which guarantee stable sorting without relying on
// the is_stable field.
class StableSortExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "stable-sort-expander"; }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_STABLE_SORT_EXPANDER_H_
