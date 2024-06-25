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

#ifndef XLA_SERVICE_STABLE_SORT_EXPANDER_H_
#define XLA_SERVICE_STABLE_SORT_EXPANDER_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/op_expander_pass.h"

namespace xla {

// HLO pass which expands Sort ops that have the is_stable field set to true
// into equivalent Sort ops which guarantee stable sorting without relying on
// the is_stable field.
class StableSortExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "stable-sort-expander"; }

  // Returns the index of the sort operand that is an iota op with an iota
  // dimension which is the same as the dimension to sort. Also it should have
  // an integral type that is large enough for the number of elements in the
  // sort dimension. For now, we only allow S32, because we expect to find a S32
  // iota operand for all Sort ops which are created by TopK.
  //
  // If no operand of the input sort matches the conditions above, returns -1.
  static int64_t IotaOperandIndexForStableSort(const HloSortInstruction& sort);

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_STABLE_SORT_EXPANDER_H_
