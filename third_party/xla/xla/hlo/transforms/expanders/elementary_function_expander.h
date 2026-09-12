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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_ELEMENTARY_FUNCTION_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_ELEMENTARY_FUNCTION_EXPANDER_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/util.h"

namespace xla {

// A pass which expands elementary math functions (such as sinh, cosh, etc.)
// into other HLO instructions when they are not natively supported or when
// an algebraic expansion is preferred for the operand type.
class ElementaryFunctionExpander : public OpExpanderPass {
 public:
  explicit ElementaryFunctionExpander(HloPredicate extra_filter = nullptr)
      : OpExpanderPass(std::move(extra_filter)) {}
  ~ElementaryFunctionExpander() override = default;
  absl::string_view name() const override {
    return "elementary-function-expander";
  }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_ELEMENTARY_FUNCTION_EXPANDER_H_
