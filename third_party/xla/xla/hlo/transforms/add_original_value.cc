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

#include "xla/hlo/transforms/add_original_value.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/shape_util.h"

namespace xla {

absl::StatusOr<bool> AddOriginalValue::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (const auto computation : module->computations()) {
    for (const auto instruction : computation->instructions()) {
      auto original_value =
          std::make_shared<OriginalValue>(instruction->shape());

      if (instruction->opcode() == HloOpcode::kGetTupleElement) {
        const auto* tuple = instruction->operand(0);
        original_value->CopySubtreeFrom(*tuple->original_value(),
                                        {instruction->tuple_index()}, {});
      } else if (instruction->opcode() == HloOpcode::kTuple) {
        for (int64_t operand_number = 0;
             operand_number < instruction->operand_count(); ++operand_number) {
          original_value->CopySubtreeFrom(
              *instruction->operand(operand_number)->original_value(), {},
              {operand_number});
        }
      } else {
        for (auto& leaf : original_value->leaves()) {
          leaf.second = {std::string(instruction->name()), leaf.first};
        }
      }
      instruction->set_original_value(original_value);
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
