/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_HLO_ORIGINAL_VALUE_UTIL_H_
#define XLA_HLO_IR_HLO_ORIGINAL_VALUE_UTIL_H_

#include <cstdint>
#include <memory>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_original_value.h"

namespace xla {

// Copies the original value of the source to the destination instruction.
// Original arrays in the source original value are rearranged in the new
// original value according to the given mapping of old to new tuple indices.
template <typename T>
std::enable_if_t<std::is_integral_v<T>> CopyOriginalValue(
    const HloInstruction* src_instruction, HloInstruction* dest_instruction,
    const absl::flat_hash_map<T, T>& old_to_new_tuple_idx) {
  std::shared_ptr<OriginalValue> old_original_value =
      src_instruction->original_value();
  if (!old_original_value) {
    return;
  }
  const int64_t src_tuple_size = old_original_value->tree().num_leaves();
  const int64_t dest_tuple_size = old_to_new_tuple_idx.size();
  std::shared_ptr<xla::OriginalValue> new_original_value =
      std::make_shared<xla::OriginalValue>(dest_instruction->shape());
  for (const auto& [old_idx, new_idx] : old_to_new_tuple_idx) {
    if (old_idx < 0 || old_idx >= src_tuple_size || new_idx < 0 ||
        new_idx >= dest_tuple_size) {
      return;
    }
    new_original_value->mutable_tree()->CopySubtreeFrom(
        old_original_value->tree(), {old_idx}, {new_idx});
  }
  dest_instruction->set_original_value(new_original_value);
}

// Copies the original value of the source to the destination instruction if the
// shapes of the source and destination are compatible. This performs a deep
// copy if clone is set to true. Otherwise, it performs a shallow copy. Print a
// warning if the shapes are not compatible and issue_warning is set to true.
void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone,
                       bool issue_warning);

// Removes duplicates of original value objects referenced in the module to save
// memory storage.
void DeduplicateOriginalValues(HloModule* module);
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_UTIL_H_
