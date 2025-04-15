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

#ifndef XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
#define XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_

#include <optional>
#include <string>

#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
// Stores information of original values.
struct OriginalArray {
  std::string instruction_name;
  ShapeIndex shape_index;
};

using OriginalValue = ShapeTree<std::optional<OriginalArray>>;

std::string OriginalValueToString(const OriginalValue& original_value);

OriginalValueProto OriginalValueToProto(const OriginalValue& original_value);

// Copy the original value from the source to the destination instruction. Note
// the original values of fused instructions are copied when they are added
// into a fusion, so it's not required to copy the value if the target is a
// fusion instruction, which should have the same original value as the root of
// the fused computation anyway. However, we will copy the value nontheless to
// simplify some use cases that involve fusions.
void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction);
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
