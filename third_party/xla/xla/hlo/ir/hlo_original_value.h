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
#include <utility>

#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The information of a tensor in an unoptimized HLO module.
struct OriginalTensor {
  // The name of the instruction that produces this tensor or a tuple that
  // includes this tensor.
  std::string instruction_name;
  // Shape index of the tensor if the instruction produces a tuple.
  ShapeIndex shape_index;
};

// The information of an HLO value produced by an instruction in an unoptimized
// HLO module.
class OriginalValue : public ShapeTree<std::optional<OriginalTensor>> {
 public:
  explicit OriginalValue(Shape shape) : ShapeTree(std::move(shape)) {}
  std::string ToString();
  OriginalValueProto ToProto();
  static std::shared_ptr<OriginalValue> FromProto(
      const xla::OriginalValueProto& original_value_proto);
};

// Copies the original value of the source to the destination instruction. This
// performs a deep copy if clone is set to true. Otherwise, it performs a
// shallow copy.
void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone);
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
