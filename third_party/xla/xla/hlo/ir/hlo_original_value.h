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

// The information of an array in an unoptimized HLO module.
struct OriginalArray {
  // The name of the instruction in the unoptimized HLO module that produces
  // this array or a tuple that includes this array.
  std::string instruction_name;
  // Shape index of the array if the instruction produces a tuple.
  ShapeIndex shape_index;
  std::string ToString() const;
  OriginalArrayProto ToProto() const;
  static OriginalArray FromProto(
      const xla::OriginalArrayProto& original_array_proto);

  friend bool operator==(const OriginalArray& lhs, const OriginalArray& rhs) {
    return lhs.instruction_name == rhs.instruction_name &&
           lhs.shape_index == rhs.shape_index;
  }

  friend bool operator!=(const OriginalArray& lhs, const OriginalArray& rhs) {
    return !(lhs == rhs);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalArray& original_array) {
    return H::combine(std::move(h), original_array.instruction_name,
                      original_array.shape_index);
  }
};

// The information of an HLO value produced by an instruction in an unoptimized
// HLO module.
class OriginalValue : public ShapeTree<std::optional<OriginalArray>> {
 public:
  explicit OriginalValue(Shape shape) : ShapeTree(std::move(shape)) {}
  std::string ToString() const;
  OriginalValueProto ToProto() const;
  static std::shared_ptr<OriginalValue> FromProto(
      const xla::OriginalValueProto& original_value_proto);
  static std::shared_ptr<OriginalValue> CreateFromInstruction(
      const HloInstruction* instruction, absl::string_view prefix = "");
};

struct OriginalValuePointer {
  OriginalValuePointer() = default;
  explicit OriginalValuePointer(
      std::shared_ptr<xla::OriginalValue> original_value) {
    this->original_value = std::move(original_value);
  }

  friend bool operator==(const OriginalValuePointer& lhs,
                         const OriginalValuePointer& rhs) {
    // Returns if any original value is empty.
    if (!lhs.original_value || !rhs.original_value) {
      return !lhs.original_value == !rhs.original_value;
    }
    // Returns if the original values have different shapes.
    if (!xla::ShapeUtil::Compatible(lhs.original_value->shape(),
                                    rhs.original_value->shape())) {
      return false;
    }
    // Compares nodes.
    for (auto& leaf : lhs.original_value->leaves()) {
      xla::ShapeIndex index = leaf.first;
      std::optional<const xla::OriginalArray> lhs_original_array = leaf.second;
      std::optional<const xla::OriginalArray> rhs_original_array =
          rhs.original_value->element(index);
      if (!lhs_original_array.has_value() || !rhs_original_array.has_value() ||
          *lhs_original_array != *rhs_original_array) {
        return false;
      }
    }
    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalValuePointer& value) {
    // Ignore layout information, which is added to shapes during the HLO
    // transformation.
    h = xla::Shape::template Hash<H, false /*kIsLayoutSensitive*/>(
        std::move(h), value.original_value->shape());
    value.original_value->ForEachElement(
        [&h, &value](const xla::ShapeIndex& shape_index,
                     const std::optional<xla::OriginalArray>& original_array) {
          if (!value.original_value->IsLeaf(shape_index)) {
            return;
          }
          if (!original_array) {
            return;
          }
          h = H::combine(std::move(h), original_array->instruction_name,
                         original_array->shape_index);
        });
    return std::move(h);
  }

  std::shared_ptr<xla::OriginalValue> original_value = nullptr;
};

// Copies the original value of the source to the destination instruction. This
// performs a deep copy if clone is set to true. Otherwise, it performs a
// shallow copy.
void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone);

// Removes duplicates of original value objects referenced in the module to save
// memory storage.
void DeduplicateOriginalValues(HloModule* module);
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
