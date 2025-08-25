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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tuple_tree.h"
#include "xla/util.h"
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

// Represents a synthetic value, e.g., from a call instruction that doesn't
// have a direct mapping to original arrays and should be removed by inlining.
struct SyntheticCall {};

namespace internal {
TupleTree<std::optional<OriginalArray>>& EmptyOriginalValueTupleTree();
}  // namespace internal

// The information of an HLO value produced by an instruction in an unoptimized
// HLO module.
class OriginalValue {
 public:
  OriginalValue() : data_(TupleTree<std::optional<OriginalArray>>()) {}

  // Constructor for a normal value with array information.
  explicit OriginalValue(
      TupleTree<std::optional<OriginalArray>>::Node&& root_node);
  explicit OriginalValue(TupleTree<std::optional<OriginalArray>>&& tree);
  explicit OriginalValue(const TupleTree<std::optional<OriginalArray>>& tree);

  // Constructor for a synthetic value.
  explicit OriginalValue(SyntheticCall synthetic) : data_(synthetic) {}

  bool is_synthetic_call() const {
    return std::holds_alternative<SyntheticCall>(data_);
  }

  std::string ToString() const;
  OriginalValueProto ToProto() const;
  static std::shared_ptr<OriginalValue> FromProto(
      const xla::OriginalValueProto& original_value_proto);
  static std::shared_ptr<OriginalValue> CreateFromInstruction(
      const HloInstruction* instruction, absl::string_view prefix = "");

  const TupleTree<std::optional<OriginalArray>>& tree() const {
    CHECK(!is_synthetic_call())
        << "Cannot get tree from a synthetic OriginalValue";
    return std::get<TupleTree<std::optional<OriginalArray>>>(data_);
  }
  TupleTree<std::optional<OriginalArray>>* mutable_tree() {
    CHECK(!is_synthetic_call())
        << "Cannot get tree from a synthetic OriginalValue";
    return &std::get<TupleTree<std::optional<OriginalArray>>>(data_);
  }

  const std::optional<OriginalArray>& original_array(
      ShapeIndexView index) const {
    return tree().element(index);
  }
  std::optional<OriginalArray>* mutable_original_array(ShapeIndexView index) {
    return mutable_tree()->mutable_element(index);
  }

  // Returns a const iterator over the pairs of ShapeIndex and
  // std::optional<OriginalArray>.
  auto original_arrays() const {
    if (is_synthetic_call()) {
      return std::as_const(internal::EmptyOriginalValueTupleTree()).leaves();
    }
    return tree().leaves();
  }
  // Returns a non-const iterator over the pairs of ShapeIndex and
  // std::optional<OriginalArray>.
  auto mutable_original_arrays() {
    if (is_synthetic_call()) {
      return internal::EmptyOriginalValueTupleTree().leaves();
    }
    return mutable_tree()->leaves();
  }

  bool operator==(const OriginalValue& other) const {
    if (is_synthetic_call() != other.is_synthetic_call()) {
      return false;
    }
    if (is_synthetic_call()) {
      return true;  // Synthetic == Synthetic
    }
    auto this_original_arrays = original_arrays();
    auto other_original_arrays = other.original_arrays();
    return std::equal(this_original_arrays.begin(), this_original_arrays.end(),
                      other_original_arrays.begin(),
                      other_original_arrays.end());
  }

  bool operator!=(const OriginalValue& other) const {
    return !(*this == other);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalValue& value) {
    h = H::combine(std::move(h), value.is_synthetic_call());
    if (!value.is_synthetic_call()) {
      auto leaves = value.original_arrays();
      int64_t leaf_count = 0;
      for (const auto& leaf : leaves) {
        leaf_count++;
        h = H::combine(std::move(h), leaf.first, leaf.second);
      }
      h = H::combine(std::move(h), leaf_count);
    }
    return h;
  }

 private:
  void ClearInternalNodeValues();
  std::variant<SyntheticCall, TupleTree<std::optional<OriginalArray>>> data_;
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
