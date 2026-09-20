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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

  friend bool operator==(const OriginalArray& lhs, const OriginalArray& rhs);
  friend bool operator!=(const OriginalArray& lhs, const OriginalArray& rhs);
  // Used for sorting OriginalArrays in canonical order.
  friend bool operator<(const OriginalArray& lhs, const OriginalArray& rhs) {
    if (lhs.instruction_name != rhs.instruction_name) {
      return lhs.instruction_name < rhs.instruction_name;
    }
    return lhs.shape_index < rhs.shape_index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalArray& original_array) {
    return H::combine(std::move(h), original_array.instruction_name,
                      original_array.shape_index);
  }
};

// The information of an HLO value produced by an instruction in an unoptimized
// HLO module.
class OriginalValue {
 public:
  // Constructor for a normal value with array information.
  explicit OriginalValue(
      TupleTree<std::optional<OriginalArray>>::Node&& root_node,
      std::optional<std::string> call_hierarchy = std::nullopt);
  explicit OriginalValue(
      TupleTree<std::optional<OriginalArray>>&& tree,
      std::optional<std::string> call_hierarchy = std::nullopt);
  explicit OriginalValue(
      const TupleTree<std::optional<OriginalArray>>& tree,
      std::optional<std::string> call_hierarchy = std::nullopt);
  explicit OriginalValue(
      const Shape& shape,
      std::optional<std::string> call_hierarchy = std::nullopt)
      : tree_(TupleTree<std::optional<OriginalArray>>(shape)),
        call_hierarchy_(std::move(call_hierarchy)) {}

  bool is_synthetic_call() const {
    return call_hierarchy_.has_value() && call_hierarchy_->empty();
  }

  std::string ToString() const;
  OriginalValueProto ToProto() const;
  static std::shared_ptr<OriginalValue> FromProto(
      const xla::OriginalValueProto& original_value_proto);
  static std::shared_ptr<OriginalValue> CreateFromInstruction(
      const HloInstruction* instruction, absl::string_view prefix = "");

  const TupleTree<std::optional<OriginalArray>>& tree() const { return tree_; }
  TupleTree<std::optional<OriginalArray>>* mutable_tree() { return &tree_; }

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
    return tree().leaves();
  }
  // Returns a non-const iterator over the pairs of ShapeIndex and
  // std::optional<OriginalArray>.
  auto mutable_original_arrays() {
    return mutable_tree()->leaves();
  }

  bool IsEmpty() const {
    return std::all_of(
        tree().leaves().begin(), tree().leaves().end(),
        [](const auto& pair) { return !pair.second.has_value(); });
  }

  bool IsCompatibleWith(const Shape& shape) const;

  bool IsTuple() const { return tree().IsTuple(); }

  bool operator==(const OriginalValue& other) const;

  bool operator!=(const OriginalValue& other) const {
    return !(*this == other);
  }

  // Gets the call hierarchy string of the original call instructions
  // that this OriginalValue is associated with. Returns std::nullopt if this
  // OriginalValue is not associated with a call instruction or the call
  // hierarchy is lost (e.g., after complicated optimizations).
  const std::optional<std::string>& call_hierarchy() const {
    return call_hierarchy_;
  }
  void set_call_hierarchy(std::optional<std::string> call_hierarchy) {
    call_hierarchy_ = std::move(call_hierarchy);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OriginalValue& value) {
    h = H::combine(std::move(h), value.call_hierarchy_);
    auto original_arrays = value.original_arrays();
    h = H::combine(std::move(h), std::distance(original_arrays.begin(),
                                               original_arrays.end()));
    for (const auto& original_array : original_arrays) {
      h = H::combine(std::move(h), original_array);
    }
    return h;
  }

 private:
  static TupleTree<std::optional<OriginalArray>>& EmptyOriginalValueTupleTree();

  void ClearInternalNodeValues();
  TupleTree<std::optional<OriginalArray>> tree_;
  // The optional call hierarchy that this OriginalValue is associated with.
  // This is only available if the OriginalValue is created from a call
  // instruction or a call-like instruction (e.g., kWhile). The format is
  // "<call_instruction_name>#$" where "#$" is a placeholder. If loop unrolling
  // happens, "#$" is replaced with the specific iteration count (e.g., "#0").
  // If a nested loop is hoisted during loop-invariant hoisting, the outerloop's
  // call hierarchy is prepended to the hoisted loop's call hierarchy but with
  // "#$" replaced with "#*". An empty string (std::string("")) indicates a
  // synthetic call.
  std::optional<std::string> call_hierarchy_;
};

}  // namespace xla
#endif  // XLA_HLO_IR_HLO_ORIGINAL_VALUE_H_
