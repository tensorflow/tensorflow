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

#ifndef XLA_TESTS_CONSTRAINT_STATE_H_
#define XLA_TESTS_CONSTRAINT_STATE_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"

namespace xla {

// Represents a continuous mathematical interval [min, max], optionally
// excluding exactly 0.0.
struct ConstraintInterval {
  static constexpr double kMax = std::numeric_limits<double>::max();
  static constexpr double kMin = -std::numeric_limits<double>::max();
  double min = kMin;
  double max = kMax;
  bool exclude_zero = false;

  ConstraintInterval(double min, double max, bool exclude_zero)
      : min(min), max(max), exclude_zero(exclude_zero) {}

  ConstraintInterval() = default;

  static ConstraintInterval Unconstrained() { return ConstraintInterval{}; }

  static ConstraintInterval Positive() {
    return ConstraintInterval{0.0, kMax, false};
  }

  static ConstraintInterval StrictPositive() {
    return ConstraintInterval{0.0, kMax, true};
  }

  static ConstraintInterval Negative() {
    return ConstraintInterval{kMin, 0.0, false};
  }

  static ConstraintInterval StrictNegative() {
    return ConstraintInterval{kMin, 0.0, true};
  }

  static ConstraintInterval NonZero() {
    return ConstraintInterval{kMin, kMax, true};
  }

  bool IsUnconstrained() const {
    return min == kMin && max == kMax && !exclude_zero;
  }

  bool IsEmpty() const {
    return min > max || (min == max && min == 0.0 && exclude_zero);
  }

  bool IsPositive() const { return min >= 0.0; }

  bool IsPositiveStrict() const {
    return min > 0.0 || (min == 0.0 && exclude_zero);
  }

  bool IsNegative() const { return max <= 0.0; }

  bool IsNegativeStrict() const {
    return max < 0.0 || (max == 0.0 && exclude_zero);
  }

  bool CrossesZero() const { return min < 0.0 && max > 0.0; }

  std::string ToString() const {
    if (IsEmpty()) {
      return "Empty";
    }
    if (IsUnconstrained()) {
      return "Unconstrained";
    }
    if (IsPositiveStrict()) {
      return "PositiveStrict";
    }
    if (IsNegativeStrict()) {
      return "NegativeStrict";
    }
    if (IsPositive()) {
      return "Positive";
    }
    if (IsNegative()) {
      return "Negative";
    }
    return absl::StrCat("[", min, ", ", max, "]",
                        exclude_zero ? " (excl 0)" : "");
  }

  bool operator==(const ConstraintInterval& other) const {
    if (IsEmpty() && other.IsEmpty()) {
      return true;
    }
    return min == other.min && max == other.max &&
           exclude_zero == other.exclude_zero;
  }

  bool operator!=(const ConstraintInterval& other) const {
    return !(*this == other);
  }

  // Returns the intersection of two intervals.
  // If the intersection is empty, returns an ConstraintInterval where IsEmpty()
  // is true.
  ConstraintInterval Intersect(const ConstraintInterval& other) const {
    if (IsEmpty() || other.IsEmpty()) {
      return ConstraintInterval{1.0, -1.0, false};  // Empty interval
    }

    ConstraintInterval result;
    result.min = std::max(min, other.min);
    result.max = std::min(max, other.max);
    result.exclude_zero = exclude_zero || other.exclude_zero;

    if (result.min > result.max) {
      return ConstraintInterval{1.0, -1.0, false};  // Empty interval
    }

    return result;
  }
};

struct StructuralConstraints {
  bool no_duplicates = false;
  bool needs_sorted_indices = false;
  std::optional<int64_t> alignment = std::nullopt;
  std::optional<uint64_t> known_zeroes_mask = std::nullopt;

  void Merge(const StructuralConstraints& other) {
    no_duplicates |= other.no_duplicates;
    needs_sorted_indices |= other.needs_sorted_indices;
    if (other.alignment) {
      alignment =
          std::max(alignment.value_or(*other.alignment), *other.alignment);
    }
    if (other.known_zeroes_mask) {
      known_zeroes_mask = known_zeroes_mask.value_or(*other.known_zeroes_mask) |
                          *other.known_zeroes_mask;
    }
  }

  bool operator==(const StructuralConstraints& other) const {
    return no_duplicates == other.no_duplicates &&
           needs_sorted_indices == other.needs_sorted_indices &&
           alignment == other.alignment &&
           known_zeroes_mask == other.known_zeroes_mask;
  }

  bool operator!=(const StructuralConstraints& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
    if (no_duplicates == false && needs_sorted_indices == false &&
        alignment == std::nullopt && known_zeroes_mask == std::nullopt) {
      return "None";
    }
    return absl::StrCat("no_duplicates: ", no_duplicates,
                        " needs_sorted_indices: ", needs_sorted_indices,
                        " alignment: ", alignment.value_or(-1),
                        " known_zeroes_mask: ", known_zeroes_mask.value_or(0));
  }
};

class ConstraintState {
 public:
  ConstraintState() = default;

  // Intersects with the interval. Returns false if the resulting interval
  // is empty.
  bool AddConstraint(const ConstraintInterval& constraint) {
    interval_ = interval_.Intersect(constraint);
    return !interval_.IsEmpty();
  }

  void MergeStructural(const StructuralConstraints& struct_cons) {
    structure_.Merge(struct_cons);
  }

  void Merge(const ConstraintState& other) {
    interval_ = interval_.Intersect(other.interval_);
    structure_.Merge(other.structure_);
  }

  ConstraintInterval GetConstraintInterval() const { return interval_; }

  StructuralConstraints GetStructuralConstraints() const { return structure_; }

  bool operator==(const ConstraintState& other) const {
    return interval_ == other.interval_ && structure_ == other.structure_;
  }

  bool operator!=(const ConstraintState& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
    return absl::StrCat("ConstraintInterval: ", interval_.ToString(),
                        " StructuralConstraints: ", structure_.ToString());
  }

 private:
  ConstraintInterval interval_;
  StructuralConstraints structure_;
};

}  // namespace xla

#endif  // XLA_TESTS_CONSTRAINT_STATE_H_
