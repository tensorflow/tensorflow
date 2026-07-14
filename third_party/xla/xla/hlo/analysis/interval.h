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

#ifndef XLA_HLO_ANALYSIS_INTERVAL_H_
#define XLA_HLO_ANALYSIS_INTERVAL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {

// Interval represents a closed interval [lower_bound, upper_bound].
struct Interval {
  std::string ToString() const;
  bool IsPoint() const { return lower == upper; }
  bool IsFeasible() const { return lower <= upper; }

  // Returns the number of elements in the interval. Asserts that the number of
  // elements fits in an int64_t. For this reason, this should only be used for
  // intervals corresponding to symbols, not for general intervals. Use
  // `IsFeasible` to check if the interval is non-empty.
  int64_t GetLoopTripCount() const;

  bool Contains(int64_t value) const {
    return value >= lower && value <= upper;
  }

  // Returns true if this interval contains the entire other interval.
  bool Contains(Interval other) const { return Intersect(other) == other; }

  // The result of a range comparison. We wrap std::optional in a struct to
  // avoid accidental implicit conversion to bool:
  // if (range < 42) {
  //   Executed if the result of the comparison is known to be false!
  // }
  struct ComparisonResult {
    // true or false if the result is known, nullopt otherwise.
    std::optional<bool> result;

    ComparisonResult operator!() const {
      if (result) return {!*result};
      return {result};
    }
    bool operator==(const ComparisonResult& other) const {
      return result == other.result;
    }
    bool operator==(bool other) const { return result && *result == other; }
    bool operator==(std::nullopt_t) const { return !result; }
    bool operator!=(std::nullopt_t) const { return result.has_value(); }
    bool operator*() const { return *result; }
  };

  // All comparison operators here return true or false if the result is known,
  // or nullopt if it may be either true or false.
  // We don't use operators here, because the "==" used for hashing is not the
  // same as "Eq".
  ComparisonResult Gt(const Interval& b) const;
  ComparisonResult Lt(const Interval& b) const { return b.Gt(*this); }
  ComparisonResult Ge(const Interval& b) const { return !b.Gt(*this); }
  ComparisonResult Le(const Interval& b) const { return !this->Gt(b); }
  // This is not the same as "==".  See the implementations.
  ComparisonResult Eq(const Interval& b) const;
  // This is not the same as "!=".  See the implementations.
  ComparisonResult Ne(const Interval& b) const { return !this->Eq(b); }

  Interval Intersect(const Interval& rhs) const {
    Interval result{std::max(lower, rhs.lower), std::min(upper, rhs.upper)};
    if (result.upper < result.lower) {
      // Normalize empty results such that NumElements returns 0.
      result.upper = result.lower - 1;
    }
    return result;
  }

  Interval Union(const Interval& rhs) const {
    return {std::min(lower, rhs.lower), std::max(upper, rhs.upper)};
  }

  // Computes the range of the sum of the two intervals. Implements saturating
  // semantics (i.e. overflow and underflow get clamped to the maximum and
  // minimum int64). Additionally, bounds of the minimum/maximum value are
  // considered to be possibly saturated, i.e. `{-2 ** 63, 0} + {42, 42}`
  // returns `{-2 ** 63, 42}`, not `{-2 ** 63 + 42, 42}`.
  Interval operator+(const Interval& rhs) const;
  // Computes the range of the product of the two intervals. Implements
  // saturating semantics.
  Interval operator*(const Interval& rhs) const;
  // Computes the range of the difference of the two intervals. Implements
  // saturating semantics.
  Interval operator-(const Interval& rhs) const { return *this + (-rhs); }
  Interval operator-() const;
  Interval FloorDiv(int64_t rhs) const;

  Interval min(const Interval& rhs) const {
    return {std::min(lower, rhs.lower), std::min(upper, rhs.upper)};
  }

  Interval max(const Interval& rhs) const {
    return {std::max(lower, rhs.lower), std::max(upper, rhs.upper)};
  }

  // This is not the same as "Eq".  See the implementations.
  bool operator==(const Interval& rhs) const {
    return lower == rhs.lower && upper == rhs.upper;
  }
  // This is not the same as "Ne".  See the implementations.
  bool operator!=(const Interval& rhs) const { return !(*this == rhs); }

  int64_t lower = 0;
  int64_t upper = 0;
};

std::ostream& operator<<(std::ostream& out, const Interval& interval);
inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Interval& interval);

template <typename H>
H AbslHashValue(H h, const Interval& range) {
  return H::combine(std::move(h), range.lower, range.upper);
}

// For use in llvm::hash_combine.
inline size_t hash_value(const Interval& range) {
  return llvm::hash_combine(range.lower, range.upper);
}

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_INTERVAL_H_
