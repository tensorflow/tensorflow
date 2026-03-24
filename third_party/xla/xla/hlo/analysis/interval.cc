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

#include "xla/hlo/analysis/interval.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/strings/str_format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {

std::ostream& operator<<(std::ostream& out, const Interval& interval) {
  out << absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
  return out;
}

std::string Interval::ToString() const {
  return absl::StrFormat("[%d, %d]", lower, upper);
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Interval& interval) {
  os << absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
  return os;
}

int64_t Interval::GetLoopTripCount() const {
  if (!IsFeasible()) {
    return 0;
  }
  DCHECK((static_cast<absl::int128>(upper) - lower + 1) <=
         std::numeric_limits<int64_t>::max());
  return upper - lower + 1;
}

Interval::ComparisonResult Interval::Gt(const Interval& b) const {
  if (!IsFeasible() || !b.IsFeasible()) {
    return {std::nullopt};
  }
  if (lower > b.upper) {
    return {true};
  }
  if (upper <= b.lower) {
    return {false};
  }
  return {std::nullopt};
}

Interval::ComparisonResult Interval::Eq(const Interval& b) const {
  Interval intersection = Intersect(b);
  if (!intersection.IsFeasible()) {
    return {false};
  }
  if (intersection.IsPoint() && IsPoint() && b.IsPoint()) {
    return {true};
  }
  return {std::nullopt};
}

Interval Interval::operator+(const Interval& rhs) const {
  int64_t out_lower;
  int64_t out_upper;

  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  bool lower_overflow = llvm::AddOverflow(lower, rhs.lower, out_lower);
  bool upper_overflow = llvm::AddOverflow(upper, rhs.upper, out_upper);

  if (lower_overflow || lower == kMin || rhs.lower == kMin) {
    if (lower < 0 || rhs.lower < 0) {
      out_lower = kMin;
    } else {
      out_lower = kMax;
      out_upper = kMax;
    }
  }

  if (upper_overflow || upper == kMax || rhs.upper == kMax) {
    if (upper > 0 || rhs.upper > 0) {
      out_upper = kMax;
    } else {
      out_upper = kMin;
      out_lower = kMin;
    }
  }

  return {out_lower, out_upper};
}

Interval Interval::operator*(const Interval& rhs) const {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  auto mul = [&](int64_t p) {
    int64_t l = lower;
    int64_t u = upper;
    if (p < 0) {
      std::swap(l, u);
    }
    int64_t out_lower;
    int64_t out_upper;
    if (llvm::MulOverflow(l, p, out_lower) ||
        // -1 * max is min + 1, and doesn't overflow. We consider max a
        // special sentinel value, so the result should be min (= saturated).
        (p == -1 && l == kMax)) {
      out_lower = kMin;
    }
    if (llvm::MulOverflow(u, p, out_upper)) {
      out_upper = kMax;
    }
    return Interval{out_lower, out_upper};
  };

  return mul(rhs.lower).Union(mul(rhs.upper));
}

Interval Interval::operator-() const {
  int64_t ub = lower == std::numeric_limits<int64_t>::min()
                   ? std::numeric_limits<int64_t>::max()
                   : -lower;
  int64_t lb = upper == std::numeric_limits<int64_t>::max()
                   ? std::numeric_limits<int64_t>::min()
                   : -upper;
  return Interval{lb, ub};
}

Interval Interval::FloorDiv(int64_t rhs) const {
  auto saturate_div = [](int64_t lhs, int64_t rhs) {
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (lhs == kMin) {
      return rhs > 0 ? kMin : kMax;
    }
    if (lhs == kMax) {
      return rhs > 0 ? kMax : kMin;
    }
    return llvm::divideFloorSigned(lhs, rhs);
  };

  int64_t a = saturate_div(lower, rhs);
  int64_t b = saturate_div(upper, rhs);
  return {std::min(a, b), std::max(a, b)};
}

}  // namespace xla
