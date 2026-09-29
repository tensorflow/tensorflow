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

#ifndef XLA_HLO_UTILS_SORT_UTILS_H_
#define XLA_HLO_UTILS_SORT_UTILS_H_

#include <cstdint>
#include <utility>

#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

// Returns the parameter numbers (index0, index1) used in a simple comparator
// (compare(param0, param1) with PARTIAL or TOTAL order). Excludes WEAK order,
// which is matched by MatchNumpySortComparator instead.
// Note: TOTAL order is always used for integer comparisons, and on floats it
// matches GPU CUB radix sort's native kDefaultOrder (IEEE-754 totalOrder).
// Callers that map simple float comparators to NumPy order (such as CPU
// SortThunk) must explicitly reject TOTAL order on floating-point types.
// Returns (-1, -1) if not matched.
std::pair<int64_t, int64_t> MatchSimpleSortComparator(
    const HloCompareInstruction* compare);

// Returns the parameter numbers (index0, index1) used in a comparator for
// NumPy sort order (either a direct WEAK order parameter comparison or the
// canonicalized zero/NaN pattern). Returns (-1, -1) if the comparison is not a
// NumPy sort comparator.
std::pair<int64_t, int64_t> MatchNumpySortComparator(
    const HloCompareInstruction* compare);

}  // namespace xla

#endif  // XLA_HLO_UTILS_SORT_UTILS_H_
