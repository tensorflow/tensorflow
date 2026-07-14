/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_UTIL_SORTED_RANGE_H_
#define XLA_TSL_UTIL_SORTED_RANGE_H_

#include <functional>

#include "xla/tsl/util/sorted_range_internal.h"

namespace tsl {

// Returns a sorted view of the elements of a sized range. It does not copy or
// own any of the data from the underlying sized range, and is only valid while
// that sized range remains in scope and unmodified.
//
// Example usage:
//   std::vector<int> v = {3, 1, 4, 1, 5};
//   for (int x : SortedRange(v)) {
//     // Iterates in order: 1, 1, 3, 4, 5
//   }
template <typename Range, typename Compare = std::less<>>
auto SortedRange(Range& range, Compare cmp = Compare()) {
  return internal_sorted_range::SortedRangeImpl<Range, Compare>(range, cmp);
}

// Returns a view over the elements of a sized associative container
// (e.g. a map) sorted by key. It does not copy or own any of the data from the
// underlying container, and is only valid while that container remains in scope
// and unmodified.
//
// This is a simpler but less versatile OSS friendly version of the internal
// KeySortedRange.
//
// Particularly useful for iterating over a hash map deterministically.
//
// Example usage:
//   absl::flat_hash_map<std::string, int> m = {{"c", 3}, {"a", 1}, {"b", 2}};
//   for (const auto& [key, value] : KeySortedRange(m)) {
//     // Iterates in order by key: {"a", 1}, {"b", 2}, {"c", 3}
//   }
template <typename Range>
auto KeySortedRange(Range& range) {
  auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
  return internal_sorted_range::SortedRangeImpl<Range, decltype(cmp)>(range,
                                                                      cmp);
}

}  // namespace tsl

#endif  // XLA_TSL_UTIL_SORTED_RANGE_H_
