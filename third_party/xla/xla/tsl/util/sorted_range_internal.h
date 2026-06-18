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

#ifndef XLA_TSL_UTIL_SORTED_RANGE_INTERNAL_H_
#define XLA_TSL_UTIL_SORTED_RANGE_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

#include "absl/container/fixed_array.h"

namespace tsl {
namespace internal_sorted_range {

template <typename Range, typename Compare>
class SortedRangeImpl {
  using Pointer = decltype(&*std::begin(std::declval<Range&>()));

 public:
  SortedRangeImpl(Range& range, Compare cmp) : elements_(range.size()) {
    size_t i = 0;
    for (auto& elem : range) {
      elements_[i++] = &elem;
    }
    std::sort(elements_.begin(), elements_.end(),
              [&cmp](Pointer a, Pointer b) { return cmp(*a, *b); });
  }

  struct Iterator {
    typename absl::FixedArray<Pointer>::const_iterator it;

    bool operator==(const Iterator& other) const { return it == other.it; }
    bool operator!=(const Iterator& other) const { return it != other.it; }

    Iterator& operator++() {
      ++it;
      return *this;
    }
    decltype(auto) operator*() const { return **it; }
  };

  Iterator begin() const { return Iterator{elements_.begin()}; }
  Iterator end() const { return Iterator{elements_.end()}; }

 private:
  absl::FixedArray<Pointer> elements_;
};

}  // namespace internal_sorted_range
}  // namespace tsl

#endif  // XLA_TSL_UTIL_SORTED_RANGE_INTERNAL_H_
