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

#ifndef XLA_UTIL_INTERVAL_TREE_H_
#define XLA_UTIL_INTERVAL_TREE_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "third_party/gloop/util/intervaltree/intervaltree.h"

namespace xla {

template <typename T, typename = void>
struct has_to_string : std::false_type {};

template <typename T>
struct has_to_string<T, std::void_t<decltype(std::declval<T>().ToString())>>
    : std::is_convertible<decltype(std::declval<T>().ToString()), std::string> {
};

template <typename ValueType>
using IntervalTreeNode = ::IntervalNode<int64_t, ValueType>;

template <typename ValueType>
class IntervalTree {
  static_assert(has_to_string<ValueType>::value,
                "ValueType must have a ToString() member function returning "
                "std::string (or convertible to it)");

 public:
  void Add(int64_t start, int64_t end, const ValueType& value) {
    if (start <= end) {
      tree_.InsertVal(start, end, value);
    }
  }

  bool Remove(int64_t start, int64_t end, const ValueType& value) {
    if (start > end) {
      return false;
    }
    ::IntervalIterator<int64_t, ValueType> iter(&tree_, start, end,
                                                INTERVAL_SMALLEST);
    while (iter.Get() != nullptr) {
      if (iter.begin() == start && iter.end() == end && iter.value() == value) {
        iter.Delete();
        return true;
      }
      iter.Next();
    }
    return false;
  }

 protected:
  ::IntervalTree<int64_t, ValueType> tree_;
};

}  // namespace xla

#endif  // XLA_UTIL_INTERVAL_TREE_H_
