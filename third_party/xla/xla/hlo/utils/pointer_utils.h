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

#ifndef XLA_HLO_UTILS_POINTER_UTILS_H_
#define XLA_HLO_UTILS_POINTER_UTILS_H_

#include <cstddef>
#include <memory>

#include "absl/hash/hash.h"

namespace xla {
// Hash functor for std::shared_ptr<T>, hashing the value of the pointee.
template <typename T>
struct PointeeHash {
  size_t operator()(const std::shared_ptr<T>& value) const {
    if (!value) {
      return absl::Hash<int>()(0);  // Hash for nullptr
    }
    return absl::Hash<T>()(*value);
  }
};

// Equality functor for std::shared_ptr<T>, comparing the values of the
// pointees.
template <typename T>
struct PointeeEqual {
  bool operator()(const std::shared_ptr<T>& lhs,
                  const std::shared_ptr<T>& rhs) const {
    if (!lhs && !rhs) {
      return true;
    }
    if (!lhs || !rhs) {
      return false;
    }
    return *lhs == *rhs;
  }
};
}  // namespace xla

#endif  // XLA_HLO_UTILS_POINTER_UTILS_H_
