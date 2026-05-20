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

#ifndef XLA_TSL_LIB_MONITORING_LABEL_ARRAY_UTILS_H_
#define XLA_TSL_LIB_MONITORING_LABEL_ARRAY_UTILS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace tsl {
namespace monitoring {

struct LabelArrayHash {
  using is_transparent = std::true_type;

  template <typename ArrayType>
  size_t operator()(const ArrayType& label_array) const {
    return absl::HashOf(label_array);
  }
};

struct LabelArrayEq {
  using is_transparent = std::true_type;

  template <typename LHS, typename RHS>
  bool operator()(const LHS& lhs, const RHS& rhs) const {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }
};

// We need pointer stability for values, but not for keys. `absl::flat_hash_map`
// doesn't use pointer indirection for its keys or its values, but we can still
// achieve pointer stability for values by wrapping them in `std::unique_ptr`.
// This middle ground avoids an unnecessary level of indirection when accessing
// keys, whereas `std::unordered_map` indirectly accesses both keys and values.
template <typename ValueType, size_t NumLabels>
using LabelArrayMap = absl::flat_hash_map<std::array<std::string, NumLabels>,
                                          std::unique_ptr<ValueType>,
                                          LabelArrayHash, LabelArrayEq>;

}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_LABEL_ARRAY_UTILS_H_
