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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_

#include <cstddef>
#include <optional>

#include "absl/container/flat_hash_map.h"

// A absl based bidirectional map with optional associated properties.
// Note that removal is not supported.
template <typename LeftT, typename RightT, typename PropsT>
struct BidirectionalMap {
  struct ForwardMapping {
    RightT node;
    std::optional<PropsT> props;
  };

  // Forward and reverse maps.
  absl::flat_hash_map<LeftT, ForwardMapping> left;
  absl::flat_hash_map<RightT, LeftT> right;

  // Inserts a new mapping with properties.
  // Returns true if insertion was successful.
  bool Insert(LeftT left_element, RightT right_element, PropsT props) {
    if (left.contains(left_element) || right.contains(right_element)) {
      return false;
    }

    left[left_element] = {right_element, props};
    right[right_element] = left_element;
    return true;
  }

  // Inserts a new mapping without properties.
  // Returns true if insertion was successful.
  bool Insert(LeftT left_element, RightT right_element) {
    if (left.contains(left_element) || right.contains(right_element)) {
      return false;
    }

    left[left_element] = {right_element, std::nullopt};
    right[right_element] = left_element;
    return true;
  }

  // Returns the number of mappings.
  size_t size() const { return left.size(); }
};

#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_BIDIRECTIONAL_MAP_H_
