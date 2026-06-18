/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_UTILS_HLO_CONTAINER_UTIL_H_
#define XLA_HLO_UTILS_HLO_CONTAINER_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"

namespace xla {

// Returns the indices for sorted `data`.
template <typename T>
std::vector<int64_t> ArgSort(absl::Span<const T> data) {
  std::vector<int64_t> indices(data.size());
  absl::c_iota(indices, 0);
  absl::c_sort(indices,
               [&data](int64_t i, int64_t j) { return data[i] < data[j]; });
  return indices;
}

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_CONTAINER_UTIL_H_
