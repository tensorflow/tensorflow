// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_LAYOUT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <optional>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_layout.h"

namespace litert {

// Small standalone helper functions for working with
// c layout api.

static constexpr size_t kTensorMaxRank = LITERT_TENSOR_MAX_RANK;

// Build layout from given iterator of dimensions.
template <class Begin, class End>
inline constexpr LiteRtLayout BuildLayout(Begin begin, End end,
                                          const uint32_t* strides = nullptr) {
  LiteRtLayout res{static_cast<uint32_t>(end - begin), {}, strides};
  auto i = 0;

  for (auto* it = begin; it < end && i < kTensorMaxRank; ++it) {
    res.dimensions[i] = *it;
    ++i;
  }

  return res;
}

// Build layout from given iterable of dimensions.
template <class Dims>
inline constexpr LiteRtLayout BuildLayout(const Dims& dims,
                                          const uint32_t* strides = nullptr) {
  return BuildLayout(std::cbegin(dims), std::cend(dims), strides);
}

// Build layout from literal dimensions.
inline constexpr LiteRtLayout BuildLayout(std::initializer_list<int32_t> dims,
                                          const uint32_t* strides = nullptr) {
  return BuildLayout(dims.begin(), dims.end(), strides);
}

// Compute the number of elements in dims iterator. Nullopt if there exists
// a dynamic dimension.
template <class Begin, class End>
inline constexpr std::optional<size_t> NumElements(Begin begin, End end) {
  if (end - begin == 0) {
    return {};
  }
  size_t res = 1;
  for (auto* it = begin; it < end; ++it) {
    if (*it < 0) {
      return {};
    }
    res *= *it;
  }
  return res;
}

// Override for layouts.
inline constexpr std::optional<size_t> NumElements(const LiteRtLayout& layout) {
  auto* b = std::cbegin(layout.dimensions);
  return NumElements(b, b + layout.rank);
}

// Get dims as span.
inline constexpr absl::Span<const int32_t> DimsSpan(
    const LiteRtLayout& layout) {
  return absl::MakeConstSpan(layout.dimensions, layout.rank);
}

// Get strides as span if they exist.
inline constexpr std::optional<absl::Span<const uint32_t>> StridesSpan(
    const LiteRtLayout& layout) {
  if (layout.strides) {
    return absl::MakeConstSpan(layout.strides, layout.rank);
  }
  return {};
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_LAYOUT_H_
