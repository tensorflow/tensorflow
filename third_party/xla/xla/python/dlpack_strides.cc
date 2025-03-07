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

#include "xla/python/dlpack_strides.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {

absl::StatusOr<std::vector<int64_t>> HandleUnitDimensions(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  int64_t stride = 1;
  for (std::size_t i = dims.size(); i > 0; --i) {
    if (dims[i - 1] > 1) {
      if (strides[i - 1] < stride) {
        return Unimplemented("Not row-major.");
      }
      stride = strides[i - 1];
    }
  }
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  std::reverse(minor_to_major.begin(), minor_to_major.end());
  return minor_to_major;
}

absl::StatusOr<std::vector<int64_t>> StridesToLayout(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  CHECK_EQ(dims.size(), strides.size());
  if (dims.empty()) {
    return std::vector<int64_t>();
  }

  // A special case: if any dimension has size 1, then the stride in that
  // dimension is arbitrary. If all the other dimensions are row-major, then
  // we choose to return the full row-major layout.
  if (ABSL_PREDICT_FALSE(
          absl::c_any_of(dims, [](int64_t d) { return d <= 1; }))) {
    auto maybe_minor_to_major = HandleUnitDimensions(dims, strides);
    if (maybe_minor_to_major.ok()) {
      return maybe_minor_to_major.value();
    }
  }

  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });

  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

}  // namespace xla
