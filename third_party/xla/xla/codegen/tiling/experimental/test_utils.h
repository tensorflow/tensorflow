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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TEST_UTILS_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TEST_UTILS_H_

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"

namespace xla::gpu::experimental {

MATCHER_P(MatchString, tile_string, "") {
  const absl::string_view expected_string = tile_string;
  const std::string actual_string = arg.ToString();
  const auto [expected_index, actual_index] =
      FindApproximateMismatch(expected_string, actual_string);
  const bool matches = expected_index == expected_string.size() &&
                       actual_index == actual_string.size();
  if (!matches) {
    *result_listener << GetMismatchReport(expected_index, actual_index,
                                          tile_string, arg.ToString());
  }
  return matches;
}

Tile GetTestTile(const TilingSpace& tiling_space,
                 absl::Span<const int64_t> shape);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TEST_UTILS_H_
