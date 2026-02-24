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

#include <gmock/gmock.h>
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"

namespace xla::gpu::experimental {

MATCHER_P(MatchString, symbolic_tile_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(symbolic_tile_string, arg.ToString()),
      result_listener);
}

SymbolicTile GetTestSymbolicTile(const TilingSpace& tiling_space,
                                 absl::Span<const int64_t> shape);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TEST_UTILS_H_
