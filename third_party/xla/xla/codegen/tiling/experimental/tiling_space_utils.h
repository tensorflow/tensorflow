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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_UTILS_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xla {

// A sequence of tile sizes.
//
// This is an inlined vector to avoid too many heap allocations.
using FlatTiling = absl::InlinedVector<int64_t, 4>;

absl::StatusOr<std::vector<FlatTiling>> GetFlatTilingsForInputSpace(
    absl::Span<const int64_t> input_space);

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_UTILS_H_
