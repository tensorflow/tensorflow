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

#include "xla/codegen/tiling/experimental/tiling_space_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

namespace {

// The possible tiles sizes for one dimension.
absl::StatusOr<std::vector<int64_t>> PossibleTileSizesForOneDimension(
    int64_t dim_size) {
  if (dim_size < 0) {
    return absl::InvalidArgumentError("Dimension size must be non-negative.");
  }
  std::vector<int64_t> result;
  if (dim_size == 0) {
    result.push_back(0);
    return result;
  }

  result.reserve(absl::bit_width(static_cast<uint64_t>(dim_size)));
  for (int64_t tile_size = 1; tile_size < dim_size; tile_size *= 2) {
    result.push_back(tile_size);
  }
  result.push_back(dim_size);
  return result;
}

}  // namespace

absl::StatusOr<std::vector<FlatTiling>> GetFlatTilingsForInputSpace(
    absl::Span<const int64_t> input_space) {
  std::vector<FlatTiling> flat_tilings;
  flat_tilings.push_back({});
  for (int64_t parameter_size : input_space) {
    ASSIGN_OR_RETURN(std::vector<int64_t> possible_tile_sizes,
                     PossibleTileSizesForOneDimension(parameter_size));
    std::vector<FlatTiling> extended_tilings;
    extended_tilings.reserve(flat_tilings.size() * possible_tile_sizes.size());
    for (const FlatTiling& flat_tile_sizes : flat_tilings) {
      for (int64_t tile_size : possible_tile_sizes) {
        FlatTiling extended_tiling = flat_tile_sizes;
        extended_tiling.push_back(tile_size);
        extended_tilings.push_back(extended_tiling);
      }
    }
    flat_tilings = std::move(extended_tilings);
  }

  return flat_tilings;
}

}  // namespace xla
