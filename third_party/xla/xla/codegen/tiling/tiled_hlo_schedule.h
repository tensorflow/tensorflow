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

#ifndef XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_
#define XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {

// Helper data structure to compute a schedule.
//
// TODO(b/422676780): When rewriting to use the new tiling space instead of a
// tiling specification, we should be able to easily replace this with a
// `TilingSpace::DimensionInfo` where the dimension size corresponds to the
// iteration space over tiles.
struct DimensionInfo {
  // An identifier for the dimension.
  int64_t dimension_id;
  // The size of the iteration space of the dimension.
  int64_t dimension_size;
};

using IterationSpace = absl::Span<const DimensionInfo>;

// A `TiledHloSchedule` exposes methods for scheduling a `TiledHloComputation`,
// i.e. it specifies an iteration order over tiles.
class TiledHloSchedule {
 public:
  virtual ~TiledHloSchedule() = default;

  // Returns a schedule for the given root instruction as an indexing map.
  //
  // `iteration_space` must contain one entry for each dimension id in the
  // discrete range {0, ..., iteration_space.size() - 1}, and the number of
  // dimensions in `iteration_space` must match the number of dimension
  // parameters in `tile_offsets_indexing`.
  //
  // We unfortunately can't pass a `TilingSpecification` here directly in order
  // to handle assumption-breaking calls in the case of multi-output fusions.
  // Once those are resolved, using a `TilingSpecification` (or new
  // `TilingSpace`) should be possible and preferable.
  //
  // The resulting indexing map must satisfy the following properties:
  // (1) the map must have a single input whose range of values is the size of
  //     the iteration space (i.e. the product of `iteration_space`'s
  //     `dimension_size`s);
  // (2) the set of results generatable with the map must be equal to the set
  //     of results of `tile_offsets_indexing` (i.e. the map may only reorder
  //     how the results are generated, but may not change the results
  //     themselves);
  virtual absl::StatusOr<IndexingMap> Schedule(
      const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
      gpu::SymbolicExprContext* symbolic_expr_context) const = 0;
};

// The indexing map returned by this schedule iterates over the iteration space
// being specified in major-to-minor order (i.e. it first iterates over the
// trailing dimension of the iteration space and last over the leading
// dimension).
class MajorToMinorTiledHloSchedule : public TiledHloSchedule {
 public:
  absl::StatusOr<IndexingMap> Schedule(
      const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
      gpu::SymbolicExprContext* symbolic_expr_context) const override;
};

// TODO(b/417977182): implement the `PlanarSnakeTiledHloSchedule` schedule.

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_
