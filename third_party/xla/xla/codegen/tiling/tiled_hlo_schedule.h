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
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"

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
  // `iteration_space` must contain at most one entry for each dimension id in
  // the discrete range {0, ..., tile_offsets_indexing.GetDimVarsCount() - 1}.
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
  //     of results of `tile_offsets_indexing` on the subspace defined by the
  //     parameter iteration space (i.e. the map may only reorder how the
  //     results are generated, but may not change the results themselves);
  virtual absl::StatusOr<IndexingMap> Schedule(
      const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
      mlir::MLIRContext* ctx) const = 0;
};

// The indexing map returned by this schedule iterates over the iteration space
// being specified in major-to-minor order (i.e. it first iterates over the
// trailing dimension of the iteration space and last over the leading
// dimension).
class MajorToMinorTiledHloSchedule : public TiledHloSchedule {
 public:
  absl::StatusOr<IndexingMap> Schedule(const IndexingMap& tile_offsets_indexing,
                                       IterationSpace iteration_space,
                                       mlir::MLIRContext* ctx) const override;
};

// Convenience function to produce a `MajorToMinorTiledHloSchedule` that
// can be passed to `SymbolicTileAnalysis::ComputeTiledComputation`.
absl::StatusOr<std::unique_ptr<TiledHloSchedule>>
CreateMajorToMinorTiledHloSchedule(
    const TilingSpecification& tiling_specification);

// Given a `TilingSpecification` where some of the output tile sizes are
// provided by a `dot` operation with one left-hand-side and one
// right-hand-side non-contracting dimensions, this schedule transposes the
// iteration pattern over these output dimensions.
//
// This schedule is only constructible when the underlying `TilingSpecification`
// contains a single `dot` node.
//
// TODO(b/417977182): this is implemented as a very bespoke pattern to unblock
// the launch of the generic emitter. We probably will want to subsume this with
// a more flexible approach for user-specified transposed schedules (that don't
// rely on the "dot" instruction being at the root).
class TransposedDotTiledHloSchedule : public TiledHloSchedule {
 public:
  absl::StatusOr<IndexingMap> Schedule(const IndexingMap& tile_offsets_indexing,
                                       IterationSpace iteration_space,
                                       mlir::MLIRContext* ctx) const override;

  static absl::StatusOr<std::unique_ptr<TransposedDotTiledHloSchedule>> Create(
      const TilingSpecification& tiling_specification);

 private:
  TransposedDotTiledHloSchedule(int64_t m_dim_id, int64_t n_dim_id)
      : m_dim_id_(m_dim_id), n_dim_id_(n_dim_id) {}

  // The index of the `m` dimension within the parameter mapping of the
  // `TilingSpecification`.
  int64_t m_dim_id_;
  // The index of the `n` dimension within the parameter mapping of the
  // `TilingSpecification`.
  int64_t n_dim_id_;
};

// TODO(b/417977182): implement the `PlanarSnakeTiledHloSchedule` schedule.

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILED_HLO_SCHEDULE_H_
