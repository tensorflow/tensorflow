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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_SHARDING_IMPORT_UTILS_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_SHARDING_IMPORT_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/tile_assignment.h"

namespace xla {
namespace sdy {

// Holds the result of factorizing an `AnalyzeTileAssignmentResult` onto a
// Shardy `MeshAttr`.
struct TileFactorizationResult {
  mlir::SmallVector<mlir::SmallVector<mlir::sdy::AxisRefAttr>>
      localAxisIndexToGlobalAxes;
  mlir::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
};

// Controls whether reconstructed `DimensionShardingAttr`s are open or closed.
enum class ShardingDimClosedness { kOpen, kClosed };

// Factorizes `tileInfo`'s local tile mesh and sub-dimensions across `mesh`'s
// axes, returning both the per-local-axis mapping to global mesh axis refs and
// the reconstructed `DimensionShardingAttr`s for a tensor of `rank` dimensions.
absl::StatusOr<TileFactorizationResult> factorizeTileAssignmentToSdy(
    const xla::AnalyzeTileAssignmentResult& tileInfo, mlir::sdy::MeshAttr mesh,
    int64_t rank, ShardingDimClosedness dimClosedness);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_SHARDING_IMPORT_UTILS_H_
