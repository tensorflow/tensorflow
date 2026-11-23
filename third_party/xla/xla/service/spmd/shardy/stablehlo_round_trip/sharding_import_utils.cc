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

#include "xla/service/spmd/shardy/stablehlo_round_trip/sharding_import_utils.h"

#include <cstdint>
#include <iterator>
#include <numeric>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/util.h"

namespace xla {
namespace sdy {

absl::StatusOr<TileFactorizationResult> factorizeTileAssignmentToSdy(
    const xla::AnalyzeTileAssignmentResult& tileInfo, mlir::sdy::MeshAttr mesh,
    int64_t rank, ShardingDimClosedness dimClosedness) {
  int64_t localMeshSize = xla::Product(tileInfo.local_mesh);
  if (localMeshSize != mesh.getTotalSize()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "sharding local mesh size (", localMeshSize,
        ") does not match mesh total size (", mesh.getTotalSize(), ")"));
  }

  mlir::MLIRContext* ctx = mesh.getContext();

  // 1. Create a mapping from local axis to global axis refs.
  llvm::SmallVector<llvm::SmallVector<mlir::sdy::AxisRefAttr>>
      localAxisIndexToGlobalAxes;
  localAxisIndexToGlobalAxes.reserve(tileInfo.local_mesh.size());
  llvm::ArrayRef<mlir::sdy::MeshAxisAttr> remainingGlobalAxes = mesh.getAxes();
  int64_t globalAxisPreSize = 1;
  for (int64_t localAxisRemainingSize : tileInfo.local_mesh) {
    llvm::SmallVector<mlir::sdy::AxisRefAttr>& globalAxes =
        localAxisIndexToGlobalAxes.emplace_back();
    // The local axis size can correspond to multiple global axes or sub-axes.
    while (localAxisRemainingSize > 1) {
      if (remainingGlobalAxes.empty()) {
        return absl::InvalidArgumentError(
            "sharding local mesh exceeds mesh axes");
      }
      int64_t globalAxisRemainingSize =
          remainingGlobalAxes.front().getSize() / globalAxisPreSize;
      if (globalAxisRemainingSize == 1) {
        remainingGlobalAxes = remainingGlobalAxes.drop_front();
        globalAxisPreSize = 1;
        continue;
      }
      int64_t gcd = std::gcd(localAxisRemainingSize, globalAxisRemainingSize);
      if (gcd == 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "incompatible local and global axis sizes: ",
            localAxisRemainingSize, " vs ", globalAxisRemainingSize));
      }
      llvm::StringRef globalAxisName = remainingGlobalAxes.front().getName();
      if (gcd == globalAxisRemainingSize && globalAxisPreSize == 1) {
        // The full global axis is used.
        globalAxes.push_back(mlir::sdy::AxisRefAttr::get(ctx, globalAxisName));
      } else {
        // We use a sub-axis of the global axis.
        globalAxes.push_back(mlir::sdy::AxisRefAttr::get(
            ctx, globalAxisName, globalAxisPreSize, gcd));
      }
      globalAxisPreSize *= gcd;
      localAxisRemainingSize /= gcd;
    }
  }

  // 2. Create a mapping from dim and nested sub-dim to local axis index.
  llvm::SmallVector<llvm::SmallVector<int64_t>> dimToSubDimToLocalAxisIndex(
      rank);
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(tileInfo.sub_dims)) {
    if (subDimInfo.tile_dim_index >= rank) {
      // Subgroup dimension (e.g., replicated or unreduced).
      continue;
    }
    llvm::SmallVector<int64_t>& subDimToLocalAxisIndex =
        dimToSubDimToLocalAxisIndex[subDimInfo.tile_dim_index];
    if (subDimInfo.tile_sub_dim_index >=
        static_cast<int64_t>(subDimToLocalAxisIndex.size())) {
      subDimToLocalAxisIndex.resize(subDimInfo.tile_sub_dim_index + 1);
    }
    subDimToLocalAxisIndex[subDimInfo.tile_sub_dim_index] = localAxisIndex;
  }

  // 3. Reconstruct major-to-minor dimension shardings from minor-to-major
  // sub-dimensions.
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
  dimShardings.reserve(rank);
  for (llvm::ArrayRef<int64_t> subDimToLocalAxisIndex :
       dimToSubDimToLocalAxisIndex) {
    llvm::SmallVector<mlir::sdy::AxisRefAttr> axes;
    for (int64_t localAxisIndex : llvm::reverse(subDimToLocalAxisIndex)) {
      absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                   std::back_inserter(axes));
    }
    dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        ctx, axes,
        /*isClosed=*/dimClosedness == ShardingDimClosedness::kClosed));
  }

  return TileFactorizationResult{std::move(localAxisIndexToGlobalAxes),
                                 std::move(dimShardings)};
}

}  // namespace sdy
}  // namespace xla
