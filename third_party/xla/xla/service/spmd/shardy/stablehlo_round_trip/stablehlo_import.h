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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_IMPORT_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_IMPORT_H_

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"

namespace xla {
namespace sdy {

// Parses `sharding` to obtain a `xla::HloSharding`.
xla::HloSharding parseShardingFromString(const mlir::StringAttr& sharding);

// Converts `hloSharding` into a `TensorShardingAttr` based on the
// `globalMesh`.
//
// If `hloSharding` is unknown, return fully open sharding. Otherwise, the
// returned sharding is open iff `openDims` is true.
mlir::sdy::TensorShardingAttr convertToSdySharding(
    const xla::HloSharding& hloSharding, mlir::sdy::MeshAttr globalMesh,
    const llvm::SmallDenseMap<int64_t, mlir::StringRef>&
        deviceIdToMaximalMeshName,
    int64_t rank, bool openDims = false);

// Returns the axis sizes from the tile assignment. For example, given the input
// {devices=[6,35]<=[7,10,3]T(2,1,0)}, the function returns [7, 2, 5, 3].
mlir::SmallVector<int64_t> getAxisSizes(const TileAssignment& tileAssignment);

// Register the xla-sdy-import-shardings pass.
void registerStablehloImportShardingsPass();

// Register the xla-sdy-stablehlo-import-pipeline.
void registerStablehloImportPipeline();

// Add the xla-sdy-stablehlo-import-pipeline in `pm`. The pipeline, including a
// sequence of passes, imports a StableHLO module into the SDY (Shardonnay)
// dialect.
//
// `allowPropagationToArgs` and `allowPropagationToResults` indicate for each
// argument and result of the main function respectively, whether their existing
// sharding can be modified during propagation, i.e., should their dimension
// shardings be open. Each vector can either:
// - be empty, in which case the default is false for all args/results.
// - have a single element, in which case the value applies to all args/results.
// - have the same number of elements as the number of args/results.
void addStablehloImportPipeline(mlir::OpPassManager& pm,
                                mlir::ArrayRef<bool> allowPropagationToArgs,
                                mlir::ArrayRef<bool> allowPropagationToResults);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_IMPORT_H_
