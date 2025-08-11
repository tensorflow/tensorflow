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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_PIPELINES_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_PIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace xla {
namespace sdy {

// Add the xla-sdy-round-trip-export-pipeline in `pm`. The pipeline,
// including a sequence of passes, exports the Shardy dialect into an StableHLO
// module with no XLA shardings, but SDY shardings and meshes saved as string
// frontend attributes.
//
// This is meant for temporarily saving the Shardy attrs/meshes in order to
// run some HLO passes before coming back to Shardy later in the XLA
// pipeline to run propagation. Should only be used for frontend frameworks like
// JAX to integrate with Shardy while the Shardy team works on a more
// long-term solution moving the HLO passes either after propagation or into
// MLIR (see b/335666088). So this pass will eventually be removed.
//
// If `keepMeshesInlined` is true, the pipeline will not lift inlined meshes.
void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm,
                                   bool keepMeshesInlined = false);

// Add the xla-sdy-round-trip-import-pipeline in `pm`. The pipeline,
// including a sequence of passes, imports an StableHLO module into the
// SDY (Shardy) dialect.
//
// The module is assumed to have `HloSharding::kShardingFrontendAttrName` and
// `kMeshesRoundTripAttr`.
void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport = true,
                                   bool importOnlyUninlineableFuncCalls = true,
                                   bool liftAndDedupMeshes = false);

// Register the xla-sdy-round-trip-export-pipeline.
void registerSdyRoundTripExportPipeline();

// Register the xla-sdy-round-trip-import-pipeline.
void registerSdyRoundTripImportPipeline();

// Register the xla-sdy-round-trip-testing-pipeline.
// This takes an SDY module, exports it to StableHLO while saving the SDY attrs
// and meshes, goes to HLO, back to StableHLO, and then back to SDY.
// This is for testing roundtripping SDY modules, but should be eventually
// removed as part of b/335666088.
void registerSdyRoundTripTestingPipeline();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_PIPELINES_H_
