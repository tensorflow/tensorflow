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

#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_callback_custom_calls.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_manual_reduction_collectives.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/shard_map_export.h"

namespace xla {
namespace sdy {

void addStablehloExportPipeline(mlir::OpPassManager& pm,
                                const StablehloExportPipelineOptions& options) {
  pm.addPass(createStablehloExportManualReductionCollectivesPass());
  // This pass converts `sdy.constant` (which isn't foldable) into
  // `stablehlo.constant` (which is foldable), therefore greedy pattern
  // rewriters shouldn't be applied before converting to HLO as they apply
  // folding.
  pm.addPass(createExportOpsPass(options.keepHloShardingConstraints));
  pm.addPass(createExportNamedComputationsPass(options.dedupFunctionsFully));
  pm.addPass(createStablehloRoundTripShardMapExportPass(
      options.keepHloShardingConstraints));
  // If we don't add a sharding to a control flow op without one,
  // StableHLO -> HLO conversion won't add a sharding for that op even if a
  // free variable that has a sharding is lifted as an additional result, and in
  // effect the op will have a replicated sharding for all results.
  pm.addPass(createExportStablehloShardingsPass(
      /*addMissingShardingToControlFlow=*/options
          .addMissingShardingToControlFlow,
      /*enableHloShardingV3=*/options.enableHloShardingV3));
  pm.addPass(createStablehloRoundTripExportCallbackCustomCallsPass());
}

namespace {

void stablehloExportPipeline(mlir::OpPassManager& pm,
                             const StablehloExportPipelineOptions& options) {
  addStablehloExportPipeline(pm, options);
}

}  // namespace

void registerStablehloExportPipeline() {
  mlir::PassPipelineRegistration<StablehloExportPipelineOptions> exportPipeline(
      "xla-sdy-stablehlo-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an StableHLO module, "
      "which is ready for StableHLO -> HLO conversion.",
      stablehloExportPipeline);
}

}  // namespace sdy
}  // namespace xla
