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

#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/testing_pipeline.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/stablehlo_to_hlo_to_stablehlo.h"

namespace xla {
namespace sdy {

namespace {

struct SdyRoundTripTestingPipelineOptions
    : public mlir::PassPipelineOptions<SdyRoundTripTestingPipelineOptions> {
  Option<bool> enableHloShardingV3{
      *this, "enable-hlo-sharding-v3",
      llvm::cl::desc("Whether to enable HloShardingV3 which is mesh and axis "
                     "based sharding representation."),
      llvm::cl::init(false)};
};

void sdyRoundTripTestingPipeline(
    mlir::OpPassManager& pm,
    const SdyRoundTripTestingPipelineOptions& options) {
  addSdyRoundTripExportPipeline(pm, /*keepMeshesInlined=*/false,
                                options.enableHloShardingV3);
  pm.addPass(createSdyRoundTripStablehloToHloToStablehloPass());
  addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/true,
                                /*importFuncCalls=*/true,
                                /*liftAndDedupMeshes=*/false,
                                options.enableHloShardingV3);
}

}  // namespace

void registerSdyRoundTripTestingPipeline() {
  mlir::PassPipelineRegistration<SdyRoundTripTestingPipelineOptions>(
      "xla-sdy-round-trip-testing-pipeline",
      "Run Shardy export pipeline, then convert to HLO, then convert to "
      "StableHLO, then import back to Shardy",
      sdyRoundTripTestingPipeline);
}

}  // namespace sdy
}  // namespace xla
