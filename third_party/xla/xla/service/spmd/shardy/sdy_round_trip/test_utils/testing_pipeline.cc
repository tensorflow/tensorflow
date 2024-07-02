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

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/mhlo_to_hlo_to_mhlo.h"

namespace xla {
namespace sdy {

void registerSdyRoundTripTestingPipeline() {
  mlir::PassPipelineRegistration<>(
      "xla-sdy-round-trip-testing-pipeline",
      "Run Shardonnay export pipeline, then convert to HLO, then convert to "
      "MHLO, then import back to Shardonnay",
      [](mlir::OpPassManager& pm) {
        addSdyRoundTripExportPipeline(pm);
        pm.addPass(createSdyRoundTripMhloToHloToMhloPass());
        addSdyRoundTripImportPipeline(pm);
      });
}

}  // namespace sdy
}  // namespace xla
