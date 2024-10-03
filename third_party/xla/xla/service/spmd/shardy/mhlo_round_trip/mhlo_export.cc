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

#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_export.h"

#include <functional>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"

namespace xla {
namespace sdy {

void addMhloExportPipeline(mlir::OpPassManager& pm) {
  // This pass converts `sdy.constant` (which isn't foldable) into
  // `mhlo.constant` (which is foldable), therefore greedy pattern rewriters
  // shouldn't be applied before converting to HLO as they apply folding.
  pm.addPass(createExportOpsPass());
  pm.addPass(createMhloRoundTripShardMapExportPass());
  pm.addPass(createExportNamedComputationsPass());
  pm.addPass(createExportMhloShardingsPass());
}

void registerMhloExportPipeline() {
  mlir::PassPipelineRegistration<> exportPipeline(
      "xla-sdy-mhlo-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an MHLO module, "
      "which is ready for MHLO -> HLO conversion.",
      addMhloExportPipeline);
}

}  // namespace sdy
}  // namespace xla
