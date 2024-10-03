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

#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"

#include <cassert>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/remove_size_one_axes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"

namespace xla {
namespace sdy {

using ::mlir::PassPipelineRegistration;

void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm) {
  pm.addPass(createExportNamedComputationsPass());
  // Run canonicalizer to simplify `ManualComputationOp`s.
  pm.addPass(mlir::createCanonicalizerPass());
  // We save `sdy.sharding`s on those custom calls during
  // `createSdyRoundTripExportShardyAttrsPass` and make use of
  // `createSdyRoundTripImportShardyAttrsPass` to import them.
  pm.addPass(createSdyRoundTripExportOpsPass());
  pm.addPass(createSdyRoundTripShardMapExportPass());
  // Preserve the SDY shardings for `createExportMhloShardingsPass` so that
  // we have both `mhlo.sharding`s and hidden `sdy.sharding`s on the module. We
  // want to have `mhlo.sharding`s for Pathways to read from.
  pm.addPass(createSdyRoundTripExportShardyAttrsPass());
  pm.addPass(createExportMhloShardingsPass());
}

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm) {
  addCommonPreImportPasses(pm);
  pm.addPass(createSdyRoundTripImportShardyAttrsPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createSdyRoundTripRemoveSizeOneAxesPass());
  addCommonPostImportPasses(pm);
}

void registerSdyRoundTripExportPipeline() {
  PassPipelineRegistration<> exportPipeline(
      "xla-sdy-round-trip-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an MHLO module, "
      "but with the SDY ops/attrs saved for roundtripping.",
      addSdyRoundTripExportPipeline);
}

void registerSdyRoundTripImportPipeline() {
  PassPipelineRegistration<> importPipeline(
      "xla-sdy-round-trip-import-pipeline",
      "Run passes to import an mhlo module into the SDY (Shardy) dialect.",
      addSdyRoundTripImportPipeline);
}

}  // namespace sdy
}  // namespace xla
