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
#include <functional>

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_callback_custom_calls.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/remove_size_one_axes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

namespace xla {
namespace sdy {

using ::mlir::PassPipelineOptions;
using ::mlir::PassPipelineRegistration;

void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm) {
  pm.addPass(createSdyRoundTripDedupMeshesPass());
  pm.addPass(createExportNamedComputationsPass());
  pm.addPass(createSdyRoundTripExportOpsPass());
  pm.addPass(createSdyRoundTripShardMapExportPass());
  // Preserve the SDY shardings for `createExportStablehloShardingsPass` so that
  // we have both `mhlo.sharding`s and hidden `sdy.sharding`s on the module. We
  // want to have `mhlo.sharding`s for Pathways to read from.
  pm.addPass(createSdyRoundTripExportShardyAttrsPass());
  pm.addPass(createExportStablehloShardingsPass());
}

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport) {
  addCommonPreImportPasses(pm, enableConstantImport);
  pm.addPass(createSdyRoundTripImportCallbackCustomCallsPass());
  pm.addPass(createSdyRoundTripImportShardyAttrsPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createSdyRoundTripRemoveSizeOneAxesPass());
  addCommonPostImportPasses(pm);
}

void registerSdyRoundTripExportPipeline() {
  PassPipelineRegistration<> exportPipeline(
      "xla-sdy-round-trip-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an StableHLO module, "
      "but with the SDY ops/attrs saved for roundtripping.",
      addSdyRoundTripExportPipeline);
}

namespace {

struct SdyRoundTripImportPipelineOptions
    : public PassPipelineOptions<SdyRoundTripImportPipelineOptions> {
  Option<bool> enable_constant_import{*this, "enable-constant-import",
                                      llvm::cl::desc("Enable constant import."),
                                      llvm::cl::init(true)};
};

void sdyRoundTripImportPipeline(
    mlir::OpPassManager& pm, const SdyRoundTripImportPipelineOptions& options) {
  addSdyRoundTripImportPipeline(pm, options.enable_constant_import);
}

}  // namespace

void registerSdyRoundTripImportPipeline() {
  PassPipelineRegistration<SdyRoundTripImportPipelineOptions> importPipeline(
      "xla-sdy-round-trip-import-pipeline",
      "Run passes to import a StableHLO module into the SDY (Shardy) dialect.",
      sdyRoundTripImportPipeline);
}

}  // namespace sdy
}  // namespace xla
