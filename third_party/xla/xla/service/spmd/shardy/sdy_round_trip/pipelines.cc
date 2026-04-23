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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/round_trip_common/import_sdy_custom_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h"
#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

namespace xla {
namespace sdy {

using ::mlir::PassPipelineOptions;
using ::mlir::PassPipelineRegistration;
using ::mlir::func::FuncOp;

void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm,
                                   bool keepMeshesInlined,
                                   bool enableHloShardingV3) {
  // Lift meshes before deduping, since the dedup meshes pass ignores inlined
  // meshes.
  if (!keepMeshesInlined) {
    pm.addPass(mlir::sdy::createLiftInlinedMeshesPass());
  }
  pm.addPass(createSdyRoundTripDedupMeshesPass());
  pm.addPass(createSdyRoundTripExportOpsPass());
  pm.addPass(createSdyRoundTripShardMapExportPass(enableHloShardingV3));
  // Preserve the SDY shardings for `createExportStablehloShardingsPass` so that
  // we have both `mhlo.sharding`s and hidden `sdy.sharding`s on the module. We
  // want to have `mhlo.sharding`s for Pathways to read from.
  pm.addPass(createSdyRoundTripExportShardyAttrsPass(enableHloShardingV3));
  pm.addPass(createExportStablehloShardingsPass(
      /*addMissingShardingToControlFlow=*/false, enableHloShardingV3));
}

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport,
                                   bool liftAndDedupMeshes,
                                   bool enableHloShardingV3,
                                   bool enableNativeNonFlatSupport) {
  pm.addPass(mlir::createSymbolDCEPass());
  // TODO(b/333505182): remove when partitioning is done in SDY.
  // We call prepare-for-export pass before SDY propagation, so that all IR
  // changes happen before shardings are added to operations, to ensure the
  // correct shardings are added and that they are not lost by this pass.
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo_ext::createStablehloPrepareForHloExportPass());
  // We import `stablehlo.constant` ops to `sdy.constant` ops so that constants
  // aren't folded in greedy pattern rewriters, which would lift them outside of
  // nested regions (this undoes `WhileLoopConstantSinking` HLO pass).
  // Therefore, this pass needs to be applied after any StableHLO pass that
  // expects `stablehlo.constant`, and before any pass that has a greedy pattern
  // rewriter.
  if (enableConstantImport) {
    pm.addNestedPass<FuncOp>(createImportConstantsPass());
  }
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  pm.addPass(createSdyRoundTripImportShardyAttrsPass(enableHloShardingV3));
  pm.addPass(mlir::sdy::createFlattenCallGraphPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createImportSdyCustomCallsPass());
  pm.addNestedPass<FuncOp>(createOpenWhileFreeVarsShardingPass());
  if (enableHloShardingV3 || liftAndDedupMeshes) {
    // Lift and dedup inlined meshes in case of HloShardingV3 as meshes are:
    // * Inlined in HloShardingV3 and the converted sdy shardings.
    // * Inlined during sdy shard map export for storing shardy shardings.
    //
    // liftAndDedupMeshes is required here because of sdy shardings added
    // directly to hlo in tf2xla.
    pm.addPass(mlir::sdy::createLiftInlinedMeshesPass());
    pm.addPass(createSdyRoundTripDedupMeshesPass());
  }
  pm.addPass(createCanonicalizerPass(
      mlir::GreedyRewriteConfig()
          .setUseTopDownTraversal(true)
          .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Normal)
          .enableFolding(false)
          .enableConstantCSE(false)));
}

namespace {

struct SdyRoundTripExportPipelineOptions
    : public PassPipelineOptions<SdyRoundTripExportPipelineOptions> {
  Option<bool> keepMeshesInlined{
      *this, "keep-meshes-inlined",
      llvm::cl::desc("Whether to keep meshes inlined and not lift them."),
      llvm::cl::init(false)};
  Option<bool> enableHloShardingV3{
      *this, "enable-hlo-sharding-v3",
      llvm::cl::desc("Whether to enable HloShardingV3 which is the mesh and "
                     "axis based sharding representation."),
      llvm::cl::init(false)};
};

void sdyRoundTripExportPipeline(
    mlir::OpPassManager& pm, const SdyRoundTripExportPipelineOptions& options) {
  addSdyRoundTripExportPipeline(pm, options.keepMeshesInlined,
                                options.enableHloShardingV3);
}

}  // namespace

void registerSdyRoundTripExportPipeline() {
  PassPipelineRegistration<SdyRoundTripExportPipelineOptions> exportPipeline(
      "xla-sdy-round-trip-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an StableHLO module, "
      "but with the SDY ops/attrs saved for roundtripping.",
      sdyRoundTripExportPipeline);
}

namespace {

struct SdyRoundTripImportPipelineOptions
    : public PassPipelineOptions<SdyRoundTripImportPipelineOptions> {
  Option<bool> enableConstantImport{*this, "enable-constant-import",
                                    llvm::cl::desc("Enable constant import."),
                                    llvm::cl::init(true)};
  Option<bool> liftAndDedupMeshes{*this, "lift-and-dedup-meshes",
                                  llvm::cl::desc("Lift and dedup meshes."),
                                  llvm::cl::init(false)};
  Option<bool> enableHloShardingV3{
      *this, "enable-hlo-sharding-v3",
      llvm::cl::desc("Whether to enable HloShardingV3 which is the mesh and "
                     "axis based sharding representation."),
      llvm::cl::init(false)};
};

void sdyRoundTripImportPipeline(
    mlir::OpPassManager& pm, const SdyRoundTripImportPipelineOptions& options) {
  addSdyRoundTripImportPipeline(pm, options.enableConstantImport,
                                options.liftAndDedupMeshes,
                                options.enableHloShardingV3);
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
