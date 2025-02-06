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

#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/round_trip_common/import_backend_func_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/round_trip_common/import_sdy_custom_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h"

namespace xla {
namespace sdy {

using ::mlir::func::FuncOp;

void addCommonPreImportPasses(mlir::OpPassManager& pm) {
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
  pm.addNestedPass<FuncOp>(createImportConstantsPass());
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo_ext::createStablehloFlattenTuplePass());
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  config.fold = false;
  config.cseConstants = false;
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass(config));
}

void addCommonPostImportPasses(mlir::OpPassManager& pm) {
  pm.addPass(createImportSdyCustomCallsPass());
  pm.addNestedPass<FuncOp>(createOpenWhileFreeVarsShardingPass());
  pm.addPass(createImportBackendFuncCallsPass());
}

}  // namespace sdy
}  // namespace xla
