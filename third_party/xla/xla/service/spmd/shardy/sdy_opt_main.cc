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

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "shardy/round_trip_import/pipelines.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/round_trip_common/import_backend_func_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/round_trip_common/import_sdy_custom_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h"
#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_callback_custom_calls.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/sdy_round_trip/remove_size_one_axes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/stablehlo_to_hlo_to_stablehlo.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/testing_pipeline.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_callback_custom_calls.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::stablehlo_ext::registerPasses();
  mlir::DialectRegistry dialects;
  mlir::sdy::registerAllDialects(dialects);
  mlir::func::registerAllExtensions(dialects);
  dialects.insert<mlir::mhlo::MhloDialect>();
  xla::sdy::registerMhloExtensions(dialects);

  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  xla::sdy::registerStablehloImportPipeline();
  xla::sdy::registerStablehloImportShardingsPass();
  xla::sdy::registerStablehloRoundTripShardMapImportPass();
  xla::sdy::registerImportSdyCustomCallsPass();
  xla::sdy::registerOpenWhileFreeVarsShardingPass();
  xla::sdy::registerImportBackendFuncCallsPass();
  xla::sdy::registerImportConstantsPass();

  xla::sdy::registerStablehloExportPipeline();
  xla::sdy::registerStablehloExportShardingsPass();
  xla::sdy::registerStablehloRoundTripExportCallbackCustomCallsPass();
  xla::sdy::registerStablehloRoundTripShardMapExportPass();
  xla::sdy::registerExportNamedComputationsPass();
  xla::sdy::registerExportOpsPass();

  xla::sdy::registerSdyRoundTripStablehloToHloToStablehloPass();
  xla::sdy::registerSdyRoundTripExportShardyAttrsPass();
  xla::sdy::registerSdyRoundTripImportCallbackCustomCallsPass();
  xla::sdy::registerSdyRoundTripImportShardyAttrsPass();
  xla::sdy::registerSdyRoundTripRemoveSizeOneAxesPass();
  xla::sdy::registerSdyRoundTripExportOpsPass();
  xla::sdy::registerSdyRoundTripExportPipeline();
  xla::sdy::registerSdyRoundTripDedupMeshesPass();
  xla::sdy::registerSdyRoundTripShardMapExportPass();
  xla::sdy::registerSdyRoundTripShardMapImportPass();
  xla::sdy::registerSdyRoundTripImportPipeline();
  xla::sdy::registerSdyRoundTripTestingPipeline();

  // Test SdyRoundTripImportPipeline cloned on Shardy Github.
  mlir::sdy::registerSdyRoundTripImportPipeline();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "XLA SDY pass driver\n", dialects));
}
