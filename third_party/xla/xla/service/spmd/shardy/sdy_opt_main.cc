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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_export.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_import.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/round_trip_common/convert_sharding_custom_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"
#include "xla/service/spmd/shardy/round_trip_common/import_backend_func_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/sdy_round_trip/remove_size_one_axes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"
#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/mhlo_to_hlo_to_mhlo.h"
#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/testing_pipeline.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();

  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect, mlir::mhlo::MhloDialect,
                  mlir::sdy::SdyDialect, mlir::stablehlo::StablehloDialect>();
  mlir::func::registerAllExtensions(dialects);

  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  xla::sdy::registerMhloImportPipeline();
  xla::sdy::registerMhloImportShardingsPass();
  xla::sdy::registerMhloRoundTripShardMapImportPass();
  xla::sdy::registerConvertShardingCustomCallsPass();
  xla::sdy::registerOpenWhileFreeVarsShardingPass();
  xla::sdy::registerImportBackendFuncCallsPass();
  xla::sdy::registerImportConstantsPass();

  xla::sdy::registerMhloExportPipeline();
  xla::sdy::registerMhloExportShardingsPass();
  xla::sdy::registerMhloRoundTripShardMapExportPass();
  xla::sdy::registerExportNamedComputationsPass();
  xla::sdy::registerExportOpsPass();

  xla::sdy::registerSdyRoundTripMhloToHloToMhloPass();
  xla::sdy::registerSdyRoundTripExportShardyAttrsPass();
  xla::sdy::registerSdyRoundTripImportShardyAttrsPass();
  xla::sdy::registerSdyRoundTripRemoveSizeOneAxesPass();
  xla::sdy::registerSdyRoundTripExportOpsPass();
  xla::sdy::registerSdyRoundTripExportPipeline();
  xla::sdy::registerSdyRoundTripShardMapExportPass();
  xla::sdy::registerSdyRoundTripShardMapImportPass();
  xla::sdy::registerSdyRoundTripImportPipeline();
  xla::sdy::registerSdyRoundTripTestingPipeline();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "XLA SDY pass driver\n", dialects));
}
