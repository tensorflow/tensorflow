/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "xla/tsl/lib/monitoring/counter.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/error_logging.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpPassManager;
using mlir::PassManager;
using mlir::func::FuncOp;

auto *tf_dialect_to_executor_dialect_status = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/api/v2/tf_dialect_to_executor_dialect_status",
    "Counts how often a successful export from TF Dialect to Executor Dialect "
    "is",
    "status");

constexpr char kExportSuccess[] = "success";
constexpr char kExportFailed[] = "failed";

namespace {

void AddTfDialectToExecutorPasses(OpPassManager &pm) {
  pm.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());

  // First, we need to convert from functional, to executor dialect.
  pm.addNestedPass<FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());

  // Do a single pass to split the graph's single island op into an island per
  // op as expected by the following passes.
  pm.addNestedPass<FuncOp>(mlir::TF::CreateSplitIntoIslandPerOpPass());

  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateReplicateToIslandPass(
      /*legacy_graph_export=*/false));
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateReplicaIDToDeviceOrdinalPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateParallelExecuteToIslandsPass(
      /*legacy_graph_export=*/false));
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateLaunchToDeviceAttributePass(
      /*legacy_graph_export=*/false));

  // Do a single pass to encode necessary control deps in the IR according to
  // the results of side effect analysis.
  pm.addPass(
      mlir::tf_executor::CreateTFExecutorUpdateControlDependenciesPass());

  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUDevicePropagationPass());
  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUColocateSplitsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_convert_control_to_data_outputs_pass) {
    bool composite_tpuexecute_side_effects =
        tensorflow::GetMlirCommonFlags()
            ->tf_mlir_enable_composite_tpuexecute_side_effects;
    pm.addPass(
        mlir::tf_executor::CreateTFExecutorConvertControlToDataOutputsPass(
            composite_tpuexecute_side_effects));
  }
  pm.addPass(mlir::TF::CreateVerifySuitableForExportPass());
}

absl::Status RecordStatusIfError(absl::Status status) {
  if (status.ok()) {
    return absl::OkStatus();
  }

  tf_dialect_to_executor_dialect_status->GetCell(kExportFailed)->IncrementBy(1);
  VLOG(1) << "Failed to export from TF Dialect to TF Executor Dialect. "
          << status;

  constexpr char bridge_subcomponent[] =
      "TFXLA_TF_FUNCTIONAL_TO_EXECUTOR_EXPORT_v2";
  constexpr char kBridgeComponent[] = "TFXLABridge";

  tsl::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_1,
      status);

  tsl::error_logging::Log(kBridgeComponent, bridge_subcomponent,
                          status.ToString())
      .IgnoreError();

  return status;
}

}  // namespace

absl::Status ExportFromTensorflowDialectToExecutor(
    ModuleOp module, llvm::StringRef module_name) {
  PassManager tf_to_executor(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(tf_to_executor);
  tf_to_executor.enableVerifier();
  AddTfDialectToExecutorPasses(tf_to_executor);

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), kDebugGroupMain,
            "tfxla_bridge_v2_tfdialect_to_executor_before"),
        module, llvm::StringRef(), &tf_to_executor);

    if (VLOG_IS_ON(2) ||
        DEBUG_DATA_DUMPER()->ShouldDump(
            module_name.str(), kDebugGroupBridgePhase1ExecutorExport)) {
      internal::EnablePassIRPrinting(
          tf_to_executor, kDebugGroupBridgePhase1ExecutorExport, module_name);
    }
  }

  LogicalResult result = tf_to_executor.run(module);

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), kDebugGroupMain,
            "tfxla_bridge_v2_tfdialect_to_executor_after"),
        module, llvm::StringRef(), &tf_to_executor);
  }

  if (result.failed()) {
    return RecordStatusIfError(
        absl::InternalError("Failed to export from TF Dialect to TF Executor "
                            "Dialect. Read LLVM Pipeline Error"));
  }

  tf_dialect_to_executor_dialect_status->GetCell(kExportSuccess)
      ->IncrementBy(1);
  return absl::OkStatus();
}

mlir::PassPipelineRegistration<> tf_dialect_to_executor_pipeline(
    "tf-dialect-to-executor-v2",
    "Run passes to convert from TF Dialect to Executor in preparation for "
    "exporting module back to TF Graph.",
    AddTfDialectToExecutorPasses);

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
