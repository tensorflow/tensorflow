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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpPassManager;
using mlir::Pass;
using mlir::PassManager;
using mlir::func::FuncOp;

namespace {

void AddTfDialectToExecutorPasses(OpPassManager &pm) {
  auto add_pass = [&](std::unique_ptr<Pass> pass) {
    pm.addNestedPass<FuncOp>(std::move(pass));
    pm.addPass(mlir::CreateBreakUpIslandsPass());
  };

  pm.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
  add_pass(mlir::CreateFunctionalToExecutorDialectConversionPass());
  add_pass(mlir::TFDevice::CreateReplicateToIslandPass(
      /*legacy_graph_export=*/true));
  add_pass(mlir::TFDevice::CreateReplicaIDToDeviceOrdinalPass());
  add_pass(mlir::TFDevice::CreateParallelExecuteToIslandsPass(
      /*legacy_graph_export=*/true));
  add_pass(mlir::TFDevice::CreateLaunchToDeviceAttributePass(
      /*legacy_graph_export=*/true));
  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUDevicePropagationPass());
  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUColocateSplitsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_convert_control_to_data_outputs_pass) {
    pm.addPass(
        mlir::tf_executor::CreateTFExecutorConvertControlToDataOutputsPass());
  }
  pm.addPass(mlir::TF::CreateVerifySuitableForExportPass());
}

// Add logger to bridge passmanager.
// Enable timing statistics per pass for the bridge passmanager.
void EnableDetailedLogging(PassManager *pm,
                           llvm::StringRef module_name = llvm::StringRef()) {
  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm->getContext()->disableMultithreading();
  pm->enableIRPrinting(std::make_unique<::tensorflow::DataDumperLoggerConfig>(
      [module_name](const std::string &pass_tag_name, mlir::Operation *op) {
        return DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), kDebugGroupBridgePhase1, pass_tag_name);
      },
      "",
      /*print_module_scope=*/true));
}

}  // namespace

tensorflow::Status ExportFromTensorflowDialectToExecutor(
    ModuleOp module, llvm::StringRef module_name) {
  PassManager tf_to_executor(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(tf_to_executor);

  AddTfDialectToExecutorPasses(tf_to_executor);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), kDebugGroupMain,
            "tfxla_bridge_v1_tfdialect_to_executor_before"),
        module, llvm::StringRef(), &tf_to_executor);

    if (VLOG_IS_ON(2) || DEBUG_DATA_DUMPER()->ShouldDump(
                             module_name.str(), kDebugGroupBridgePhase1)) {
      EnableDetailedLogging(&tf_to_executor, module_name);
    }
  }

  LogicalResult result = tf_to_executor.run(module);
  (void)result;  // Ignore the error since it's captured by the diag_handler.

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), kDebugGroupMain,
            "tfxla_bridge_v1_tfdialect_to_executor_after"),
        module, llvm::StringRef(), &tf_to_executor);
  }

  return diag_handler.ConsumeStatus();
}

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
