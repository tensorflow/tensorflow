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

#include <memory>
#include <string>

#include "absl/log/log.h"
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
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/util/debug_data_dumper.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

using mlir::LogicalResult;
using mlir::OpPassManager;
using mlir::PassManager;
using mlir::func::FuncOp;
using mlir::TF::StandardPipelineOptions;

// Setup the input pass manager to enable IR dumping after each pass.
// Note a side effect of this method is that multi threading will be disabled.
void EnablePassIRPrinting(PassManager& pm, const std::string& dump_group_name,
                          llvm::StringRef module_name) {
  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm.getContext()->disableMultithreading();
  pm.enableIRPrinting(std::make_unique<::tensorflow::DataDumperLoggerConfig>(
      [module_name, dump_group_name](const std::string& pass_tag_name,
                                     mlir::Operation* op) {
        return DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), dump_group_name, pass_tag_name);
      },
      /*pass_prefix=*/"",
      /*print_module_scope=*/true));
  pm.enableTiming();
}

void AddLowerClusterToRuntimeOpsPassPipeline(OpPassManager& pm,
                                             llvm::StringRef module_name) {
  pm.addPass(mlir::TFTPU::CreateTPURewritePass(module_name));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateEmbeddingProgramKeyPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addPass(mlir::TFTPU::CreateTPUMergeVariablesWithExecutePass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateExtractTPUCopyWithDynamicShapeOpPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUColocateCompositeResourceOps());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_tpu_variable_runtime_reformatting_pass) {
    pm.addPass(mlir::TFTPU::CreateTPUVariableRuntimeReformattingPass());
  }
}

void CreateLowerClusterToRuntimeOpsPassPipeline(
    OpPassManager& pm, const StandardPipelineOptions& options) {
  AddLowerClusterToRuntimeOpsPassPipeline(pm, /*module_name=*/"");
}

}  // namespace

absl::Status RunLowerClusterToRuntimeOpsPassPipeline(
    mlir::ModuleOp module, llvm::StringRef module_name) {
  PassManager runtime_lowering(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(runtime_lowering);

  AddLowerClusterToRuntimeOpsPassPipeline(runtime_lowering, module_name);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "runtime_lowering_before"),
        module, llvm::StringRef(), &runtime_lowering);
  }

  if (VLOG_IS_ON(2) || DEBUG_DATA_DUMPER()->ShouldDump(
                           module_name.str(), kDebugGroupRuntimeLowering)) {
    EnablePassIRPrinting(runtime_lowering, kDebugGroupRuntimeLowering,
                         module_name);
  }

  // Ignore the result since diag_handler consumes it
  LogicalResult result = runtime_lowering.run(module);
  (void)result;

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "runtime_lowering_after"),
        module, llvm::StringRef(), &runtime_lowering);
  }

  return diag_handler.ConsumeStatus();
}

void RegisterLowerClusterToRuntimeOpsPassPipeline() {
  static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
      "tfrt-lower-cluster-to-runtime-ops",
      "Run all the passes involved after the clustering transformations from "
      "the TF2XLA Bridge. Takes as input a Module with tf_device.cluster ops "
      "and outputs TFRT runtime ops such as TPUCompile",
      CreateLowerClusterToRuntimeOpsPassPipeline);
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
