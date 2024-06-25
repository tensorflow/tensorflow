
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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"

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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/core/util/debug_data_dumper.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using mlir::LogicalResult;
using mlir::OpPassManager;
using mlir::PassManager;
using mlir::func::FuncOp;

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

void AddClusterToIfrtRuntimeOpsPassPipeline(OpPassManager& pm,
                                            llvm::StringRef module_name) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateCanonicalizeCompileAndReplicateAttributesPass());

  pm.addNestedPass<mlir::func::FuncOp>(CreateTfIdentityPropagationPass());

  pm.addNestedPass<mlir::func::FuncOp>(CreateTfRestoreSplittingPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateTfRestorePruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateTfRestoreMergingPass());

  pm.addPass(CreateLowerToIfrtRestoreVariablePass());

  pm.addPass(CreateRewriteClusterToIfrtCallPass());

  // Sink VarHandle with ReadVariableOp: subsequent SinkVariableAsNamedArrayPass
  // rely on the co-existence of VarHandle and ReadVariable in the same
  // function.
  // First, we inline all the function calls. This will sink VarHandle
  // with ReadVariable in most cases. Then SinkInvariantOpsPass will sink
  // VarHandle to a few special Ops that inliner does not handle.
  // TODO(b/319045348): the bridge before this pipeline already does some
  // inlining. Consider removing this inliner.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(::tensorflow::CreateSinkInInvariantOpsPass());

  // Decompose resource ops as resource variables are loaded by ReadVariableOp
  // and can be lowered to IfrtLoadVariableOp in the subsequent
  // SinkVariableAsNamedArrayPass.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  // Sink variable tensor as named array in IFRT.
  pm.addPass(CreateSinkVariableAsNamedArrayPass());
}

}  // namespace

absl::Status RunClusterToIfrtRuntimeOpsPassPipeline(
    mlir::ModuleOp module, llvm::StringRef module_name) {
  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  PassManager runtime_lowering(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(runtime_lowering);

  AddClusterToIfrtRuntimeOpsPassPipeline(runtime_lowering, module_name);

  if (VLOG_IS_ON(1)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "ifrt_runtime_lowering_before"),
        module, llvm::StringRef(), &runtime_lowering);
  }

  if (VLOG_IS_ON(2)) {
    EnablePassIRPrinting(runtime_lowering, kDebugGroupRuntimeLowering,
                         module_name);
  }

  // Ignore the result since diag_handler consumes it
  LogicalResult result = runtime_lowering.run(module);
  (void)result;

  if (VLOG_IS_ON(1)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "ifrt_runtime_lowering_after"),
        module, llvm::StringRef(), &runtime_lowering);
  }

  return diag_handler.ConsumeStatus();
}

// Register all IfrtPass
void RegisterTfIfrtPasses() { registerTfrtIfrtServingPasses(); }

}  // namespace ifrt_serving
}  // namespace tensorflow
