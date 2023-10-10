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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/cluster_tf.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/inference/inference_passes.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/error_logging.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpPassManager;
using mlir::PassManager;
using mlir::func::FuncOp;

namespace {

// Name of component for error logging. This name is fixed and required to
// enable logging.
constexpr char kBridgeComponent[] = "TFXLABridge";

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
            module_name.str(), kDebugGroupBridgePhase1Clustering,
            pass_tag_name);
      },
      "",
      /*print_module_scope=*/true));
  pm->enableTiming();
}

void CreateTPUBridgePipelineV1(OpPassManager &pm) {
  pm.addPass(mlir::tf2xla::internal::CreateInferenceMetricsPass());

  // Convert to unified compilation and replication attributes.
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  // For V1 compatibility, we process a module where the graph does not have
  // feeds and fetched. We extract first the TPU computation in a submodule,
  // where it'll be in a function with args and returned values, much more
  // like a TF v2 module. We can then run the usual pipeline on this nested
  // module. Afterward we inline back in the parent module and delete the
  // nested one.
  pm.addPass(mlir::tf_executor::CreateTFExecutorTPUV1IslandCoarseningPass());
  pm.addPass(mlir::tf_executor::CreateTFExecutorTPUV1IslandOutliningPass());
  OpPassManager &nested_module = pm.nest<ModuleOp>();
  internal::AddBridgeClusteringPipelinePasses(nested_module);

  pm.addPass(mlir::tf_executor::CreateTFExecutorTPUV1IslandInliningPass());
  // There are cases where we don't consume all compilation and replication
  // attributes like we do for the V2 pipeline, so we need to convert them
  // from unified to legacy attributes before they get exposed to outside of
  // the bridge.
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateConvertToLegacyCompileAndReplicateAttributesPass());
}

// Run the TF XLA Bridge based on the input pipeline, which can be either TPU
// bridge pipeline or non TPU bridge pipeline.
tensorflow::Status RunTFXLABridge(
    ModuleOp module,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder,
    llvm::StringRef module_name = llvm::StringRef()) {
  // Explicitly check that the TensorFlow dialect can constant fold ops.
  // Constant folding is essential for the bridge. Without this check, the
  // bridge may fail with an error that is difficult to understand and not
  // actionable.
  if (!mlir::TF::TensorFlowDialect::HasConstantFoldHook()) {
    return tensorflow::errors::Internal(
        "TensorFlow dialect missing constant fold hook in TFXLA bridge phase "
        "1; this could happen if the binary doesn't link the constant fold "
        "hook registration library.");
  }

  PassManager bridge(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(bridge);

  // Populate a passmanager with the list of passes that implement the bridge.
  pipeline_builder(bridge);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "tf_xla_bridge_before"),
        module, llvm::StringRef(), &bridge);
  }

  if (VLOG_IS_ON(2) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(),
                                      kDebugGroupBridgePhase1Clustering)) {
    EnableDetailedLogging(&bridge, module_name);
  }

  LogicalResult result = bridge.run(module);
  (void)result;

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "tf_xla_bridge_after"),
        module, llvm::StringRef(), &bridge);
  }

  return diag_handler.ConsumeStatus();
}

}  // namespace

tensorflow::Status RunSessionTf2xlaClusteringBridge(ModuleOp module) {
  VLOG(2) << "TPU Sessions Bridge called stack trace is "
          << "(NOTE: this is not an error; rather the stack trace for "
             "debugging) : "
          << tensorflow::CurrentStackTrace();

  bool fallback_enabled = false;
  Status bridge_status = RunTFXLABridge(
      module, [](OpPassManager &pm) { CreateTPUBridgePipelineV1(pm); });

  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      "tpu", "v1", fallback_enabled,
      bridge_status.ok() ? "success" : "failure");
  if (!bridge_status.ok()) {
    tsl::error_logging::Log(kBridgeComponent,
                            "TFXLA_PHASE_ONE_MLIR_TPU_V1_COMPAT_BRIDGE",
                            bridge_status.ToString())
        .IgnoreError();
    return bridge_status;
  }

  return tensorflow::tf2xla::v1::ExportFromTensorflowDialectToExecutor(module);
}

// Registers a pipeline builder function for TF TPU V1 bridge.
mlir::PassPipelineRegistration<> tpu_pipeline_v1(
    "tf-cluster-tpu-bridge-v1",
    "Run all the passes involved in transforming a TensorFlow V1 graph before "
    "execution so that it is suitable for targeting TPUs.",
    CreateTPUBridgePipelineV1);

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
