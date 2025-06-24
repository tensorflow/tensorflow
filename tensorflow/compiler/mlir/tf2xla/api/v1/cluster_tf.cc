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
#include "absl/status/status.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/inference/inference_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
#include "xla/tsl/framework/device_type.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"
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

void CreateReplicatedBridgePipelineV1(OpPassManager &pm) {
  pm.addPass(
      tensorflow::tf2xla::internal::CreateTPUValidateSessionInputsPass());
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
}

// Run the TF XLA Bridge based on the input pipeline, which can be either TPU
// bridge pipeline or non TPU bridge pipeline.
absl::Status RunTFXLABridge(
    ModuleOp module,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder,
    llvm::StringRef module_name = llvm::StringRef(),
    llvm::StringRef dump_prefix = "tf_xla_bridge_v1") {
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
  bridge.enableVerifier();
  ::tensorflow::applyTensorflowAndCLOptions(bridge);

  // Populate a passmanager with the list of passes that implement the bridge.
  pipeline_builder(bridge);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(DEBUG_DATA_DUMPER()->GetDumpFilename(
                                       module_name.str(), kDebugGroupMain,
                                       "_" + dump_prefix.str() + "_before"),
                                   module, llvm::StringRef(), &bridge);
  }

  if (VLOG_IS_ON(2) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(),
                                      kDebugGroupBridgePhase1Clustering)) {
    internal::EnablePassIRPrinting(bridge, kDebugGroupBridgePhase1Clustering,
                                   module_name);
  }

  LogicalResult result = bridge.run(module);
  (void)result;

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(DEBUG_DATA_DUMPER()->GetDumpFilename(
                                       module_name.str(), kDebugGroupMain,
                                       "_" + dump_prefix.str() + "_after"),
                                   module, llvm::StringRef(), &bridge);
  }

  return diag_handler.ConsumeStatus();
}

}  // namespace

absl::Status RecordStatusIfError(const std::string error_prefix,
                                 bool is_in_fallback_enabled_mode,
                                 absl::Status status) {
  if (status.ok()) {
    return status;
  }

  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      /*bridge_type=*/mlir::TF::kMlirPh1BridgeCounterReplicated,
      /*bridge_version=*/mlir::TF::kMlirPh1BridgeCounterV1,
      /*device_type*/ mlir::TF::kMlirPh1BridgeCounterTpu,
      /*fallback_enabled=*/is_in_fallback_enabled_mode,
      /*result=*/"failure");
  tsl::error_logging::Log(mlir::TF::kBridgeComponent,
                          "TFXLA_PHASE_ONE_MLIR_TPU_V1_COMPAT_BRIDGE",
                          status.ToString())
      .IgnoreError();

  return status;
}

// V1 Compat Bridge takes a TF Executor dialect and extracts the TF2 portion
// and inserts it into a submodule. We just want to run the clustering
// portion of the pipeline on just the single submodule.
absl::Status RunClusteringPipelineOnSubmodule(
    ModuleOp parent_module, bool is_in_fallback_enabled_mode) {
  int num_submodules = 0;
  absl::Status clustering_pipeline_status;
  parent_module.walk([&](ModuleOp submodule) {
    if (submodule == parent_module) return mlir::WalkResult::advance();
    num_submodules++;
    clustering_pipeline_status = RunTFXLABridge(
        submodule,
        [](OpPassManager &pm) {
          internal::AddReplicatedBridgeClusteringPipelinePasses(pm);
        },
        /*module_name=*/"", /*dump_prefix=*/"tf_xla_clustering_bridge_v1");
    if (num_submodules > 1) {
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });

  if (num_submodules > 1) {
    auto num_submodules_error = absl::InternalError(
        "V1 Compat Bridge has more than one submodule. Erroring out.");
    TF_RETURN_IF_ERROR(RecordStatusIfError(
        /*error_prefix=*/"Bridge has more than one submodule:",
        is_in_fallback_enabled_mode, num_submodules_error));
  }

  if (!clustering_pipeline_status.ok()) {
    TF_RETURN_IF_ERROR(RecordStatusIfError(
        /*error_prefix=*/"Bridge Errored running clustering pipeline:",
        is_in_fallback_enabled_mode, clustering_pipeline_status));
  }

  return absl::OkStatus();
}

absl::Status RunSessionTf2xlaClusteringBridge(
    ModuleOp module, bool is_in_fallback_enabled_mode) {
  VLOG(2) << "TPU Sessions Bridge called stack trace is "
          << "(NOTE: this is not an error; rather the stack trace for "
             "debugging) : "
          << tensorflow::CurrentStackTrace();

  absl::Status functional_import_status = RunTFXLABridge(
      module, [](OpPassManager &pm) { CreateReplicatedBridgePipelineV1(pm); },
      /*module_name=*/"", /*dump_prefix=*/"tf_xla_functional_import_bridge_v1");
  TF_RETURN_IF_ERROR(RecordStatusIfError(
      /*error_prefix=*/"Bridge Function Import V1", is_in_fallback_enabled_mode,
      functional_import_status));

  TF_RETURN_IF_ERROR(
      RunClusteringPipelineOnSubmodule(module, is_in_fallback_enabled_mode));

  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      /*bridge_type=*/mlir::TF::kMlirPh1BridgeCounterReplicated,
      /*bridge_version=*/mlir::TF::kMlirPh1BridgeCounterV1,
      /*device_type*/ mlir::TF::kMlirPh1BridgeCounterTpu,
      /*n_fallback_enabled*/ is_in_fallback_enabled_mode,
      /*result=*/"success");

  return absl::OkStatus();
}

// Registers a pipeline builder function for TF TPU V1 bridge.
mlir::PassPipelineRegistration<> replicated_clustering_bridge_v1(
    "tf-replicated-clustering-bridge-v1",
    "Run all the passes involved in transforming a TensorFlow V1 graph before "
    "execution so that it is suitable for targeting devices.",
    CreateReplicatedBridgePipelineV1);

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
