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
#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/runtime_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/tsl/framework/device_type.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/error_logging.h"
#include "tsl/platform/errors.h"

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

}  // namespace

void AddTPULowerClusterToRuntimeOpsPassPipeline(OpPassManager& pm,
                                                llvm::StringRef module_name) {
  pm.addPass(mlir::TFTPU::CreateTPURewritePass(module_name));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateEmbeddingProgramKeyPass());
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

void AddNonTPULowerClusterToRuntimeOpsPassPipeline(
    OpPassManager& pm, llvm::StringRef module_name) {
  // Rewrite cluster functions into XLA launch ops.
  pm.addPass(mlir::TFDevice::CreateXlaRewritePass());
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void CreateTPULowerClusterToRuntimeOpsPassPipeline(
    OpPassManager& pm, const StandardPipelineOptions& options) {
  AddTPULowerClusterToRuntimeOpsPassPipeline(pm, /*module_name=*/"");
}

void CreateNonTPULowerClusterToRuntimeOpsPassPipeline(
    OpPassManager& pm, const StandardPipelineOptions& options) {
  AddNonTPULowerClusterToRuntimeOpsPassPipeline(pm, /*module_name=*/"");
}

// TODO(b/306728216): Move this out of the Bridge component and into a Host
// runtime component.
tensorflow::Status RecordIfErrorStatus(const std::string error_prefix,
                                       std::string bridge_type,
                                       tsl::DeviceType device_type,
                                       absl::Status status) {
  if (status.ok()) {
    return status;
  }

  VLOG(2) << error_prefix << " " << status;
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      bridge_type,
      /*bridge_version=*/mlir::TF::kMlirPh1BridgeCounterV2,
      device_type.type_string(),
      /*fallback_enabled=*/false,
      /*result=*/"failure");

  std::string bridge_subcomponent = "TFXLA_PHASE_ONE_MLIR_TPU_BRIDGE";

  tsl::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_1,
      status);

  if (device_type != DeviceType(DEVICE_TPU_XLA_JIT)) {
    bridge_subcomponent = "TFXLA_PHASE_ONE_MLIR_CPU/GPU_BRIDGE";
  }

  tsl::error_logging::Log(mlir::TF::kBridgeComponent, bridge_subcomponent,
                          status.ToString())
      .IgnoreError();

  return status;
}

absl::Status RunLowerClusterToRuntimeOpsPassPipeline(
    mlir::ModuleOp module, tsl::DeviceType xla_device_type,
    llvm::StringRef module_name) {
  PassManager runtime_lowering(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(runtime_lowering);

  if (xla_device_type == DeviceType(DEVICE_TPU_XLA_JIT)) {
    AddTPULowerClusterToRuntimeOpsPassPipeline(runtime_lowering, module_name);
  } else {
    AddNonTPULowerClusterToRuntimeOpsPassPipeline(runtime_lowering,
                                                  module_name);
  }

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

  std::string bridge_type = xla_device_type == DeviceType(DEVICE_TPU_XLA_JIT)
                                ? mlir::TF::kMlirPh1BridgeCounterReplicated
                                : mlir::TF::kMlirPh1BridgeCounterNonReplicated;
  auto result_status = diag_handler.ConsumeStatus();
  TF_RETURN_IF_ERROR(
      RecordIfErrorStatus(/*error_prefix=*/"lower_cluster_to_runtime",
                          bridge_type, xla_device_type, result_status));

  return absl::OkStatus();
}

// TODO(b/305211853): Unify the CPU/TPU/GPU Execution Ops and thus these two
// passes should merge together.
void RegisterTPULowerClusterToRuntimeOpsPassPipeline() {
  static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
      "tfrt-lower-cluster-to-runtime-ops-tpu",
      "Run all the passes involved after the clustering transformations from "
      "the TF2XLA Bridge. Takes as input a Module with tf_device.cluster ops "
      "and outputs TFRT runtime ops such as TPUCompile. This pipeline is for "
      "TPU.",
      CreateTPULowerClusterToRuntimeOpsPassPipeline);
}

void RegisterNonTPULowerClusterToRuntimeOpsPassPipeline() {
  static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
      "tfrt-lower-cluster-to-runtime-ops-non-tpu",
      "Run all the passes involved after the clustering transformations from "
      "the TF2XLA Bridge. Takes as input a Module with tf_device.cluster ops "
      "and outputs TFRT runtime ops such as XlaLaunch. This is for CPU/GPU",
      CreateNonTPULowerClusterToRuntimeOpsPassPipeline);
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
