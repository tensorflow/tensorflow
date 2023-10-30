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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/device_type.pb.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/framework/device_type.h"
#include "tsl/platform/error_logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpPassManager;
using mlir::PassManager;
using mlir::func::FuncOp;

constexpr char kBridgeComponent[] = "TFXLABridge";

// Run the TF XLA Bridge based on the input pipeline, which can be either TPU
// bridge pipeline or non TPU bridge pipeline.
tensorflow::Status RunTFXLABridge(
    ModuleOp module,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder,
    llvm::StringRef module_name = llvm::StringRef(),
    llvm::StringRef dump_prefix = "tf_xla_bridge_v2") {
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
                                             dump_prefix.str() + "_before"),
        module, llvm::StringRef(), &bridge);
  }

  if (VLOG_IS_ON(2) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(),
                                      kDebugGroupBridgePhase1Clustering)) {
    ::tensorflow::tf2xla::internal::EnablePassIRPrinting(
        bridge, kDebugGroupBridgePhase1Clustering, module_name);
  }

  LogicalResult result = bridge.run(module);
  (void)result;

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             dump_prefix.str() + "_after"),
        module, llvm::StringRef(), &bridge);
  }

  return diag_handler.ConsumeStatus();
}

void CreateTPUBridgePipeline(OpPassManager &pm, llvm::StringRef module_name) {
  pm.addPass(mlir::TFTPU::CreateTPUValidateInputsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  tensorflow::tf2xla::internal::AddBridgeClusteringPipelinePasses(pm,
                                                                  module_name);
}

tensorflow::Status RecordIfErrorStatus(const std::string error_prefix,
                                       bool fallback_enabled,
                                       std::string device_type,
                                       absl::Status status) {
  if (status.ok()) {
    return status;
  }

  VLOG(2) << error_prefix << " " << status;
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      device_type, /*bridge_version=*/"v2",
      /*fallback_enabled=*/fallback_enabled,
      /*result=*/"failure");

  tsl::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_1,
      status);

  std::string bridge_subcomponent = "TFXLA_PHASE_ONE_MLIR_TPU_BRIDGE";
  if (device_type != "tpu") {
    bridge_subcomponent = "TFXLA_PHASE_ONE_MLIR_CPU/GPU_BRIDGE";
  }

  tsl::error_logging::Log(kBridgeComponent, bridge_subcomponent,
                          status.ToString())
      .IgnoreError();

  return status;
}

tensorflow::Status TPUBridge(ModuleOp module, bool fallback_enabled,
                             llvm::StringRef module_name) {
  VLOG(2)
      << "TPU Bridge called stack trace is "
      << "(NOTE: this is not an error; rather the stack trace for debugging) : "
      << tensorflow::CurrentStackTrace();
  std::string device_type = "tpu";
  Status clustering_status = RunTFXLABridge(
      module,
      [module_name](OpPassManager &pm) {
        CreateTPUBridgePipeline(pm, module_name);
      },
      module_name, /*dump_prefix=*/"tf_xla_bridge_v2_tpu");

  TF_RETURN_IF_ERROR(RecordIfErrorStatus(/*error_prefix=*/"clustering_v2",
                                         fallback_enabled, device_type,
                                         clustering_status));

  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      device_type, /*bridge_version=*/"v2",
      /*fallback_enabled=*/fallback_enabled,
      /*result=*/"success");

  return absl::OkStatus();
}

tensorflow::Status RunNonTPUBridge(ModuleOp module,
                                   bool is_in_fallback_enabled_mode,
                                   llvm::StringRef module_name) {
  VLOG(2)
      << "CPU/GPU Bridge called stack trace is "
      << "(NOTE: this is not an error; rather the stack trace for debugging) : "
      << tensorflow::CurrentStackTrace();

  std::string device_type = "cpu/gpu";
  Status clustering_status = RunTFXLABridge(
      module,
      [](OpPassManager &pm) {
        tensorflow::tf2xla::internal::AddNonTPUBridgeClusteringPipelinePasses(
            pm);
      },
      module_name, /*dump_prefix=*/"tf_xla_bridge_v2_nontpu");

  TF_RETURN_IF_ERROR(RecordIfErrorStatus(/*error_prefix=*/"clustering_v2",
                                         is_in_fallback_enabled_mode,
                                         device_type, clustering_status));

  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      device_type, /*bridge_version=*/"v2", is_in_fallback_enabled_mode,
      /*result=*/"success");

  return absl::OkStatus();
}

tensorflow::Status RunFunctionTf2xlaClusteringBridge(
    ModuleOp module, DeviceType device_type, bool is_in_fallback_enabled_mode,
    llvm::StringRef module_name) {
  if (device_type == DeviceType::XLA_TPU_JIT) {
    return TPUBridge(module, /*fallback_enabled=*/is_in_fallback_enabled_mode,
                     /*module_name=*/module_name);
  }

  return RunNonTPUBridge(module, is_in_fallback_enabled_mode, module_name);
}

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
