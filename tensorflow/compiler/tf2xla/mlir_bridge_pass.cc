/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"
#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/WalkResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/cluster_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_bridge_pass_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/tsl/framework/device_type.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

auto* mlir_bridge_gauge_v1 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v1",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF1 models");
auto* mlir_bridge_gauge_v2 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v2",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF2 models");

namespace {

using ::mlir::ModuleOp;

bool HasTPUDevice(const DeviceSet& device_set) {
  for (const Device* device : device_set.devices()) {
    if (!device) continue;
    const DeviceNameUtils::ParsedName& name = device->parsed_name();
    if (name.has_type && name.type == "TPU") return true;
  }
  return false;
}

bool HasTPUDevice(mlir::ModuleOp module) {
  mlir::TF::RuntimeDevices devices;
  if (failed(GetDevicesFromOp(module.getOperation(), &devices))) return false;
  return absl::c_any_of(
      devices.device_names(),
      [](const tensorflow::DeviceNameUtils::ParsedName& device) {
        return device.has_type && device.type == kTpuDevice;
      });
}

bool HasDevice(mlir::ModuleOp module) {
  mlir::TF::RuntimeDevices devices;
  if (failed(GetDevicesFromOp(module.getOperation(), &devices))) return false;
  return !devices.device_names().empty();
}

// V1 Compat Bridge extracts out a program into a submodule and runs clustering
// only on the submodule.
absl::Status RunLowerToRuntimeOpsOnSubmodule(ModuleOp parent_module,
                                             bool is_in_fallback_enabled_mode) {
  int num_submodules = 0;
  absl::Status runtime_lowering_status;
  parent_module.walk([&](ModuleOp submodule) {
    if (submodule == parent_module) return mlir::WalkResult::advance();
    num_submodules++;
    runtime_lowering_status =
        tensorflow::tfrt_compiler::RunLowerClusterToRuntimeOpsPassPipeline(
            submodule, tsl::DeviceType(DEVICE_TPU_XLA_JIT));
    if (num_submodules > 1) {
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });

  if (num_submodules > 1) {
    return absl::InternalError(
        "Lower to runtime has more than one submodule. Erroring out.");
  }

  return runtime_lowering_status;
}

}  // namespace

// Analyzes the user requested policy as well as the contents of the graph and
// function_library_definition to determine whether the MLIR Bridge should be
// run.
//
// If the user explicitly requests the bridge be enabled or disabled, this
// function will respect the request. If the user does not explicitly request
// enabled or disabled, it will decide whether or not to run the bridge.
//
// The config_proto param is a required input for all TF1 graphs but it is
// redundant for TF2 graphs.
MlirOptimizationPassState GetPassStateImpl(
    bool is_supported_by_replicated_brige, const ConfigProto& config_proto,
    const Graph& graph, const FunctionLibraryDefinition& function_library) {
  // Skip MLIR TF/XLA Bridge if no XLA-compilable ops are found.
  if (!is_supported_by_replicated_brige &&
      !IsSupportedByNonReplicatedBridge(graph, &function_library)) {
    VLOG(3) << "Skipping MLIR Bridge, graph is not qualified to run the bridge";
    return MlirOptimizationPassState::Disabled;
  }

  // GetMlirBridgeRolloutPolicy will analyze a TPU graph if users have not
  // explicltly requested a policy.
  MlirBridgeRolloutPolicy policy = GetMlirBridgeRolloutPolicy(
      graph, &function_library, config_proto, is_supported_by_replicated_brige,
      /*is_v1_compat=*/false, /*record_stats=*/false);
  // GetPassState is called once before MlirBridgePass starts, and the pass
  // gets skipped if it is disabled. Log such cases in this function. The cases
  // where the pass is enabled will only be logged during their execution to
  // prevent them from being counted twice.
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser: {
      VLOG(1) << "Skipping MLIR "
              << (is_supported_by_replicated_brige ? "Replicated"
                                                   : "Non-replicated")
              << " Bridge, disabled by user. "
                 "The fallback will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter(
          /*bridge_type*/ is_supported_by_replicated_brige
              ? mlir::TF::kMlirPh1BridgeCounterReplicated
              : mlir::TF::kMlirPh1BridgeCounterNonReplicated,
          /*bridge_version*/ mlir::TF::kMlirPh1BridgeCounterV2,
          /*device_type*/
          is_supported_by_replicated_brige
              ? mlir::TF::kMlirPh1BridgeCounterTpu
              : mlir::TF::kMlirPh1BridgeCounterNonTpu,
          /*fallback_enabled*/ true,
          /*result*/ "disabled_by_user");
      return MlirOptimizationPassState::Disabled;
    }
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      // Graph analysis only runs on TPU graph.
      VLOG(1) << "Skipping MLIR TPU Bridge, disabled because the "
                 "graph has unsupported features. The fallback will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter(
          /*bridge_type*/ mlir::TF::kMlirPh1BridgeCounterReplicated,
          /*bridge_version*/ mlir::TF::kMlirPh1BridgeCounterV2,
          /*device_type*/ mlir::TF::kMlirPh1BridgeCounterTpu,
          /*fallback_enabled*/ true,
          /*result*/ "invalid_graph");
      // For Invalid Graph Analysis we need to log here because Run will not
      // be called.
      LogGraphFeatures(graph, &function_library, config_proto,
                       /*is_v1_compat=*/false);
      return MlirOptimizationPassState::Disabled;
  }
}

MlirOptimizationPassState MlirBridgePass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
  // While we do not use device type information to choose which pass pipeline
  // to execute, it's needed for successful execution.
  if (!device_set) {
    // This is not expected in practice.
    VLOG(1) << "Device set is empty!";
    return MlirOptimizationPassState::Disabled;
  }

  // TODO(b/328084279): when MlirBridgePass::GetPassState() returns
  // MlirOptimizationPassState::FallbackEnabled or
  // MlirOptimizationPassState::Enabled, Tensorflow imports a Graph to an
  // MLIR module, calls MlirBridgePass::Run(), and exports the MLIR module to a
  // Graph. The Graph->MLIR module->Graph round trip will not happen if
  // MlirOptimizationPassState::Disabled is returned. Some input graphs with a
  // TPU device in device_set yet without replication depends on the round
  // trip, which does not always produce the same Graph. Call
  // HasTPUDevice(*device_set) to ensure such graps work. Note
  // MlirBridgePass::Run() will still reject such graphs that they do not go
  // through the Phase 1 Bridge.
  return GetPassStateImpl(
      /*is_supported_by_replicated_brige*/ IsSupportedByReplicatedBridge(
          graph, &function_library) ||
          HasTPUDevice(*device_set),
      config_proto, graph, function_library);
}

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
absl::Status MlirBridgePass::Run(
    const std::string& function_name, const ConfigProto& config_proto,
    mlir::ModuleOp module, const Graph& graph,
    const FunctionLibraryDefinition& function_library) {
  static absl::once_flag flag;
  absl::call_once(flag, tsl::UpdateLogVerbosityIfDefined,
                  "TF_DEBUG_LOG_VERBOSITY");

  if (!HasDevice(module)) {
    LOG(INFO) << "No devices in " << function_name << "\n";
    return absl::OkStatus();
  }

  if (HasTPUPartitionedCallOpInModule(module)) {
    VLOG(1) << "Skipping MLIR TF2XLA Bridge. This is an inference graph, "
               "Session V1 Bridge should be used during execution of "
               "TPUPartitionedCall.";
    return absl::OkStatus();
  }

  // TODO(b/241853328): Add caching of pass state and call logging/metrics
  // related to graph analysis from here.
  bool is_supported_by_replicated_brige = IsSupportedByReplicatedBridge(module);
  auto pass_state = GetPassStateImpl(is_supported_by_replicated_brige,
                                     config_proto, graph, function_library);

  if (pass_state == MlirOptimizationPassState::Disabled) {
    // GetPassState is called before run() and run() will only be called if the
    // pass is not disabled. However, the graph may have been updated between
    // when the pass state was originally calculated and now, so this check is
    // required to reflect any possible changes.
    VLOG(1) << "MlirBridgePass is disabled and will not run.";
    return absl::OkStatus();
  }

  bool fallback_enabled = false;
  if (is_supported_by_replicated_brige) {
    if (pass_state == MlirOptimizationPassState::FallbackEnabled) {
      // TODO (b/241853328) Consider moving logging if caching for graph
      // analysis or GetPassState is added
      LogGraphFeatures(graph, &function_library, config_proto,
                       /*is_v1_compat=*/false);
      fallback_enabled = true;
    }
    VLOG(1) << "Running MLIR Replicated Bridge";
    mlir_bridge_gauge_v2->GetCell()->Set(true);

    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::RunFunctionTf2xlaClusteringBridge(
            module, /*is_supported_by_replicated_brige*/ true, fallback_enabled,
            function_name));

    TF_RETURN_IF_ERROR(
        tensorflow::tfrt_compiler::RunLowerClusterToRuntimeOpsPassPipeline(
            module, tsl::DeviceType(DEVICE_TPU_XLA_JIT), function_name));
  } else {
    VLOG(1) << "Running MLIR Non-replicated Bridge";
    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::RunFunctionTf2xlaClusteringBridge(
            module, /*is_supported_by_replicated_brige*/ false,
            fallback_enabled, function_name));

    TF_RETURN_IF_ERROR(
        tensorflow::tfrt_compiler::RunLowerClusterToRuntimeOpsPassPipeline(
            module, tsl::DeviceType(DEVICE_GPU_XLA_JIT), function_name));
  }

  return tensorflow::tf2xla::v2::ExportFromTensorflowDialectToExecutor(
      module, function_name);
}

MlirOptimizationPassState MlirBridgeV1CompatPass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
  // Skip MLIR Bridge if no potential XLA clusters are found.
  if (!IsSupportedByReplicatedBridge(graph, &function_library))
    return MlirOptimizationPassState::Disabled;
  MlirBridgeRolloutPolicy policy = GetMlirBridgeRolloutPolicy(
      graph, /*function_library=*/&function_library, config_proto,
      /*is_supported_by_replicated_brige*/ true,
      /*is_v1_compat=*/true,
      /*record_stats=*/false);
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser:
      VLOG(1) << "Skipping MLIR Replicated Bridge V1 Compat, MLIR Replicated "
                 "bridge disabled "
                 "by user. Fallback will evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter(
          /*bridge_type*/ mlir::TF::kMlirPh1BridgeCounterReplicated,
          /*bridge_version*/ mlir::TF::kMlirPh1BridgeCounterV1,
          /*device_type*/ mlir::TF::kMlirPh1BridgeCounterTpu,
          /*fallback_enabled*/ true,
          /*result*/ "disabled_by_user");
      return MlirOptimizationPassState::Disabled;
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      VLOG(1) << "Skipping MLIR Replicated Bridge V1 Compat, MLIR Replicated "
                 "bridge disabled "
                 "because graph has unsupported features. Old bridge will "
                 "evaluate.";
      metrics::UpdateTfMlirBridgeFirstPhaseCounter(
          /*bridge_type*/ mlir::TF::kMlirPh1BridgeCounterReplicated,
          /*bridge_version*/ mlir::TF::kMlirPh1BridgeCounterV1,
          /*device_type*/ mlir::TF::kMlirPh1BridgeCounterTpu,
          /*fallback_enabled*/ true,
          /*result*/ "invalid_graph");
      // For Invalid Graph Analysis we need to log here because Run will not be
      // called.
      LogGraphFeatures(graph, &function_library, config_proto,
                       /*is_v1_compat=*/true);
      return MlirOptimizationPassState::Disabled;
  }
}

absl::Status MlirBridgeV1CompatPass::Run(
    const GraphOptimizationPassOptions& options, mlir::ModuleOp module) {
  static absl::once_flag flag;
  absl::call_once(flag, tsl::UpdateLogVerbosityIfDefined,
                  "TF_DEBUG_LOG_VERBOSITY");

  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return absl::OkStatus();

  // Skip MLIR Replicated Bridge if no eligible ops found.
  if (!IsSupportedByReplicatedBridge(module)) {
    VLOG(1) << "Skipping MLIR Replicated Bridge V1 Compat, no eligible ops "
               "found";
    return absl::OkStatus();
  }

  MlirOptimizationPassState pass_state =
      GetPassState(/*device_set=*/nullptr, options.session_options->config,
                   **options.graph, *options.flib_def);

  // Set device_set to nullptr here as the device specific checks are performed
  // based on the devices in the module.
  if (pass_state == MlirOptimizationPassState::Disabled) {
    // GetPassState is called before run() and run() will only be called if the
    // pass is not disabled. However, the graph may have been updated between
    // when the pass state was originally calculated and now, so this check is
    // required to reflect any possible changes.
    VLOG(1) << "Skipping MLIR Replicated Bridge V1 Compat, session flag not "
               "enabled";
    mlir_bridge_gauge_v1->GetCell()->Set(false);
    return absl::OkStatus();
  }

  // 1) If the MLIR module contains a TPUPartitionedCall, we skip here
  // 2) When TPUPartitionedCall starts executing, it calls MLIR bridge as a
  // part of PRE_PLACEMENT optimization
  // 3) This MLIR bridge version is V1 Compat
  if (HasTPUPartitionedCallOpInModule(module)) {
    VLOG(1) << "Skipping MLIR Replicated Bridge V1 Compat. This is an "
               "inference graph, V1 "
               "Compat should be used during execution of TPUPartitionedCall.";
    return absl::OkStatus();
  }

  bool fallback_enabled = false;
  if (pass_state == MlirOptimizationPassState::FallbackEnabled) {
    // TODO (b/241853328) Consider moving logging if caching for graph analysis
    // or GetPassState is added
    LogGraphFeatures(**options.graph, options.flib_def,
                     options.session_options->config,
                     /*is_v1_compat=*/true);
    fallback_enabled = true;
  }

  VLOG(1) << "Running MLIR Replicated Bridge V1 Compat";
  mlir_bridge_gauge_v1->GetCell()->Set(true);
  TF_RETURN_IF_ERROR(tensorflow::tf2xla::v1::RunSessionTf2xlaClusteringBridge(
      module, fallback_enabled));

  auto lower_cluster_to_runtime_ops_pass_pipeline =
      RunLowerToRuntimeOpsOnSubmodule(module, fallback_enabled);
  if (!lower_cluster_to_runtime_ops_pass_pipeline.ok()) {
    VLOG(1) << "Error while lowering cluster to runtime ops: "
            << lower_cluster_to_runtime_ops_pass_pipeline;
    return lower_cluster_to_runtime_ops_pass_pipeline;
  }

  return tensorflow::tf2xla::v1::ExportFromTensorflowDialectToExecutor(module);
}

}  // namespace tensorflow
