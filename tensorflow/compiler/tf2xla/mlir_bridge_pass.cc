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

#include <string>

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
// Values for the label 'ProcessingState'
// MLIR Phase 1 bridge is a no op
const char kMlirBridgeNoOp[] = "kMlirBridgeNoOp";
// MLIR Phase 1 bridge failed (manually enabled)
constexpr char kMlirBridgeFail[] = "kMlirBridgeFail";
// MLIR Phase 1 bridge failed with fallback
constexpr char kMlirBridgeFallbackFail[] = "kMlirBridgeFallbackFail";
// MLIR Phase 1 bridge was successful
constexpr char kMlirBridgeSuccess[] = "kMlirBridgeSuccess";
// Graph analysis/policy are used to determine graph status
constexpr char kGraphAnalysis[] = "kGraphAnalysis";

// Values for the label 'PassState'
constexpr char kEnabled[] = "kEnabled";
constexpr char kDisabled[] = "kDisabled";
constexpr char kFallbackEnabled[] = "kFallbackEnabled";
// Graph has unsupported features so will not be processed by MLIR Phase 1
constexpr char kMlirUnsupportedGraph[] = "kMlirUnsupportedGraph";
// MLIR bridge has been disabled by the user.
constexpr char kMlirDisabled[] = "kMlirDisabled";
// There are no TPU Devices or Ops
constexpr char kMlirNoTpuDevicesOrOps[] = "kMlirNoTpuDevicesOrOps";
// MLIR bridge is enabled by the user.
constexpr char kMlirEnabled[] = "kMlirEnabled";
// Graph has supported features so will be processed by MLIR Phase 1
constexpr char kMlirSupportedGraph[] = "kMlirSupportedGraph";

auto* mlir_bridge_gauge_v1 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v1",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF1 models");
auto* mlir_bridge_gauge_v2 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v2",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF2 models");

namespace {

constexpr char kTPUReplicateAttr[] = "_tpu_replicate";

bool HasTPUDevice(mlir::ModuleOp module) {
  mlir::TF::RuntimeDevices devices;
  if (failed(GetDevicesFromOp(module.getOperation(), &devices))) return false;
  return absl::c_any_of(
      devices.device_names(),
      [](const tensorflow::DeviceNameUtils::ParsedName& device) {
        return device.has_type && device.type == "TPU";
      });
}

bool HasTPUOp(mlir::ModuleOp module) {
  auto walk_result = module.walk([&](mlir::Operation* op) {
    auto replicate_attr =
        op->getAttrOfType<mlir::StringAttr>(kTPUReplicateAttr);
    if (replicate_attr) return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return walk_result.wasInterrupted();
}

// Checks that the module has both - TPU devices in its device list and contains
// TPU ops (identifed by `_tpu_replicate` attribute on ops).
bool HasTPUDevicesAndOps(mlir::ModuleOp module) {
  return HasTPUDevice(module) && HasTPUOp(module);
}

bool HasTPUDevice(const DeviceSet& device_set) {
  for (const Device* device : device_set.devices()) {
    if (!device) continue;
    const DeviceNameUtils::ParsedName& name = device->parsed_name();
    if (name.has_type && name.type == "TPU") return true;
  }
  return false;
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
MlirOptimizationPassState MlirBridgePass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
  // Skip MLIR TPU Bridge if no TPU devices found.
  if (device_set && !HasTPUDevice(*device_set)) {
    return MlirOptimizationPassState::Disabled;
  }

  // We set `uses_uninitialized_resource_args` to false here because the first
  // phase of the bridge is not affected by uninitialized resource args.
  MlirBridgeRolloutPolicy policy =
      GetMlirBridgeRolloutPolicy(graph, &function_library, config_proto,
                                 /*uses_uninitialized_resource_args=*/false);
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysisSafeModeFallback:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser:
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      return MlirOptimizationPassState::Disabled;
  }
}

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph,
                           const FunctionLibraryDefinition& function_library) {
  // Skip MLIR TPU Bridge if no TPU devices or TPU ops found.
  // This check needs to precede GetPassState for instrumentation purposes.
  if (!HasTPUDevicesAndOps(module)) {
    VLOG(1) << "Skipping MLIR TPU Bridge, no TPU devices or TPU ops found";
    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(
        kMlirNoTpuDevicesOrOps, kMlirBridgeNoOp);
    return Status::OK();
  }
  // Set device_set to nullptr here as the device specific checks are performed
  // based on the devices in the module.
  auto pass_state = GetPassState(/*device_set=*/nullptr, config_proto, graph,
                                 function_library);
  if (pass_state == MlirOptimizationPassState::Disabled) {
    ConfigProto::Experimental::MlirBridgeRollout state =
        GetMlirBridgeRolloutState(config_proto);
    // For metrics, differentiate difference beteween pass being disabled by
    // user and by graph analysis failure
    if (state == ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED) {
      VLOG(1) << "Skipping MLIR TPU Bridge, MLIR TPU bridge disabled by user. "
                 "Old bridge will evaluate.";
      metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kMlirDisabled,
                                                             kGraphAnalysis);
    } else {
      VLOG(1) << "Skipping MLIR TPU Bridge, MLIR TPU bridge disabled because "
                 "graph has unsupported features. Old bridge will evaluate.";
      metrics::UpdateTfMlirGraphOptimizationPassStateCounter(
          kMlirUnsupportedGraph, kGraphAnalysis);
    }

    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kDisabled,
                                                           kMlirBridgeNoOp);
    return Status::OK();
  }

  if (pass_state == MlirOptimizationPassState::FallbackEnabled)
    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kMlirSupportedGraph,
                                                           kGraphAnalysis);
  else
    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kMlirEnabled,
                                                           kGraphAnalysis);

  VLOG(1) << "Running MLIR TPU Bridge";

  mlir_bridge_gauge_v2->GetCell()->Set(true);
  Status status =
      mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1));
  if (!status.ok()) {
    if (pass_state == MlirOptimizationPassState::Enabled)
      metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kEnabled,
                                                             kMlirBridgeFail);
    else
      metrics::UpdateTfMlirGraphOptimizationPassStateCounter(
          kFallbackEnabled, kMlirBridgeFallbackFail);
    return status;
  }
  if (pass_state == MlirOptimizationPassState::Enabled)
    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kEnabled,
                                                           kMlirBridgeSuccess);
  else
    metrics::UpdateTfMlirGraphOptimizationPassStateCounter(kFallbackEnabled,
                                                           kMlirBridgeSuccess);
  return Status::OK();
}

MlirOptimizationPassState MlirBridgeV1CompatPass::GetPassState(
    const DeviceSet* device_set, const ConfigProto& config_proto,
    const Graph& graph,
    const FunctionLibraryDefinition& function_library) const {
  // Skip MLIR TPU Bridge if no TPU devices found.
  if (device_set && !HasTPUDevice(*device_set))
    return MlirOptimizationPassState::Disabled;

  // Do not run the bridge if it's enabled by the graph analysis,
  // only run if it's enabled by the user explicitly.
  // We set `uses_uninitialized_resource_args` to false here because the first
  // phase of the bridge is not affected by uninitialized resource args.
  MlirBridgeRolloutPolicy policy = GetMlirBridgeRolloutPolicy(
      graph, /*function_library=*/&function_library, config_proto,
      /*uses_uninitialized_resource_args=*/false);
  switch (policy) {
    case MlirBridgeRolloutPolicy::kEnabledByUser:
      return MlirOptimizationPassState::Enabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysisSafeModeFallback:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis:
      return MlirOptimizationPassState::FallbackEnabled;
    case MlirBridgeRolloutPolicy::kDisabledByUser:
    case MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis:
      return MlirOptimizationPassState::Disabled;
  }
}

Status MlirBridgeV1CompatPass::Run(const GraphOptimizationPassOptions& options,
                                   mlir::ModuleOp module) {
  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return Status::OK();

  // Set device_set to nullptr here as the device specific checks are performed
  // based on the devices in the module.
  if (GetPassState(/*device_set=*/nullptr, options.session_options->config,
                   **options.graph,
                   *options.flib_def) == MlirOptimizationPassState::Disabled) {
    VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, session flag not enabled";
    mlir_bridge_gauge_v1->GetCell()->Set(false);
    return Status::OK();
  }

  // Skip MLIR TPU Bridge if no TPU devices or TPU ops found.
  if (!HasTPUDevicesAndOps(module)) {
    VLOG(1) << "Skipping MLIR TPU Bridge V1 Compat, no TPU devices or TPU ops "
               "found";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR TPU Bridge V1 Compat";

  mlir_bridge_gauge_v1->GetCell()->Set(true);
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridgeV1Compat(module, /*enable_logging=*/VLOG_IS_ON(1)));

  return Status::OK();
}

}  // namespace tensorflow
