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
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace {

// Checks if the module has any TPU devices in its device list.
bool HasTPUDevice(mlir::ModuleOp op) {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(op.getOperation(), &devices)))
    return false;

  for (const auto& device : devices.device_names()) {
    if (device.has_type && device.type == "TPU") return true;
  }
  return false;
}
}  // namespace

namespace tensorflow {

auto* mlir_bridge_gauge_v1 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v1",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF1 models");
auto* mlir_bridge_gauge_v2 = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/config/experimental/enable_mlir_bridge_gauge_v2",
    "Tracks usage of the MLIR-based TF2XLA bridge among TF2 models");

// Analyzes the user requested policy as well as the contents of the graph and
// determines whether the MLIR Bridge should be run.
//
// If the user explicitly requests the bridge be enabled or disabled, this
// function will respect the request. If the user does not explicitly request
// enabled or disabled, it will decide whether or not to run the bridge.
//
// The config_proto param is a required input for all TF1 graphs but it is
// redundant for TF2 graphs.
bool IsMlirBridgePassEnabled(const Graph& graph,
                             const absl::optional<ConfigProto>& config_proto) {
  MlirBridgeRolloutPolicy policy =
      GetMlirBridgeRolloutPolicy(graph, config_proto);
  return (policy == MlirBridgeRolloutPolicy::kEnabledByUser ||
          policy == MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis);
}

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph) {
  if (!IsEnabled(config_proto, graph)) {
    VLOG(0) << "Skipping MLIR TPU Bridge, session flag not enabled";
    mlir_bridge_gauge_v2->GetCell()->Set(false);
    return Status::OK();
  }

  // Skip MLIR TPU Bridge if no TPU devices found.
  if (!HasTPUDevice(module)) {
    VLOG(0) << "Skipping MLIR TPU Bridge, no TPU devices found";
    return Status::OK();
  }

  VLOG(0) << "Running MLIR TPU Bridge";
  mlir_bridge_gauge_v2->GetCell()->Set(true);
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1)));

  return Status::OK();
}
Status MlirBridgeV1CompatPass::Run(const GraphOptimizationPassOptions& options,
                                   mlir::ModuleOp module) {
  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return Status::OK();

  if (!IsEnabled(options.session_options->config, **options.graph)) {
    VLOG(0) << "Skipping MLIR TPU Bridge V1 Compat, session flag not enabled";
    mlir_bridge_gauge_v1->GetCell()->Set(false);
    return Status::OK();
  }

  // Skip MLIR TPU Bridge if no TPU devices found.
  if (!HasTPUDevice(module)) {
    VLOG(0) << "Skipping MLIR TPU Bridge V1 Compat, no TPU devices found";
    return Status::OK();
  }

  VLOG(0) << "Running MLIR TPU Bridge V1 Compat";
  mlir_bridge_gauge_v1->GetCell()->Set(true);
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridgeV1Compat(module, /*enable_logging=*/VLOG_IS_ON(1)));

  return Status::OK();
}

}  // namespace tensorflow
