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

#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const ConfigProto& config_proto,
                           mlir::ModuleOp module) {
  if (!config_proto.experimental().enable_mlir_bridge()) {
    VLOG(1) << "Skipping MLIR Bridge Pass, session flag not enabled";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Bridge Pass";
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1)));

  return Status::OK();
}
Status MlirBridgeV1CompatPass::Run(const GraphOptimizationPassOptions& options,
                                   mlir::ModuleOp module) {
  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return Status::OK();

  if (!options.session_options->config.experimental().enable_mlir_bridge()) {
    VLOG(1) << "Skipping MLIR Bridge V1 Compat Pass, session flag not enabled";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Bridge V1 Compat Pass";
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridgeV1Compat(module, /*enable_logging=*/VLOG_IS_ON(1)));

  return Status::OK();
}

}  // namespace tensorflow
