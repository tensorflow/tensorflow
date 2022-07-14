/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_MLIR_BRIDGE_ROLLOUT_POLICY_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_MLIR_BRIDGE_ROLLOUT_POLICY_H_

#include "mlir/IR/BuiltinOps.h"
#include "absl/types/optional.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

enum class MlirBridgeRolloutPolicy {
  // The MLIR bridge is explicitly disabled by the user and must not be run.
  kDisabledByUser = 0,
  // The MLIR bridge is explicitly enabled by the user and must be run. If the
  // MLIR bridge errors, the fallback path should NOT be used.
  kEnabledByUser,
  // The bridge was not explicitly enabled or disabled by the user. Based on the
  // features in the model, the MLIR bridge should not be run.
  kDisabledAfterGraphAnalysis,
  // The bridge was not explicitly enabled or disabled by the user. Based on the
  // features in the model, the MLIR bridge should be run. If the MLIR Bridge
  // errors, the fallback path should be used whenever possible.
  kEnabledAfterGraphAnalysis,
  // The bridge was fallback enabled in a safe mode and passed all graph
  // analysis checks.
  kEnabledAfterGraphAnalysisSafeModeFallback
};

// Analyzes the user requested policy as well as the contents of the graph and
// returns true when the MLIR Bridge should be run.
//
// If the user explicitly requests the bridge be enabled or disabled, this
// function will respect the request. If the user does not explicitly request
// enabled or disabled, it will decide whether or not to run the bridge.
//
// The config_proto param is a required input for all TF1 graphs but it is
// redundant for TF2 graphs.
// If getting rollout policy involves graph analysis, `record_stats` is used
// to decide whether to emit metrics on unsupported features of the graph.
MlirBridgeRolloutPolicy GetMlirBridgeRolloutPolicy(
    const tensorflow::Graph& graph,
    const FunctionLibraryDefinition* function_library,
    std::optional<tensorflow::ConfigProto> config_proto,
    bool uses_uninitialized_resource_args, bool record_stats = false);

static inline MlirBridgeRolloutPolicy GetMlirBridge2ndPhaseRolloutPolicy(
    mlir::ModuleOp module) {
  return MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis;
}

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_MLIR_BRIDGE_ROLLOUT_POLICY_H_
