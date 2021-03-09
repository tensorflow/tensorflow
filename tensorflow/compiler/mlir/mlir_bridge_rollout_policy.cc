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

#include "tensorflow/compiler/mlir/mlir_bridge_rollout_policy.h"

#include "tensorflow/compiler/jit/flags.h"

namespace tensorflow {

static ConfigProto::Experimental::MlirBridgeRollout GetUserRequest(
    absl::optional<ConfigProto> config_proto) {
  // TF1 graphs that do not override Sessions's ConfigProto and TF2 graphs
  // can enable/disable the graph via tf_mlir_enable_mlir_bridge.
  auto tf_mlir_enable_mlir_bridge =
      GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge;
  if (tf_mlir_enable_mlir_bridge !=
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED) {
    return tf_mlir_enable_mlir_bridge;
  }

  // If a ConfigProto was not passed in, we can assume the caller is
  // checking if TF2 graph should have the bridge enabled / disabled. In that
  // case, we have already checked tf_mlir_enable_mlir_bridge so it is safe to
  // return UNSPECIFIED here.
  if (!config_proto.has_value()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  }

  // TF1 graphs that do override Session's ConfigProto and set
  // ConfigProto's enable_mlir_bridge or mlir_bridge_rollout fields will not
  // update tf_mlir_enable_mlir_bridge so check their values.

  // ConfigProto's enable_mlir_bridge defaults to false so only respect it
  // when it is true.
  if (config_proto.value().experimental().enable_mlir_bridge()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  }
  return config_proto.value().experimental().mlir_bridge_rollout();
}

MlirBridgeRolloutPolicy GetMlirBridgeRolloutPolicy(
    const tensorflow::Graph& graph,
    const FunctionLibraryDefinition* function_library,
    absl::optional<ConfigProto> config_proto,
    bool uses_uninitialized_resource_args, bool record_stats) {
  switch (GetUserRequest(config_proto)) {
    case ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED:
      return MlirBridgeRolloutPolicy::kEnabledByUser;
    case ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED:
      return MlirBridgeRolloutPolicy::kDisabledByUser;
    default:
      // User did not explicitly enable or disable the bridge. For now, disable
      // the bridge.
      return MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis;
  }
}

}  // namespace tensorflow
