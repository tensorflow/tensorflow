/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_GRAPH_TO_TF_EXECUTOR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_GRAPH_TO_TF_EXECUTOR_UTIL_H_

#include <optional>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// These are used for grouping the recorded stats appropriately. Specifically,
// we're considering different entrypoints to the bridge as having potentially
// interesting differences at least in the domain of accepted graphs so we want
// to separately track graph features based on these unique entrypoints. One key
// example of this distinction is for TFRT which uses the "nominal" TPU bridge
// pipeline, but may potentially allow graphs with v1 control flow. This
// separate grouping will allow us to dig into these differences granularly.
enum class TF2XLABridgeVersion {
  kNominal = 0,
  kV1Compat,
  kTFRTNominal,
  kNotBridgeUseCase,
};

// Analyzes whether the graph has features not guaranteed to be supported by the
// MLIR-based TF XLA bridge for phase 1. If MLIR bridge phase 1 is not used,
// then MLIR bridge phase 2 will not be used. The optional `function_library`
// can be provided if it contains function definitions not including in the
// `graph` FunctionLibraryDefinition.
//
// Conservatively, during the initial rollout, we are not supporting graphs for
// which any of the following are true:
//
//  - Not known to be TF2
//  - Contains one or more reference variables
//  - Contains one or more TPUPartitionedCall ops (which is a proxy for
//    inference), but the graph is not v1 compat
//  - Uses V1 control flow
//  - Graph is invalid or otherwise encounters error during traversal
// If `single_core_inference_mode` is true, we skip some of check conditions
// because they are not applicable.
// TODO(b/241702857): remove single_core_inference_mode
bool GraphHasUnsupportedFeaturesInMlirBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library,
    std::optional<ConfigProto> config_proto, TF2XLABridgeVersion bridge_version,
    bool single_core_inference_mode);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_GRAPH_TO_TF_EXECUTOR_UTIL_H_
