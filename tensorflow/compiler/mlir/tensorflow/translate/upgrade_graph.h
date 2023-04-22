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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_UPGRADE_GRAPH_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_UPGRADE_GRAPH_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

class GraphDef;
class MetaGraphDef;

// Generate the shared_name for resource handle ops in the graph and functions
// if their shared_names are empty. Resource handle ops with empty shared_name
// may have undesired semantics.
Status GenerateResourceSharedNameIfEmpty(
    GraphDef& gdef, const OpRegistryInterface* default_registry);

// Run grapler passes over `meta_graph_def`.graph_def() and returns the
// optimized graphdef.
stream_executor::port::StatusOr<GraphDef> RunGrappler(
    const MetaGraphDef& meta_graph_def);

// Upgrade the `graph` and `flib_def` by applying control flow
// functionalization.
Status UpgradeLegacyGraph(Graph* graph, FunctionLibraryDefinition* flib_def,
                          bool restrict_functionalization_to_tpu_nodes);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_UPGRADE_GRAPH_H_
