/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_DEF_BUILDER_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_DEF_BUILDER_UTIL_H_

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class Graph;

// Converts the `GraphDef` being built by `builder` to a `Graph` and
// stores it in `*graph`.
// TODO(josh11b): Make this faster; right now it converts
// Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
// edges from the source and to the sink node, resolves back edges
// by name), and makes sure the resulting graph is valid.
absl::Status GraphDefBuilderToGraph(const GraphDefBuilder& builder,
                                    Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_DEF_BUILDER_UTIL_H_
