/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Optimize the graph by reducing cross-host control output edges.
// Once we find any nodes in the graph having not less than
// `cross_host_edges_threshold` control output edges in one host, we create
// a `NoOp` node in the destination host to proxy the control edges between the
// oringal node and the destination control output nodes.
Status OptimizeCrossHostControlOutputEdges(Graph* graph,
                                           int cross_host_edges_threshold);

// Optimize the graph by reducing cross-host control input edges.
// Once we find any nodes in the graph having not less than
// `cross_host_edges_threshold` control input edges in one host, we create
// a `NoOp` node in the source host to proxy the control edges between the
// source control input nodes and oringal node.
Status OptimizeCrossHostControlInputEdges(Graph* graph,
                                          int cross_host_edges_threshold);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZE_CROSS_HOST_CONTROL_DEPS_H_
