/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_

#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// TODO(ezhulenev, b/121379902): We should be consistent with GraphTopologyView
// and use `GraphView::Edge` to pass extra dependencies.
struct TopologicalDependency {
  TopologicalDependency(const NodeDef* from, const NodeDef* to)
      : from(from), to(to) {}
  const NodeDef* from;
  const NodeDef* to;
};

// Computes a topological ordering for the graph nodes and outputs nodes in the
// topological order to the `topo_order` output argument.
//
// It's possible to pass additional edges that do not exists in a graph, but
// must be respected when computing graph topological order. Example: Tensorflow
// runtime allows concurrent execution of dequeue/enqueue ops from the same
// queue resource, but we might want to enforce ordering between them.
Status ComputeTopologicalOrder(
    const GraphDef& graph,
    absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<const NodeDef*>* topo_order);
Status ComputeTopologicalOrder(const GraphDef& graph,
                               std::vector<const NodeDef*>* topo_order);

// Sorts a graph in topological order.
Status TopologicalSort(GraphDef* graph);

// Sorts a graph in topological order and reverse it.
Status ReversedTopologicalSort(GraphDef* graph);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_TOPOLOGICAL_SORT_H_
