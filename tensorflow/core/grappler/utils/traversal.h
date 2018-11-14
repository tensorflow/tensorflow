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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_

#include <functional>
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"

namespace tensorflow {
namespace grappler {

// Traverse the graph in reverse dfs order, starting from the list of nodes
// specified in the 'from' argument. The pre_order and post_order functors will
// be called on each reachable node (including the 'from' nodes) in pre and post
// order. If loops are found, the on_back_edge functor will be called on the
// corresponding back edges. Moreover, the pre and post order will assume that
// these back edges will be cut.
void ReverseDfs(
    const GraphView& graph_view, const std::vector<const NodeDef*>& from,
    const std::function<void(const NodeDef*)>& pre_order,
    const std::function<void(const NodeDef*)>& post_order,
    const std::function<void(const NodeDef*, const NodeDef*)>& on_back_edge);

void ReverseDfs(
    const MutableGraphView& graph_view, const std::vector<const NodeDef*>& from,
    const std::function<void(const NodeDef*)>& pre_order,
    const std::function<void(const NodeDef*)>& post_order,
    const std::function<void(const NodeDef*, const NodeDef*)>& on_back_edge);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
