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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_COLOCATION_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_COLOCATION_H_

#include <unordered_map>
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace grappler {

// Evaluates the colocation relation in the graph and rewrites the new
// colocation relation in the graph. We scan the graph nodes sequentially, and
// builds a disjoint-sets of nodes (within each disjoint-set the nodes are
// colocated with each other). We then select the root node of each set as a
// representative node, and then colocate each node within the set (should also
// exist in graph) with the representative node.
// Note that there is current one situation this function can't handle:
// Node A colocates with X, node B colocates with Y, X colocates with Y but
// X, Y are removed from graph. In this case we can't know A colocates with B.
void ReassignColocation(GraphDef* graph);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_COLOCATION_H_
