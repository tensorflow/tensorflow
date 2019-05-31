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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_SCC_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_SCC_H_

#include <unordered_map>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace grappler {

// Computes modified strongly connected components:
// All nodes that are not part of a loop are assigned the special -1 id
// All nodes that are part of at least one loop are assigned a positive
// component id: if 2 nodes v and w are reachable from one another (i.e. if they
// belong to the same scc), they'll be assigned the same id, otherwise they'll
// be assigned distinct ids. *num_components is set to the number of distinct
// ids.
void StronglyConnectedComponents(
    const GraphDef& graph, std::unordered_map<const NodeDef*, int>* components,
    int* num_components);

// Returns the number of individual loops present in the graph, and populate the
// 'loops' argument with the collection of loops (denoted by their loop ids) a
// node is part of. Loops ids are arbitrary.
int IdentifyLoops(const GraphDef& graph,
                  std::unordered_map<const NodeDef*, std::vector<int>>* loops);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_SCC_H_
