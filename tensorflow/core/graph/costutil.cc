/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/graph/costutil.h"

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

std::vector<int64> LongestOutgoingPathCost(const Graph& graph,
                                           const CostModel& cm) {
  std::vector<int64> result(graph.num_node_ids());
  DFS(graph, nullptr, [&result, &cm](Node* n) {
    int64 max_child = 0;
    for (const Node* out : n->out_nodes()) {
      max_child = std::max(max_child, result[out->id()]);
    }
    result[n->id()] = max_child + (n->IsOp() ? cm.TimeEstimate(n).value() : 0);
  });
  return result;
}

}  // namespace tensorflow
