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

#ifndef TENSORFLOW_GRAPH_COSTUTIL_H_
#define TENSORFLOW_GRAPH_COSTUTIL_H_

#include <vector>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class CostModel;
class Graph;

// result[i] is an estimate of the longest execution path from
// the node with id i to the sink node.
std::vector<int64> LongestOutgoingPathCost(const Graph& graph,
                                           const CostModel& cm);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_COSTUTIL_H_
