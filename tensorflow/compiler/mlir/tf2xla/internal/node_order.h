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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_NODE_ORDER_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_NODE_ORDER_H_

#include <functional>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

struct GroupByDevice {
  std::string operator()(const Node* node) const {
    return node->requested_device();
  }
};

// Performs a topological ordering of nodes.
// This has the property that any child node of a parent node p is emitted
// before p. A grouping function is used to break ties if multiple child nodes
// (of possibly different parents) are ready to be emitted at some point, which
// is when we prefer to stay in the current group. Remaining ties are broken by
// node name.
// The "emit" function is used for outputing the result, and is called once
// for each node.
// This algorithm is O(n * k * log k), with k the largest node degree.
void TopologicalOrdering(
    const Graph& g, const std::function<void(Node*)>& emit,
    const std::function<std::string(Node*)>& get_grouping_key);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_NODE_ORDER_H_
