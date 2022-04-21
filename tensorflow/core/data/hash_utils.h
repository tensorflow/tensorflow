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
#ifndef TENSORFLOW_CORE_DATA_HASH_UTILS_H_
#define TENSORFLOW_CORE_DATA_HASH_UTILS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Returns a stable hash of the subgraph rooted at the given node.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashNode(const GraphDef& graph, const NodeDef& node, uint64* hash);
Status HashNode(const GraphDef& graph, const NodeDef& node,
                const FunctionLibraryDefinition& flib_def, uint64* hash);

// Returns a stable hash of the given tensor.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashTensor(const Tensor& tensor, uint64* hash);

// Returns a stable hash of the given graph.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashGraph(const GraphDef& graph, uint64* hash);

// Determines whether the given graphs are equal, following the same logic used
// for HashGraph. Returns OK if the graphs can be determined to be equal,
// otherwise returns an error message explaining why the graphs couldn't be
// determined to be equal.
Status CheckGraphsEqual(const GraphDef& a, const GraphDef& b);

// Determines whether the subgraphs rooted at the given nodes are equal
// following the same logic used for HashGraph. Returns OK if the graphs can be
// determined to be equal, otherwise returns an error message explaining why the
// graphs couldn't be determined to be equal.
Status CheckSubgraphsEqual(const GraphDef& a, const NodeDef* node_a,
                           const GraphDef& b, const NodeDef* node_b);
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_HASH_UTILS_H_
