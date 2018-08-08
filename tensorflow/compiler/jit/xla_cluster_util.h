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

// Contains utilities for clustering compilable graph nodes via XLA.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_

#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks nodes in a cluster to be placed outside the xla
// compilation by the encapsulate subgraphs pass.
extern const char* const kXlaOutsideCompilationAttr;

using OrderedNodeSet = std::set<Node*, NodeComparatorID>;

// Returns the DeviceType corresponding to 'device'.
Status DeviceToDeviceType(const string& device, DeviceType* device_type);

// Returns true if `node` has a ref tensor input that it forwards to its output.
bool HasForwardedRefInput(const Node& node);

// Creates a graph representation to enable cycle detection when clustering.
// This representation handles loops in graph by disconnecting each loop from
// the enclosing graph.
Status CreateCycleDetectionGraph(const Graph* graph, GraphCycles* cycles);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
