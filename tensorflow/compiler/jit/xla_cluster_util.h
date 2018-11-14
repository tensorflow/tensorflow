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

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks nodes in a cluster to be placed outside the xla
// compilation by the encapsulate subgraphs pass.
extern const char* const kXlaOutsideCompilationAttr;

// The attribute that marks certain inputs to a Node as required to be a
// constant at compile time.  If this attribute is present then the
// CompileTimeConstantInput information in the corresponding XlaOpKernel is
// ignored.
//
// The value for this attribute, if present, has to be a list of strings naming
// the inputs to the node that must be constant.
extern const char* const kXlaCompileTimeConstantInputsAttr;

using OrderedNodeSet = std::set<Node*, NodeComparatorID>;

// Returns the DeviceType corresponding to 'device'.
Status DeviceToDeviceType(const string& device, DeviceType* device_type);

// Returns true if `node` has a ref tensor input that it forwards to its output.
bool HasForwardedRefInput(const Node& node);

// Creates a graph representation to enable cycle detection when clustering.
// This representation handles loops in graph by disconnecting each loop from
// the enclosing graph.
Status CreateCycleDetectionGraph(const Graph* graph, GraphCycles* cycles);

// Returns the XLA cluster in which `node` is placed if it is in an XLA cluster,
// otherwise returns nullopt.
absl::optional<absl::string_view> GetXlaClusterForNode(const Node& node);

// Removes `node_def` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(NodeDef* node_def);

// Removes `node` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(Node* node);

// Returns true if `node` has a DT_RESOURCE typed input or output.
bool HasResourceInputOrOutput(const Node& node);

// Adds edges to `cycles` to prevent clustering resource operations that cannot
// be legally clustered.
Status AdjustCycleDetectionGraphForResourceOps(
    const Graph* graph, const FunctionLibraryDefinition* flib_def,
    const std::function<Status(const Node&, bool*)>& resource_ops_to_ignore,
    GraphCycles* cycles);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
