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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks certain inputs to a Node as required to be a
// constant at compile time.  If this attribute is present then the
// CompileTimeConstantInput information in the corresponding XlaOpKernel is
// ignored.
//
// The value for this attribute, if present, has to be a list of strings naming
// the inputs to the node that must be constant.
extern const char* const kXlaCompileTimeConstantInputsAttr;

using OrderedNodeSet = std::set<Node*, NodeComparatorID>;

// Returns true if `node` has a ref tensor input that it forwards to its output.
bool HasForwardedRefInput(const Node& node);

// Creates a graph representation to enable cycle detection when clustering.
// This representation handles loops in graph by disconnecting each loop from
// the enclosing graph.
//
// Returns true for success and false for valid graphs that we can't handle yet
// (b/127521408).
StatusOr<bool> CreateCycleDetectionGraph(const Graph* graph,
                                         GraphCycles* cycles);

// Returns the XLA cluster in which `node` is placed if it is in an XLA cluster,
// otherwise returns nullopt.
std::optional<absl::string_view> GetXlaClusterForNode(const Node& node);

// Removes `node_def` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(NodeDef* node_def);

// Removes `node` its XLA cluster (by clearing its _XlaCluster attribute).
void RemoveFromXlaCluster(Node* node);

// Returns true if `node` has a DT_RESOURCE typed input or output.
bool HasResourceInputOrOutput(const Node& node);

// Determines the global jit level based on GraphOptimizationPassOptions,
// --tf_xla_auto_jit and whether the graph is a single GPU graph.
OptimizerOptions::GlobalJitLevel GetGlobalJitLevelForGraph(
    const GraphOptimizationPassOptions& options);

// Returns true if `g` is a single-GPU graph.  A single-GPU graph uses exactly
// one GPU (and any number of CPUs).
bool IsSingleGpuGraph(const Graph& g);

// Returns true if it is possible (but not guaranteed) that `n` calls a
// function.
bool MayCallFunction(const Node& n, const FunctionLibraryDefinition* flib_def);

// Returns true if `node` an operator that consumes only the shape of its input,
// not the data itself.
bool IsShapeConsumerOp(const Node& node);

// Computes a clustering summary for `graph`.  See documentation on
// `XlaAutoClusteringSummary` for details.
XlaAutoClusteringSummary GetXlaAutoClusteringSummary(const Graph& graph);

// Returns the set of nodes that have a path to or from nodes that may have ref
// variables as input or output.
//
// We assume each node has a trivial path to itself so the returned set includes
// all of the nodes that have ref variables as input or output.
StatusOr<absl::flat_hash_set<Node*>> GetNodesRelatedToRefVariables(
    const Graph& graph, FunctionLibraryRuntime* lib_runtime);

// Deterministically serialized the graph to a byte string.
StatusOr<std::string> SerializeGraphDeterministic(const Graph& graph);

// Computes a fingerprint of the given `graph`. The fingerprint can use used to
// check if two graphs are likely the same but should not be relied on
// determining if the graphs are identical.
StatusOr<uint64> FingerprintGraph(const Graph& graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_CLUSTER_UTIL_H_
