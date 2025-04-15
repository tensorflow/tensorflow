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

#ifndef TENSORFLOW_COMPILER_TF2XLA_SIDE_EFFECT_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_SIDE_EFFECT_UTIL_H_

#include <vector>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Side-effecting nodes will have this attribute set. Its value is the list of
// node names which this node has side-effect dependencies on.
//
// Nodes like HostCompute, SendToHost, RecvFromHost always have this attribute,
// because they always have side-effect.
// If and While nodes may or may not have this attribute, depending on whether
// their bodies have side-effecting nodes.
extern const char kXlaTokenInputNodesAttrName[];

// This node name is used in kXlaTokenInputNodesAttrName attr to signal that a
// node has side-effect dependency on current graph's token input.
extern const char kXlaTokenArgNodeName[];

// This node have XlaRecvAtHost/XlaSendFromHost in its associated functions.
extern const char kXlaHasHostTransferAttrName[];

// This attribute is the replica id for an outside compilation node node.
extern const char kXlaReplicaIdAttrName[];

// This node is a Placeholder node added for tail outside compilation.
extern const char kXlaIsPlaceholderForTailOcAttrName[];

// This attribute is the original node name for this node.
extern const char kXlaOriginalOutsideCompilationNodeName[];

// Sets device ordinal attribute for nodes with attribute
// `kXlaHasHostTransferAttrName`.
absl::Status SetDeviceOrdinalAttributeForNode(Node* node, int device_ordinal);

// Calculates side-effect dependencies for the graph's token output.
// Returns a set of node names representing these dependencies.
std::set<std::string> CalculateTokenInputsForOutputToken(const Graph& g);

// Returns whether a graph contains side-effecting nodes.
bool HasSideEffectingNodes(const Graph& g);

// Parse the mapping from outside_compilation_subgraph name to core number,
// which is specified in an attr as a list of strings
// <subgraph_name>:<core_index>.
absl::Status ParseHostComputeCoreList(absl::Span<const string> list_from_attr,
                                      std::map<string, int>* host_compute_core);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_SIDE_EFFECT_UTIL_H_
