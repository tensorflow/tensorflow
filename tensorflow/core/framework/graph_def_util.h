/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_UTIL_H_

#include <set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Forward declare proto so that it's symbols can be removed from .so exports
class GraphDef;
class NodeDef;

// Produce a human-readable version of a GraphDef that is more concise
// than a text-format proto.
string SummarizeGraphDef(const GraphDef& graph_def);

// Validates the syntax of a GraphDef provided externally.
//
// The following is an EBNF-style syntax for GraphDef objects. Note that
// Node objects are actually specified as tensorflow::NodeDef protocol buffers,
// which contain many other fields that are not (currently) validated.
//
// Graph        = Node *
// Node         = NodeName, Inputs
// Inputs       = ( DataInput * ), ( ControlInput * )
// DataInput    = NodeName, ( ":", [1-9], [0-9] * ) ?
// ControlInput = "^", NodeName
// NodeName     = [A-Za-z0-9.], [A-Za-z0-9_./] *
absl::Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def);

// Adds default attributes to NodeDefs in 'graph_def' starting
// from the 'node_offset' node in 'graph_def'.
//
// Default attributes are defined by 'op_registry'.
//
// Returns OK on success, an error if 'graph_def' has a NodeDef
// that cannot be found in 'op_registry'.
//
// REQUIRES: 'graph_def' and 'op_registry' are not nullptr.
absl::Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                       const OpRegistryInterface& op_registry,
                                       int node_offset);

// Same as above, except for the fact that it skips nodes that aren't found in
// op_registry if skip_unknown_ops is true.
absl::Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                       const OpRegistryInterface& op_registry,
                                       int node_offset, bool skip_unknown_ops);

// Remove attrs from 'graph_def' that have the default value according
// to 'producer_op_registry', but don't exist according to
// 'consumer_op_registry'. This can allow 'graph_def' to run on the
// consumer even if consumer was built at an earlier CL (before an
// attr with a default was added). Note that this will not affect
// attrs with non-default values, so you must run a
// ValidateGraphDef...() function to see if the result is in fact
// compatible. If not nullptr, the op/attr pairs that were removed
// are added to '*op_attr_removed'.
//
// Expected usage, for a producer that wants to prepare a graph for
// a consumer:
// // For each consumer, update 'graph_def':
//   OpListOpRegistry consumer_op_registry(consumer_server_op_list);
//   std::unordered_set<std::pair<string, string>> op_attr_removed;
//   TF_RETURN_IF_ERROR(RemoveNewDefaultAttrsFromGraphDef(
//       &graph_def, consumer_op_registry, *OpRegistry::Global(),
//       &op_attr_removed));
// // Validate that each consumer can understand the resulting 'graph_def'
//   TF_RETURN_IF_ERROR(graph::ValidateGraphDefAgainstOpRegistry(
//       graph_def, consumer_op_registry));
// // Consumer can use 'graph_def', and 'op_attr_removed' summarizes
// // what changes had to be made to 'graph_def' for it to work.
//
// Expected usage, for a consumer that has a graph and a
// (optionally-stripped) op_list from a producer (say from a call to
// StrippedOpListForGraph(), or in the MetaGraphDef):
//   OpListOpRegistry producer_op_registry(producer_stripped_op_list);
//   TF_RETURN_IF_ERROR(RemoveNewDefaultAttrsFromGraphDef(
//       &graph_def, *OpRegistry::Global(), producer_op_registry, nullptr));
absl::Status RemoveNewDefaultAttrsFromGraphDef(
    GraphDef* graph_def, const OpRegistryInterface& consumer_op_registry,
    const OpRegistryInterface& producer_op_registry,
    std::set<std::pair<string, string>>* op_attr_removed);

// Goes over the `nodes` and removes attributes that are set to their
// default values according to op_registry.
// If some node's definition is not found in the `op_registry`, this node is
// simply skipped. In most cases, these nodes would be function calls.
// If a stricter behavior is desired, one can add FunctionLibraryDefinition
// argument to check for functions and their attributes.
// This is obvious from signature, but as a warning, if `nodes` contain
// nodes calling functions, e.g. PartitionCallOp or FunctionalIf, this
// function does not "recurse" into them.
void StripDefaultAttributes(const OpRegistryInterface& op_registry,
                            protobuf::RepeatedPtrField<NodeDef>* nodes);

// Two functions that collect the ops used by a graph.
//
// This returns the ops used as a set of strings.
void OpsUsedByGraph(const GraphDef& graph_def,
                    std::set<string>* ops_used_in_graph);

// This function computes the stripped_op_list field of MetaGraphDef
// and similar protos.  The op_registry should contain the ops used to
// produce graph_def.  The resulting stripped_op_list can be
// communicated from the producer to the consumer, which can use
// RemoveNewDefaultAttrsFromGraphDef() to improve forwards compatibility
// (using an OpListOpRegistry as indicated in the example above).
//
// Most users will pass *OpRegistry::Global() for op_registry to strip against
// the list of ops registered in this process.
absl::Status StrippedOpListForGraph(const GraphDef& graph_def,
                                    const OpRegistryInterface& op_registry,
                                    OpList* stripped_op_list);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_GRAPH_DEF_UTIL_H_
