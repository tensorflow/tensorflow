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

#ifndef TENSORFLOW_CORE_GRAPH_VALIDATE_H_
#define TENSORFLOW_CORE_GRAPH_VALIDATE_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace graph {

// Returns OK if every NodeDef in `graph_def` is valid with respect to
// its corresponding OpDef (as defined by ValidateNodeDef()) as
// registered in `op_registry`.  Also checks for deprecated ops.
//
// REQUIRES:
//  * `op_registry` is not nullptr.
//  * `graph_def` has default attrs filled in (see AddDefaultAttrsToGraphDef()).
Status ValidateGraphDef(const GraphDef& graph_def,
                        const OpRegistryInterface& op_registry);

// Like ValidateGraphDef() except it makes a copy of `graph_def` and calls
// AddDefaultAttrsToGraphDef() on the copy, removing that requirement from the
// caller.
Status ValidateGraphDefAgainstOpRegistry(
    const GraphDef& graph_def, const OpRegistryInterface& op_registry);

// Like ValidateGraphDefAgainstOpRegistry() except it takes an OpList
// instead of an OpRegistryInterface.  Note that the OpList need not
// have descriptions, which can be a big space savings, see
// GetOpListForValidation() below.
Status ValidateGraphDefAgainstOpList(const GraphDef& graph_def,
                                     const OpList& op_list);

// Get an OpList from `*op_registry` with all the descriptions removed.
void GetOpListForValidation(
    OpList* op_list, const OpRegistry& op_registry = *OpRegistry::Global());

// Validate that the graph has no cycle except for legal while loop cycles.
// This traverses the specified nodes in topological order to verify there are
// no cycles. Starting with inputless nodes, it visits nodes whose inputs have
// all been visited, and counts the total number of visited nodes. If there is a
// cycle, nodes in the cycle will never be visited, and the visited count will
// be less than the total node count.
Status ValidateGraphHasNoCycle(const Graph& graph);

}  // namespace graph
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_VALIDATE_H_
