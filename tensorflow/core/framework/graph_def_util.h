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

#ifndef TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
#define TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

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
Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def);

// Adds default attributes to NodeDefs in 'graph_def' starting
// from the 'node_offset' node in 'graph_def'.
//
// Default attributes are defined by 'op_registry'.
//
// Returns OK on success, an error if 'graph_def' has a NodeDef
// that cannot be found in 'op_registry'.
//
// REQUIRES: 'graph_def' and 'op_registry' are not nullptr.
Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                 const OpRegistryInterface* op_registry,
                                 int node_offset);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
