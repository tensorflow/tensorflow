#ifndef TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
#define TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_

#include "tensorflow/core/framework/graph.pb.h"
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

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
