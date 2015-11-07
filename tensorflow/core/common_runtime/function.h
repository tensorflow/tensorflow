#ifndef TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
#define TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_

#include <functional>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Creates a FunctionLibraryRuntime, which instantiates functions
// defined in "lib_def" and executes functions on the "device".
//
// The returned object does not take ownerships of "device" or
// "lib_def".  The caller must ensure "device" and "lib_def" outlives
// the returned object.
typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;
FunctionLibraryRuntime* NewFunctionLibraryRuntime(
    Device* device, Runner runner, const FunctionLibraryDefinition* lib_def);

// FunctionLibraryRuntime::GetFunctionBody returns a description of an
// instantiated function that is represented as a Graph with arg/ret
// nodes annotated.
struct FunctionBody {
  FunctionDef fdef;
  Graph* graph = nullptr;  // owned.
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  gtl::InlinedVector<Node*, 4> arg_nodes;
  gtl::InlinedVector<Node*, 4> ret_nodes;

  FunctionBody() {}
  FunctionBody(const FunctionDef& f, DataTypeSlice arg_types,
               DataTypeSlice ret_types, Graph* g);
  ~FunctionBody();
};

// Debugging facility.  Returns a debug string for a graph
// representing an instantiated function.
string DebugString(const Graph* instantiated_func_graph);

// A few hand-crafted optimization on the instantiated function body
// (a Graph*).

// Removes nodes that are
//   1. not stateful; and
//   2. not _Arg; and
//   3. not reachable from _Retval.
// Returns true iff any node is removed from "g".
bool RemoveDeadNodes(Graph* g);

// Find a pattern:
//   src -(in)-> node -(out)-> dst, where
// 1) node is an identity node;
// 2) in is the only incoming data edge;
// 3) out is the only outgoing data edge;
//
// Rewrites the above pattern with src->dst and relevant data
// dependencies updated. Repeat the process until no such pattern
// left.
bool RemoveIdentityNodes(Graph* g);

// Rewrites _ListToArray and _ArrayToList to a set of Identity nodes.
bool RemoveListArrayConverter(Graph* g);

// For each node in "graph", if "lib" indicates that the node is a
// function call, inline the function body.  Returns true if at least
// one node is inlined.
//
// This routine goes through "graph" nodes once and applies the
// inlining.  The caller may decide to apply the inlining on "graph"
// multiple times by calling ExpandInlineFunctions a few times.
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph);

// Applies graph rewrite optimzation such as inlining, dead code
// removal, etc.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
void OptimizeGraph(FunctionLibraryRuntime* lib, Graph** g);

// Given a numerical function "f", returns another numerical function
// "g", such that if "f" takes N inputs and produces M outputs, "g"
// takes N + M inputs and produces N outputs. I.e., if
//   (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
// g is a function which is
//   (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
//                                     dL/dy1, dL/dy2, ..., dL/dy_M),
// where L is a scalar-value function of (...x_i...).
//
// TODO(zhifengc): Asks math expert to say the comment again.
FunctionBody* SymbolicGradient(const FunctionBody& f);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
