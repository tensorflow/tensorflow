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

#ifndef TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
#define TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

static constexpr const char* const kNoInlineAttr = "_noinline";

// Registers a default customizable kernel creator for a function call.
//
// If 'cb()' returns a non-OK, we still fall back to an executor-based
// interpreter op kernel to execute a function. If 'cb()' returns OK,
// takes ownership of the returned OpKernel.
//
// TODO(zhifengc/phawkins): b/32379046
void RegisterDefaultCustomKernelCreator(CustomKernelCreator cb);

// Creates a FunctionLibraryRuntime, which instantiates functions
// defined in "lib_def" and executes functions on the "device".
// "device_mgr" must contain the "device". If not nullptr,
// "custom_kernel_creator" is consulted by the returned runtime to
// create kernels.
//
// The returned object does not take ownerships of "device" or
// "lib_def".  The caller must ensure "device" and "lib_def" outlives
// the returned object.
//
// The "parent" is a pointer to the ProcessFunctionLibraryRuntime object that
// typically owns the created FunctionLibraryRuntime object. The parent pointer
// is not owned by the FunctionLibraryRuntime object.
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, Device* device,
    int graph_def_version, const FunctionLibraryDefinition* lib_def,
    thread::ThreadPool* thread_pool, const OptimizerOptions& optimizer_options,
    CustomKernelCreator custom_kernel_creator,
    ProcessFunctionLibraryRuntime* parent);

// Same as above except that the returned runtime consults with the
// global default custom kernel creator registered by
// RegisterDefaultCustomKernelCreator.
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, Device* device,
    int graph_def_version, const FunctionLibraryDefinition* lib_def,
    thread::ThreadPool* thread_pool, const OptimizerOptions& optimizer_options,
    ProcessFunctionLibraryRuntime* parent);

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

// Dump the contents of the "graph" to log files if the logging level is
// sufficiently high.
void DumpGraph(StringPiece label, const Graph* g);

// Applies graph rewrite optimization such as inlining, dead code
// removal, etc.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g);

// Convert the Graph of a function to a GraphDef.
//
// Handles renaming of nodes to avoid duplicate names which may
// be present after various rewriting operations.
void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty = false);

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

// Given a "caller" in graph "g", which is a function call of a function
// to "fbody". Replaces the "caller" with fbody->graph and connects
// edges properly.
void InlineFunctionBody(const FunctionLibraryDefinition& flib_def, Graph* g,
                        Node* caller, const FunctionBody* fbody);

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef.
Status FunctionDefToBodyHelper(
    const FunctionDef& fdef, const AttrSlice& attrs,
    const FunctionLibraryDefinition* const lib_def,
    const std::function<Status(const string&, const OpDef**)>& get_func_sig,
    FunctionBody** fbody);
}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
