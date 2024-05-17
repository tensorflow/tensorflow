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

#ifndef TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_
#define TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "xla/status_macros.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Rewrite function for outside compilation subgraphs. It will perform the
// following steps:
//
// 1. Add a XLA computation key placeholder node (it will be used as input for
//    XlaRecvAtHost and XlaSendFromHost);
// 2. Replace all _Arg nodes with one single XlaRecvAtHost node;
// 3. Replace all _Retval nodes with one single XlaSendFromHost node;
// 4. Mark all nodes except key placeholder with attr `xla_cluster_attr_name`
//    and `outside_compilation_attr_name`;
// 5. For nodes marked with attr kXlaConnectedToXlaComputationAttrName, add a
//    control edge from the node to XlaSendFromHost; for nodes marked with attr
//    kXlaConnectedFromXlaComputationAttrName, add a control edge from
//    XlaRecvAtHost node to the node;
// 6. Try pruning XlaRecvAtHost/XlaSendFromHost/key placeholder node.
// 7. Add necessary attributes to `node_def`, so we can replace it with a
//    XlaHostCompute node later. If all input shapes for XlaSendFromHost are
//    known, "shapes" attr will be set to the list of input shapes; otherwise
//    "shape_inference_graph" attr will be set to shape inference function name.
class RewriteOutsideCompilationSubgraphFn {
 public:
  RewriteOutsideCompilationSubgraphFn(
      const string& xla_cluster_attr_name,
      const string& outside_compilation_attr_name,
      const string& xla_cluster_name, const string& new_function_name)
      : xla_cluster_attr_name_(xla_cluster_attr_name),
        outside_compilation_attr_name_(outside_compilation_attr_name),
        xla_cluster_name_(xla_cluster_name),
        new_function_name_(new_function_name) {}

  Status operator()(const std::vector<OutputTensor>&,
                    std::unique_ptr<Graph>* graph,
                    std::vector<int>* input_permutation,
                    std::vector<int>* output_permutation, NodeDef* node_def);

 private:
  string xla_cluster_attr_name_;
  string outside_compilation_attr_name_;
  string xla_cluster_name_;
  string new_function_name_;
};

// For an XLA computation function, replace all outside compilations with
// XlaHostCompute nodes. Each outside compilation subgraph will be rewritten by
// `RewriteOutsideCompilationSubgraphFn`, and they will be merged into one
// single host side graph (`host_graph`).
//
// xla_cluster_attr_name and outside_compilation_attr_name: attr name for XLA
//   computation and outside compilation. Required for
//   `RewriteOutsideCompilationSubgraphFn`.
// xla_cluster_name: XLA cluster name for this XLA computation. We need it
//   because XLA cluster name might be different from `func_name`.
// func_name_attrs: they will be used to instantiate the XLA computation func.
// new_func_name: new function name for rewritten XLA computation func.
// host_compute_core: mapping from outside compilation cluster name to XLA
//   device assignment.
// fld: FunctionLibraryDefinition object.
// host_graph: Graph object to store host side graph for all outside
//   compilations within this XLA computation func. If there is no outside
//   compilation, it will be empty.
// shape_inference_graphs: a list of outside compilation shape inference
//   function names. These functions need to be rewritten later.
// has_outside_compilation: a bool indicating whether this function has any
//   outside compilation nodes.
Status ExtractOutsideCompilationForFunction(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const NameAttrList& func_name_attrs, const string& new_func_name,
    const string& host_graph_func_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation);

// Rewrites XLA computation in `clusters` to replace outside compilation nodes
// with XlaHostCompute, and moves those outside compilations into `g`. If shapes
// of outside compilation outputs cannot be determined now, we will store shape
// inference graph into `fld`.
Status ExtractOutsideCompilation(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    bool* modified);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_EXTRACT_OUTSIDE_COMPILATION_PASS_H_
