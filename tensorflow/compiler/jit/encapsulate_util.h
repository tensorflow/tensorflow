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

// This file contains some utility functions for encapsulating XLA computation
// in host graph and encapsulating outside compilation in XLA computation.

#ifndef TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Attribute marking output tensor shapes inferred by XLA. Attribute value is
// a list of PartialTensorShape objects.
extern const char kXlaInferredShapesAttrName[];

// Infers output shapes for all nodes in graph `g`. The output shapes will be
// stored in node attribute `kXlaInferredShapesAttrName`.
//
// We have to perform shape inference before encapsulation because after
// encapsulation, some nodes will be encapsulated into function call, and shape
// inference does not handle function call at the moment.
absl::Status PerformStaticShapeInferenceBeforeEncapsulation(Graph* g);

// Attribute indicating that some ops in this node's XLA computation has control
// dependency on this node. Attribute value will always be "true".
extern const char kXlaConnectedToXlaComputationAttrName[];

// Attribute indicating that this node has control dependency on some ops in
// this node's XLA computation. Attribute value will always be "true".
extern const char kXlaConnectedFromXlaComputationAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an outside compilation node. Attribute value will be
// string (original input node name).
extern const char kOutsideCompilationOriginalNodeAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an outside compilation node. Attribute value will be
// int (src_output for original edge).
extern const char kOutsideCompilationSrcOutputAttrName[];

// Attribute indicating that this node has control dependencies on some other
// nodes within the same XLA cluster. Attribute value will be a list of string
// (node names).
extern const char kXlaControlDependenciesWithinXlaClusterAttrName[];

// Attribute indicating that this node is an outside compilation node which is
// lifted out of If/While/function node. Attribute value will always be boolean
// value "true".
extern const char kXlaIsLiftedArgAttrName[];

// Attribute indicating that this node is a Placeholder node for an _Arg node
// lifted out of If/While/function node. Attribute value will be a string, which
// is the outside compilation cluster name sending the lifted arg node to host.
extern const char kXlaLiftedArgOutsideCompilationAttrName[];

// Attribute indicating that this is an IdentityN node receiving inputs for a
// outside compilation Placeholder node (the original outside compilation node
// is moved out of TPU computation, and we left a Placeholder node there).
// Attribute value will be a string, which is the outside compilation cluster
// name for the outside compilation Placeholder node.
extern const char kXlaOutsideCompilationInputsAttrName[];

// Attribute indicating that this is a Placeholder node for an _Arg node used in
// outside compilation. We should not move this node out of XLA computation.
// Attribute value will always be boolean value "true".
extern const char kXlaIsPlaceholderForArg[];

// Information for XLA computation.
struct XlaClusterInfo {
  // Add an explicitly-defined default constructor for this class.
  //
  // The compiler may delete the default constructor here because
  // host_compute_core is a const member whose type (std::map) doesn't
  // necessarily have a user provided constructor -- while libc++ and
  // libstdc++ 4.8 provide a user defined default constructor, libstdc++ at
  // least >= 7.3 does not. See also c++11 [class.ctor] p5.
  //
  // TODO(klimek): In c++17 we'll be able to initialize host_compute_core
  // without losing aggregate initialization, which allows us to get rid of
  // the constructor definitions again.
  XlaClusterInfo() {}
  XlaClusterInfo(const string& cluster_name,
                 const NameAttrList& func_name_attrs, Node* node,
                 const std::map<string, int>& host_compute_core)
      : cluster_name(cluster_name),
        func_name_attrs(func_name_attrs),
        node(node),
        host_compute_core(host_compute_core) {}
  // XLA cluster name. It might be different from `func_name`.
  const string cluster_name;
  // Name and attributes of XLA computation function.
  const NameAttrList func_name_attrs;
  // The XLA computation node in the graph.
  Node* node;
  // A mapping from outside compilation cluster name to its device assignment.
  const std::map<string, int> host_compute_core;
};

// Finds dependencies between outside compilation clusters, including both data
// dependencies and control dependencies. cluster_deps maps the name name of an
// outside compilation cluster to a set of names of outside compilation clusters
// that it depends on.
absl::StatusOr<
    std::unique_ptr<absl::flat_hash_map<string, std::vector<string>>>>
OutsideCompilationClusterDependencies(
    const Graph* g, const string& outside_compilation_attr_name);

// Preprocesses edges within the same XLA cluster. It will perform the following
// operations in order:
//
// 0.  Remove edges from source node to outside compilation nodes, and edges
//     from outside compilation nodes to sink node.
// 1a. For edges between different outside compilation clusters, remove the edge
//     and add attr "kXlaControlDependenciesWithinXlaClusterAttrName = src node
//     name" to dst node.
// 1b. For control edges between outside compilation and its XLA computation,
//     add attr "kXlaConnected{From, To}XlaComputationAttrName = true" to the
//     outside compilation node.
// 2.  For data edges between different outside compilations, remove the edge
//     and create a Placeholder node as dst node's input.
absl::Status PreprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name);

// Postprocesses edges within the same XLA cluster. This function reverts what
// `PreprocessEdgesBetweenOutsideCompilations` did. It will perform the
// following operations in order:
//
// 1. Remove Placeholder nodes between different outside compilations (created
//    in `PreprocessEdgesBetweenOutsideCompilations` step 2).
// 2a. Reconnect control edges between different outside compilations (marked by
//     `PreprocessEdgesBetweenOutsideCompilations` step 1a).
// Notice that control edges marked by
// `PreprocessEdgesBetweenOutsideCompilations` step 1b are not handled here.
// They are handled in `RewriteOutsideCompilationSubgraphFn`.
absl::Status PostprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
