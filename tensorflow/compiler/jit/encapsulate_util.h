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

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Attribute marking output tensor shapes inferred by XLA. Attribute value is
// a list of PartialTensorShape objects.
extern const char kXlaInferredShapesAttrName[];

// Infer output shapes for outside compilation nodes which have output data
// edges to XLA computation nodes. These shapes will be used later by XLA
// compiler as output shapes of the outside compilation's XlaHostCompute op.
// XLA computation nodes will be mark by attr `xla_computation_attr_name`;
// outside compilation nodes will be marked by both attr
// `xla_computation_attr_name` and `outside_compilation_attr_name`.
//
// Those outside compilation nodes will be marked with attribute
// `kXlaInferredShapesAttrName`.
//
// We have to perform shape inference before encapsulation because after
// encapsulation, some nodes will be encapsulated into function call, and shape
// inference does not handle function call at the moment.
Status PerformStaticShapeInferenceBeforeEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name);

// Attribute indicating that some ops in other XLA computation has control
// dependency on this node. Attribute value will be a list of string (XLA
// computation names).
extern const char kXlaConnectedToOtherXlaComputationAttrName[];

// Attribute indicating that this node has control dependency on some ops in
// other XLA computation. Attribute value will be a list of string (XLA
// computation names).
extern const char kXlaConnectedFromOtherXlaComputationAttrName[];

// Attribute indicating that this node has control dependencies on some other
// nodes. Attribute value will be a list of string (node names).
extern const char kXlaControlDependenciesAttrName[];

// Attribute indicating that this is an Identity node added to act as a bridge
// between different XLA computations. Attribute value will be string (source
// node name).
extern const char kBridgeSourceNodeAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an outside compilation node. Attribute value will be
// string (original input node name).
extern const char kOutsideCompilationToHostOriginalNodeAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an outside compilation node. Attribute value will be
// int (src_output for original edge).
extern const char kOutsideCompilationToHostSrcOutputAttrName[];

// Attribute indicating that some ops in this node's XLA computation has control
// dependency on this node. Attribute value will always be "true".
extern const char kXlaConnectedToXlaComputationAttrName[];

// Attribute indicating that this node has control dependency on some ops in
// this node's XLA computation. Attribute value will always be "true".
extern const char kXlaConnectedFromXlaComputationAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an host node. Attribute value will be string
// (original input node name).
extern const char kHostToOutsideCompilationOriginalNodeAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for a host node. Attribute value will be int (src_output
// for original edge).
extern const char kHostToOutsideCompilationSrcOutputAttrName[];

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

// Preprocesses edges between different XLA clusters for encapsulation. It will
// perform the following operations in order:
//
// 1a. For control edges between outside compilation and another XLA
//     computation, add attr "kXlaConnected{From, To}OtherXlaComputationAttrName
//     = XLA computation node name" to the outside compilation node.
// 1b. For control edges between different outside compilations (in different
//     XLA computations), remove the edge and add attr
//     "kXlaControlDependenciesAttrName = src node name" to dst node.
// 1c. For control edges between outside compilation and host computation,
//     remove the edge and add attr "kXlaControlDependenciesAttrName = src node
//     name" to dst node.
// 2. For data edges between different XLA computations, if either src or dst
//    is outside compilation, add an Identity node in between the edge. The
//    identity node will have attr kBridgeSourceNodeAttrName.
// 3. For data edges between outside compilation and host computation, remove
//    the edge and create a Placeholder node as dst node's input.
Status PreprocessForEncapsulation(Graph* g,
                                  const string& xla_computation_attr_name,
                                  const string& outside_compilation_attr_name);

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

// Postprocesses edges between different XLA clusters for encapsulation. This
// function reverts what `PreprocessForEncapsulation` did. It will perform the
// following operations in order:
//
// 1. Remove Placeholder nodes between outside compilation and host computation
//     (created in `PreprocessForEncapsulation` step 3).
// 2. Remove Identity nodes created in `PreprocessForEncapsulation` step 2.
// 3a. Reconnect control edges between outside compilation and another XLA
//     computation (marked by `PreprocessForEncapsulation` step 1a).
// 3b. Reconnect control edges between different outside compilations (marked by
//     `PreprocessForEncapsulation` step 1b).
// 3c. Reconnect control edges between outside compilation and host computation
//     (marked by `PreprocessForEncapsulation` step 1c).
Status PostprocessForEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters);

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
Status PreprocessEdgesBetweenOutsideCompilations(
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
Status PostprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
