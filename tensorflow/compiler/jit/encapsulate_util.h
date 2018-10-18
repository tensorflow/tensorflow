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

// Attribute indicating that some ops in this node's XLA computation has control
// dependency on this node. Attribute value will always be "true".
extern const char kXlaConnectedToXlaComputationAttrName[];

// Attribute indicating that this node has control dependency on some ops in
// this node's XLA computation. Attribute value will always be "true".
extern const char kXlaConnectedFromXlaComputationAttrName[];

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

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for an host node. Attribute value will be string
// (original input node name).
extern const char kHostToOutsideCompilationOriginalNodeAttrName[];

// Attribute indicating that this is an Placeholder node added to act as a
// temporary input node for a host node. Attribute value will be int (src_output
// for original edge).
extern const char kHostToOutsideCompilationSrcOutputAttrName[];

// Preprocesses the graph for encapsulation. It will perform the following
// operations in order:
//
// 1a. For control edges between outside compilation and its XLA computation,
//     add attr "kXlaConnected{From, To}XlaComputationAttrName = true" to the
//     outside compilation node.
// 1b. For control edges between outside compilation and another XLA
//     computation, add attr "kXlaConnected{From, To}OtherXlaComputationAttrName
//     = XLA computation node name" to the outside compilation node.
// 1c. For control edges between different outside compilations, remove the edge
//     and add attr "kXlaControlDependenciesAttrName = src node name" to dst
//     node.
// 1d. For control edges between outside compilation and host computation,
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

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
