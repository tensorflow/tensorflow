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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_GRAPH_GRADIENTS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_GRAPH_GRADIENTS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// GradNodeOutput represents a single gradient node output.
struct GradNodeOutput {
  Node* node;
  int index;
};

// NOTE: This API is a work in progress and will likely be changing frequently.
//
// Given initial gradient nodes 'y_grad_nodes' (which compute the symbolic
// partial derivatives of some loss function 'L' w.r.t the inputs of each
// node in 'y_nodes'), adds gradient nodes to 'graph' that compute the sum
// of all gradients flowing into the single output of each node in 'x_nodes'.
// Note that gradient nodes will not be added to 'graph' which compute
// the symbolic partial derivative of 'L' w.r.t. each node in 'x_nodes' (i.e.
// backprop will stop at these nodes). This restriction will be lifted in
// a subsequent CL.
//
// REQUIRES: Each node in 'x_nodes' must have a single output (this
// restriction will be removed in a subsequent change).

// TODO(andydavis) Add support for returning 'x_node' gradients by endpoint
// (i.e. {node, index}).
// TODO(andydavis) Add symbolic gradient support for general graphs (the current
// implementation only supports gradients for functions). In particular,
// the nodes in 'x_nodes' are currently restricted to have one output.
Status AddSymbolicGradients(gtl::ArraySlice<Node*> y_nodes,
                            gtl::ArraySlice<Node*> x_nodes,
                            gtl::ArraySlice<Node*> y_grad_nodes,
                            std::vector<GradNodeOutput>* x_grad_nodes,
                            Graph* graph);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPH_GRADIENTS_H_
