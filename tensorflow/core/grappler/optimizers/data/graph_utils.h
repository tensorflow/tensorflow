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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {

// Adds a node to the graph.
Status AddNode(const string& name, const string& op,
               const std::vector<string>& inputs,
               const std::vector<std::pair<string, AttrValue>>& attributes,
               GraphDef* graph, NodeDef** result);

// Adds a Const node with the given value to the graph.
template <typename T>
Status AddScalarConstNode(T v, GraphDef* graph, NodeDef** result) {
  return errors::Unimplemented("Type %s is not supported.",
                               DataTypeToEnum<T>::value);
}
template <>
Status AddScalarConstNode(bool v, GraphDef* graph, NodeDef** result);
template <>
Status AddScalarConstNode(double v, GraphDef* graph, NodeDef** result);
template <>
Status AddScalarConstNode(float v, GraphDef* graph, NodeDef** result);
template <>
Status AddScalarConstNode(int v, GraphDef* graph, NodeDef** result);
template <>
Status AddScalarConstNode(int64 v, GraphDef* graph, NodeDef** result);
template <>
Status AddScalarConstNode(StringPiece v, GraphDef* graph, NodeDef** result);

// Checks whether the two graphs are the same.
bool Compare(const GraphDef& g1, const GraphDef& g2);

// Checks whether the graph contains a node with the given name.
bool ContainsNodeWithName(const string& name, const GraphDef& graph);

// Checks whether the graph contains a node with the given op.
bool ContainsNodeWithOp(const string& op, const GraphDef& graph);

// Deletes nodes from the graph.
Status DeleteNodes(const std::set<string>& nodes_to_delete, GraphDef* graph);

// Returns the index of the node with the given name or -1 if the node does
// not exist.
int FindNodeWithName(const string& name, const GraphDef& graph);

// Returns the index of a node with the given op or -1 if no such  node
// exists.
int FindNodeWithOp(const string& op, const GraphDef& graph);

// Sets the node name using the op name as a prefix while guaranteeing the name
// is unique across the graph.
void SetUniqueName(const string& op, GraphDef* graph, NodeDef* node);

}  // end namespace graph_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
