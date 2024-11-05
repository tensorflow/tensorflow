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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {

// Returns the index of the first element in collection that fulfills predicate.
// If no such element exists, returns -1.
template <typename Predicate, typename Collection>
int GetFirstElementIndexWithPredicate(const Predicate& predicate,
                                      const Collection& collection) {
  unsigned idx = 0;
  for (auto&& element : collection) {
    if (predicate(element)) {
      return idx;
    }
    idx++;
  }
  return -1;
}

// Adds a node to the graph.
NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 MutableGraphView* graph);

// Adds Placeholder node for given type.
NodeDef* AddScalarPlaceholder(DataType dtype, MutableGraphView* graph);

// Adds a Const node with the given value to the graph.
template <typename T>
NodeDef* AddScalarConstNode(T v, MutableGraphView* graph) {
  // is_same is an idiomatic hack for making it compile if not instantiated.
  // Replacing with false will result in a compile-time error.
  static_assert(!std::is_same<T, T>::value,
                "Invalid specialization of this method for type T.");
  return {};
}

template <>
NodeDef* AddScalarConstNode(bool v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(double v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(float v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(int v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(int64_t v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(StringPiece v, MutableGraphView* graph);

// Retrieves the value of a const node. Returns an error
// if the node is not const, or its value is of a different type.
template <typename T>
absl::Status GetScalarConstNodeValue(const NodeDef& node, T* value) {
  // is_same is an idiomatic hack for making it compile if not instantiated.
  // Replacing with false will result in a compile-time error.
  static_assert(!std::is_same<T, T>::value,
                "Invalid specialization of this method fo rtype T.");
}

template <>
absl::Status GetScalarConstNodeValue(const NodeDef& node, int64_t* value);
template <>
absl::Status GetScalarConstNodeValue(const NodeDef& node, bool* value);

// Checks whether the two graphs are the same.
bool Compare(const GraphDef& g1, const GraphDef& g2);

// Checks whether the graph contains a node with the given name.
bool ContainsGraphNodeWithName(StringPiece name, const GraphDef& graph);

// Checks whether the library contains a function with the given name.
bool ContainsGraphFunctionWithName(StringPiece name,
                                   const FunctionDefLibrary& library);

// Checks whether the graph contains a node with the given op.
bool ContainsNodeWithOp(StringPiece op, const GraphDef& graph);

// Returns the index of the node with the given name or -1 if the node does
// not exist.
int FindGraphNodeWithName(StringPiece name, const GraphDef& graph);

// Returns the index of the function with the given name or -1 if the function
// does not exist.
int FindGraphFunctionWithName(StringPiece name,
                              const FunctionDefLibrary& library);

// Returns the index of the first node with the given op or -1 if no such  node
// exists.
int FindGraphNodeWithOp(StringPiece op, const GraphDef& graph);

// Gets the 0th input to a node in the graph.
NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph);

// Gets the ith input to a node in the graph.
NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph,
                      int64_t i);

// Gets the attr corresponding to a dataset node's output types, if it exists.
absl::Status GetDatasetOutputTypesAttr(const NodeDef& node,
                                       DataTypeVector* output_types);

// Returns the list of indices of all nodes with the given op or empty list if
// no such node exists.
std::vector<int> FindAllGraphNodesWithOp(const string& op,
                                         const GraphDef& graph);

// Sets the node name using `prefix` as a prefix while guaranteeing the name
// is unique across the graph.
void SetUniqueGraphNodeName(StringPiece prefix, GraphDef* graph, NodeDef* node);

// Sets the function name using the `prefix` name as a prefix while guaranteeing
// the name is unique across the function library.
void SetUniqueGraphFunctionName(StringPiece prefix,
                                const FunctionDefLibrary* library,
                                FunctionDef* function);

// Copies attribute having name `attribute_name` from node `from` to node
// `to_node`.
void CopyAttribute(const string& attribute_name, const NodeDef& from,
                   NodeDef* to_node);

// Concatenates list attribute having name `attribute_name` from `first` and
// `second` node, setting it to `to_node`.
void ConcatAttributeList(const string& attribute_name, const NodeDef& first,
                         const NodeDef& second, NodeDef* to_node);

// Checks that all nodes in the graphs have unique names, and sets their names
// to be unique if they are not already.  This is necessary as Graph does not
// have the provisions to deduplicate names, and name deduplication elsewhere
// in tensorflow happens in other layers (for example, in the Scope class of the
// C++ API). Note that the nodes in the graph are identified by their id,
// and renaming nodes does not mutate any edges.
absl::Status EnsureNodeNamesUnique(Graph* g);

// Returns the item's fetch node, if there is exactly one. Otherwise, returns an
// error.
absl::Status GetFetchNode(const MutableGraphView& graph,
                          const GrapplerItem& item, NodeDef** fetch_node);

// Returns true if `item` is derived from a `FunctionDef`, false otherwise.
// Currently, we determine this heuristically: If we don't have any fetch nodes
// or all fetch nodes are `Retval` ops, then we consider this item as derived
// from a `FunctionDef`.
bool IsItemDerivedFromFunctionDef(const GrapplerItem& item,
                                  const MutableGraphView& graph_view);

// If both input nodes have the "metadata" attribute set, it populates the
// "metadata" attribute for the fused node.
void MaybeSetFusedMetadata(const NodeDef& node1, const NodeDef& node2,
                           NodeDef* fused_node);

// Copies the attributes `output_shapes`, `output_types` from node `from` to
// node `to_node` if they exist. The method will return `true` if attributes
// copied successfully, otherwise it will return `false`.
//
// Some tf.data transformations set `Toutput_types` instead of `output_types`
// when the attribute describes type of tensor inputs (e.g. TensorDataset,
// TensorSliceDataset, and PaddedBatchDataset). In this case the method copies
// the attribute `Toutput_types` of node `from` to the attribute `output_types`
// of node `to_node`.
bool CopyShapesAndTypesAttrs(const NodeDef& from, NodeDef* to_node);

// Checks whether the op has a "sloppy" attribute.
bool HasSloppyAttr(const string& op);

// Checks whether the op has a "replicate_on_split" attribute.
bool HasReplicateOnSplitAttr(const string& op);

// Checks whether the op has a "deterministic" attribute.
bool HasDeterministicAttr(const string& op);

// Sets the `name` as the metadata name of the `node`. It returns an error if
// the `node` already has a metadata name.
absl::Status SetMetadataName(const std::string& name, NodeDef* node);

}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
