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

#ifndef TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
#define TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_

#include <set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace graph_transforms {

// Used to quickly look up nodes in the graph def from a name.
void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result);

// For every node in the graph create a list of the nodes that use it as an
// input.
void MapNodesToOutputs(const GraphDef& graph_def,
                       std::map<string, std::vector<const NodeDef*>>* result);

// NodeDef input strings can contain other information besides the name of an
// input node. These include:
//  - Optional '^' prefix, indicating this is a control edge.
//  - The required name of the input node.
//  - Optional ':<number>' suffix, showing which output of the node to use.
// This function takes a raw string, and breaks it into those component parts.
// The rules for inputs in function libraries are a bit more complex, and
// aren't handled by this routine.
void NodeNamePartsFromInput(const string& input_name, string* prefix,
                            string* node_name, string* suffix);

// Adds a ':0' port to any inputs with no suffix, to make comparisons easier.
string CanonicalInputName(const string& input_name);

// Convenience function to strip the optional prefix and suffix components from
// a string pulled from a NodeDef input, and return the plain node name.
string NodeNameFromInput(const string& input_name);

// Returns a stable hash for the contents of the NodeDef, so that equivalent
// nodes should have equal hashes.
uint64 HashNodeDef(const NodeDef& node);

// Adds the given node name to the end of the node's inputs.
void AddNodeInput(const string& input_name, NodeDef* node);

// Copies an attribute from one NodeDef to another.
void CopyNodeAttr(const NodeDef& source, const string& source_key,
                  const string& dest_key, NodeDef* dest);

// Inserts a value into a NodeDef's map of attributes.
// This is a bit different than AddNodeAttr in node_def_util.h because it
// overwrites any existing attributes with the same key.
template <class T>
inline void SetNodeAttr(const string& key, const T& value, NodeDef* node) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  auto* attr_map = node->mutable_attr();
  (*attr_map)[key] = attr_value;
}

template <class T>
inline void SetNodeTensorAttr(const string& key, const Tensor& tensor,
                              NodeDef* node) {
  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  SetNodeAttr(key, tensor_proto, node);
}

// Inserts a Tensor into the specified attribute of a NodeDef.
template <class T>
inline void SetNodeTensorAttr(const string& key, const TensorShape& shape,
                              const std::vector<T>& values, NodeDef* node) {
  const DataType dtype = DataTypeToEnum<T>::v();
  CHECK_EQ(shape.num_elements(), values.size());
  Tensor tensor(dtype, shape);
  T* dest_data = tensor.flat<T>().data();
  std::copy_n(values.data(), values.size(), dest_data);
  SetNodeTensorAttr<T>(key, tensor, node);
}

// Retrieves a tensor value from a NodeDef attribute.
Tensor GetNodeTensorAttr(const NodeDef& node, const string& key);

// Creates a copy of the input GraphDef, but only containing the nodes where the
// supplied selector function returned true.
void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def);

// Creates a copy of the input graph, with all occurences of the attributes with
// the names in the argument removed from the node defs.
void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def);

// For a lot of replacement and matching operations it's useful to have the
// nodes processed in a controlled order, so this does a topological sort to
// ensure that nodes always appear in the GraphDef.node list after their inputs.
Status SortByExecutionOrder(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// Finds inputs that refer to nodes that are not in the graph.
void FindInvalidInputs(const GraphDef& graph_def,
                       std::vector<std::pair<string, string>>* invalid_inputs);

// Returns a descriptive error status if there are problems spotted with the
// graph.
Status IsGraphValid(const GraphDef& graph_def);

// This is used to spot particular subgraphs in a larger model. To use it,
// create a pattern like:
// OpTypePattern pattern({"Conv2D", {{"ResizeBilinear", {{"MirrorPad"}}}}});
// This defines a subgraph where a Conv2D has a ResizeBilinear input, which
// pulls from a MirrorPad op.
// Regular expressions aren't supported for the op names, but you can use "*" to
// match any op. You can also use | as a separator to match multiple op names,
// like "Reshape|Concat|Conv2D".
struct OpTypePattern {
  string op;
  std::vector<OpTypePattern> inputs;
  string DebugString() const;
};

// Returns a sub-graph of nodes that match a pattern.
struct NodeMatch {
  NodeMatch() : node() {}
  NodeDef node;
  std::vector<NodeMatch> inputs;
  string DebugString() const;
};

// Utility class to spot subgraphs matching particular patterns.
class GraphMatcher {
 public:
  GraphMatcher(const GraphDef& graph_def);

  // Sorts the input nodes into execution order, and then skips any previously
  // matches so that no node appears in more than one match. The NodeDef
  // pointers contained in the results are owned by the GraphMatcher object, and
  // so will be invalid after its lifetime.
  Status GetOpTypeMatches(const OpTypePattern& pattern,
                          std::vector<NodeMatch>* matches);

 private:
  bool DoesOpTypeMatch(const NodeDef& node, const OpTypePattern& pattern,
                       const std::set<string>& previously_matched_nodes,
                       NodeMatch* match);

  GraphDef graph_def_;
  std::map<string, const NodeDef*> node_map_;
};

struct ReplaceMatchingOpTypesOptions {
  // Whether to raise an error if the graph is left with dangling inputs. If you
  // enable this option, you must fix inconsistencies in a later pass.
  bool allow_inconsistencies;
};

// Replaces all of the matching sub-graphs with new ops. This calls into the
// given function, and expects to receive a set of new nodes to replace each
// matched sub-graph. It has some logic to protect the integrity of the
// resulting graph, for example making sure that nodes needed by other nodes
// outside the sub-graph aren't removed. These are passed in as the set of
// outputs, and nodes with the same names must be added to the new nodes
// produced by the replacement function. Many of these checks can be disabled
// by setting allow_inconsistencies to true in the options, but then it's the
// caller's responsibility to patch up any problems before passing on the graph
// to others. There's more comprehensive usage documentation in the README.
Status ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
        node_generator,
    const ReplaceMatchingOpTypesOptions& options, GraphDef* output_graph_def);

// Returns a list of the unique nodes found in this match.
void MatchedNodesAsArray(const NodeMatch& match, std::vector<NodeDef>* result);

// Changes all input references to a particular node name.
Status RenameNodeInputs(const GraphDef& input_graph_def,
                        const std::map<string, string>& inputs_to_rename,
                        GraphDef* output_graph_def);

// Utility function that copies all the nodes found in a match into the
// new_nodes list. This is useful in replacement functions when you decide to
// leave the original matched subgraph untouched and make no changes.
void CopyOriginalMatch(const NodeMatch& match, std::vector<NodeDef>* new_nodes);

// Holds information that's needed for transform functions.
typedef std::map<string, std::vector<string>> TransformFuncParameters;
struct TransformFuncContext {
  std::vector<string> input_names;
  std::vector<string> output_names;
  TransformFuncParameters params;
};

// Returns how many occurrences of the given parameter are present.
int CountParameters(const TransformFuncContext& context, const string& name);

// Gets a simple occurrence of a parameter, using a default if it isn't present.
Status GetExactlyOneParameter(const TransformFuncContext& context,
                              const string& name, const string& default_value,
                              string* result);

// This is the function API for all graph transformations, taking an input
// GraphDef and other arguments, and returning a transformed GraphDef.
typedef std::function<Status(const GraphDef&,
                             const TransformFuncContext& context, GraphDef*)>
    TransformFunc;

// To add a new graph transform function, call the macro:
// REGISTER_GRAPH_TRANSFORM("fold_constants", FoldConstants);
// Under the hood this adds the function to the list of known transforms, so you
// just need to link in the .cc file with your registration call to have access
// to it through the command line tool.
// The rest of the machinery below is to enable that automagical registration.
typedef std::map<string, TransformFunc> TransformRegistry;
TransformRegistry* GetTransformRegistry();
class TransformRegistrar {
 public:
  TransformRegistrar(const string& name, TransformFunc transform_func) {
    TransformRegistry* transform_registry = GetTransformRegistry();
    (*transform_registry)[name] = transform_func;
  }
};
#define REGISTER_GRAPH_TRANSFORM(name, func) \
  REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(__COUNTER__, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(ctr, name, func) \
  REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)    \
  static tensorflow::graph_transforms::TransformRegistrar \
      registrar__body__##ctr##__object(name, func);

}  // namespace graph_transforms
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
