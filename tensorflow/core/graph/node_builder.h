#ifndef TENSORFLOW_GRAPH_NODE_BUILDER_H_
#define TENSORFLOW_GRAPH_NODE_BUILDER_H_

#include <vector>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

// This is a helper for creating a Node and adding it to a Graph.
// Internally, it uses a NodeDefBuilder to automatically set attrs
// that can be inferred from the inputs, and use default values
// (where they exist) for unspecified attrs.  Example usage:
//
//  Node* node;
//  Status status = NodeBuilder(node_name, op_name)
//                           .Input(...)
//                           .Attr(...)
//                           .Finalize(&graph, &node);
//  if (!status.ok()) return status;
//  // Use node here.
class NodeBuilder {
 public:
  // For specifying the output of a Node to provide to one of the Input()
  // functions below.  It supports both regular inputs (where you are
  // connecting to an existing Node*), and inputs from outside the graph
  // (or haven't been added to the graph yet, like back edges, where
  // you don't have a Node*). Both types can be mixed, e.g. in an
  // ArraySlice.
  struct NodeOut {
    // For referencing an existing Node.
    NodeOut(Node* n, int i = 0)  // NOLINT(runtime/explicit)
        : node(n),
          error(false),
          name(node != nullptr ? node->name() : (error = true, "")),
          index(i),
          dt(SafeGetOutput(node, i, &error)) {}

    // For referencing Nodes not in the graph being built. It is
    // useful when preparing a graph for ExtendSession or creating a
    // back edge to a node that hasn't been added to the graph yet,
    // but will be.
    NodeOut(const string& name, int i, DataType t)
        : node(nullptr), error(false), name(name), index(i), dt(t) {}

    // Default constructor for std::vector<NodeOut>.
    NodeOut() {}

    Node* node = nullptr;
    // error is set to true if:
    // * the NodeOut was default constructed and never overwritten,
    // * a nullptr Node* was passed to the NodeOut constructor, or
    // * an out-of-range index was passed to the NodeOut constructor.
    bool error = true;
    string name;
    int index = 0;
    DataType dt = DT_FLOAT;
  };

  // Specify the name and the Op (either via an OpDef or the name of
  // the Op plus a registry) for the Node.  Other fields are
  // specified by calling the methods below.
  // REQUIRES: The OpDef must satisfy ValidateOpDef().
  NodeBuilder(const string& name, const string& op_name,
              const OpRegistryInterface* op_registry = OpRegistry::Global());
  NodeBuilder(const string& name, const OpDef* op_def);

  // You must call one Input() function per input_arg in the Op,
  // *and in the same order as the input_args appear in the OpDef.*

  // For inputs that take a single tensor.
  NodeBuilder& Input(Node* src_node, int src_index = 0);
  NodeBuilder& Input(NodeOut src);

  // For inputs that take a list of tensors.
  NodeBuilder& Input(gtl::ArraySlice<NodeOut> src_list);

  // Require that this node run after src_node(s).
  NodeBuilder& ControlInput(Node* src_node);
  NodeBuilder& ControlInputs(gtl::ArraySlice<Node*> src_nodes);

  // Sets the "requested device spec" in the NodeDef (not the
  // "assigned device" in the Node).
  NodeBuilder& Device(const string& device_spec);

  // Set the value of an attr.  attr_name must match the name of one of
  // attrs defined by the Op, and value must have the corresponding type
  // (see SetAttrValue() in ../framework/attr_value_util.h for legal
  // types for value).  Note that attrs will be set automatically if
  // they can be determined by the inputs.
  template <class T>
  NodeBuilder& Attr(const string& attr_name, T&& value);
  template <class T>
  NodeBuilder& Attr(const string& attr_name, std::initializer_list<T> value);

  // Validates the described node and adds it to *graph, adding edges
  // for all (non-back) inputs.  If created_node is not nullptr,
  // *created_node will be set to the new node (or nullptr on error).
  Status Finalize(Graph* graph, Node** created_node) const;

 private:
  static DataType SafeGetOutput(Node* node, int i, bool* error) {
    if (node != nullptr && i >= 0 && i < node->num_outputs()) {
      *error = false;
      return node->output_type(i);
    } else {
      *error = true;
      return DT_FLOAT;
    }
  }

  // If SafeGetOutput indicates a range error, add it to errors_.
  void AddIndexError(Node* node, int i);

  // Set *dt and returns true if i is in range. Combines
  // SafeGetOutput() and AddIndexError().
  bool GetOutputType(Node* node, int i, DataType* dt);

  NodeDefBuilder def_builder_;
  std::vector<NodeOut> inputs_;
  std::vector<Node*> control_inputs_;
  std::vector<string> errors_;
};

// IMPLEMENTATION -------------------------------------------------------------

template <class T>
inline NodeBuilder& NodeBuilder::Attr(const string& attr_name, T&& value) {
  def_builder_.Attr(attr_name, std::forward<T>(value));
  return *this;
}

template <class T>
NodeBuilder& NodeBuilder::Attr(const string& attr_name,
                               std::initializer_list<T> value) {
  def_builder_.Attr(attr_name, value);
  return *this;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_NODE_BUILDER_H_
