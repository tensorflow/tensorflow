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

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Given a function like:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     if (opts.HaveError()) return nullptr;
//     static const string kOpName = "Identity";
//     NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
//                              opts.op_registry());
//     node_builder.Input(input);
//     return opts.FinalizeBuilder(&node_builder);
//   }
//   }  // namespace ops
//
//   // Or, alternatively:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     static const string kOpName = "Identity";
//     return UnaryOp(kOpName, input, opts);
//   }
//   }  // namespace ops
//
// You call it like:
//   GraphDefBuilder b;
//   using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
//   Node* na = Const(7, b.opts());
//   // Note: WithName() returns a copy, opts is unchanged.
//   Node* nb = Const(5, b.opts().WithName("control-input"));
//   Node* nc = Identity(na, b.opts().WithControlInput(nb));
//   GraphDef graph_def;
//   Status status = b.ToGraphDef(&graph_def);
//   if (!status.ok()) { /* Handle error */ }
//
// In tests you can skip the status handling via:
//   GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
//   ...
//   b.ToGraphDef(&graph_def);

class GraphDefBuilder {
 public:
  // Options for adding a Node to a Graph.
  class Options {
   public:
    // Sets the Graph (that Nodes will be added to) and the status.  The
    // status may be set to nullptr, in which case errors cause CHECK
    // failures.  The graph and status must outlive *this.
    Options(Graph* graph, Status* status);
    ~Options();

    // Methods for setting options.  These are const methods: they
    // return a copy of *this with the option set.
    Options WithName(StringPiece name) const;
    Options WithDevice(StringPiece device) const;
    Options WithControlInput(Node* control_input) const;
    Options WithControlInputs(absl::Span<Node* const> control_inputs) const;

    // Override the default value for an optional attr.
    template <class T>
    Options WithAttr(StringPiece attr_name, T&& value) const {
      return Options(*this).WithAttrImpl(attr_name, std::forward<T>(value));
    }
    // Note: overload needed to allow {...} expressions for value.
    template <class T>
    Options WithAttr(StringPiece attr_name,
                     std::initializer_list<T> value) const {
      return WithAttr<std::initializer_list<T>>(attr_name, std::move(value));
    }

    // Methods for using options from a function that creates a Node.

    // Returns true if the status associated with *this has an error.
    // Use this to skip processing that may depend on prior results.
    bool HaveError() const { return status_ != nullptr && !status_->ok(); }

    // Returns a string representation of the status associated with *this.
    // Returns the string `"OK"` if the status doesn't have any error.
    string StatusToString() const {
      return status_->ok() ? "OK" : std::string(status_->message());
    }

    // Given the Op type name, return a name for a node of that type.
    // Uses the value set in WithName() if that has been called.  Otherwise,
    // returns a name built out of the Op type name.
    string GetNameForOp(StringPiece op) const;

    // Sets the device, adds control inputs, adds attrs, and calls Finalize().
    // If Finalize returns an error, it is saved and this function returns
    // nullptr.
    Node* FinalizeBuilder(NodeBuilder* builder) const;

    // Updates the associated status, if any, or calls TF_CHECK_OK if none.
    void UpdateStatus(const Status& status) const;

    // Accessor
    const OpRegistryInterface* op_registry() const {
      return graph_->op_registry();
    }

   private:
    Options WithNameImpl(StringPiece name);
    Options WithDeviceImpl(StringPiece device);
    Options WithControlInputImpl(Node* control_input);
    Options WithControlInputsImpl(absl::Span<Node* const> control_inputs);
    template <class T>
    Options WithAttrImpl(StringPiece name, T&& value) {
      attrs_.emplace_back(string(name), AttrValue());
      SetAttrValue(std::forward<T>(value), &attrs_.back().second);
      return *this;
    }

    Graph* const graph_;
    Status* const status_;
    string name_;
    string device_;
    std::vector<Node*> control_inputs_;
    std::vector<std::pair<string, AttrValue>> attrs_;
  };

  // Start building a new graph.
  explicit GraphDefBuilder(
      const OpRegistryInterface* op_registry = OpRegistry::Global())
      : graph_(op_registry), flib_def_(op_registry), opts_(&graph_, &status_) {}

  // For use in tests, where you want to fail immediately on error instead
  // of checking the status at the end.
  enum TestFailImmediatelyType { kFailImmediately };
  explicit GraphDefBuilder(
      TestFailImmediatelyType,
      const OpRegistryInterface* op_registry = OpRegistry::Global())
      : graph_(op_registry), flib_def_(op_registry), opts_(&graph_, nullptr) {}

  // Gets the Options with the associated Graph and Status.
  const Options& opts() const { return opts_; }

  // Once all the nodes have been added, call this to get whether it was
  // successful, and if so fill *graph_def.
  Status ToGraphDef(GraphDef* graph_def) const;

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib) {
    return flib_def_.AddLibrary(fdef_lib);
  }

  // Returns whether a user-defined function with `name` already exists in the
  // graph.
  bool HasFunction(const string& name) {
    return flib_def_.Find(name) != nullptr;
  }

 private:
  Graph graph_;
  FunctionLibraryDefinition flib_def_;
  Status status_;
  Options opts_;
};

namespace ops {

// A NodeOut may either be a regular input or back input.  Regular
// inputs are specified via either a Node* or a Node* and an output
// index.  Back inputs are specified by a node name, output index, and
// output type.
typedef NodeBuilder::NodeOut NodeOut;

// For adding an Op with no inputs to a GraphDefBuilder.
Node* SourceOp(const string& op_name, const GraphDefBuilder::Options& opts);

// For adding an Op with one input to a GraphDefBuilder.
Node* UnaryOp(const string& op_name, NodeOut input,
              const GraphDefBuilder::Options& opts);

// For adding an Op with two inputs to a GraphDefBuilder.
Node* BinaryOp(const string& op_name, NodeOut a, NodeOut b,
               const GraphDefBuilder::Options& opts);

// For adding an Op with three inputs to a GraphDefBuilder.
Node* TernaryOp(const string& op_name, NodeOut a, NodeOut b, NodeOut c,
                const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_
