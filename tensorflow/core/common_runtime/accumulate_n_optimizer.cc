/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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


#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/node_builder.h"


namespace tensorflow {
namespace {

Tensor make_zeros(const DataType& dtype, const TensorShapeProto& shape) {
  Tensor tensor(dtype, TensorShape(shape));

  // Conveniently, all numeric data types have 0x0 == zero.  Otherwise we would
  // need a giant switch statement here.
  memset(const_cast<char*>(tensor.tensor_data().data()), 0,
         tensor.tensor_data().size());

  return tensor;
}

// Replaces occurrences of the "AccumulateNV2" stub operator with a graph of
// lower-level ops. The graph is equivalent (modulo certain corner cases)
// to the semantics of the original accumulate_n() Python op in math_ops.py.
// Implementing the op with a rewrite allows this new variant of accumulate_n
// to be differentiable.
//
// The binary code that generates AccumulateNV2 stub ops is located in a
// dynamic library built out of tensorflow/contrib/framework. Ideally, this
// class would also be in contrib, but calls to REGISTER_OPTIMIZATION() from
// third-party libraries aren't currently supported.
class AccumulateNV2RemovePass : public GraphOptimizationPass {
 public:

  Status Run(const GraphOptimizationPassOptions& options) override {
    // TODO(freiss.oss@gmail.com): Substantial shared code with
    // ParallelConcatRemovePass::Run(). Consider refactoring if someone makes
    // a third similar rewrite.
    if (options.graph == nullptr) {
      // TODO(apassos) returning OK feels weird here as we can't do anything
      // without a graph, but some tests require this.
      return Status::OK();
    }

    Graph* g = options.graph->get();
    if (g == nullptr) {
      return errors::Internal(
          "AccumulateNV2 removal should happen before partitioning and a "
          "graph should be available.");
    }

    // Build up a todo list of ops to replace, *then* modify the graph
    gtl::InlinedVector<Node*, 2> matches;
    for (Node* n : g->op_nodes()) {
      if (n->type_string() == "AccumulateNV2") {
        matches.push_back(n);
      }
    }
    for (Node* n : matches) {
      TF_RETURN_IF_ERROR(rewriteNode(n, g));
    }
    return Status::OK();
  }

  Status rewriteNode(Node* n, Graph* g) {
    AttrSlice n_attrs = n->attrs();
    auto base_make_node = [n, g, &n_attrs](const string& op,
                                           const string& name) {
      NodeBuilder node_builder(name, op);

      // The pieces of AccumulateNV2 should all be on the same node.
      node_builder.Device(n->requested_device());
      string colo;
      if (GetNodeAttr(n_attrs, kColocationAttrName, &colo).ok()) {
        node_builder.Attr(kColocationAttrName, colo);
      }
      return node_builder;
    };
    auto make_node = [n, g, &n_attrs, &base_make_node](string op) {
      return base_make_node(
          op, g->NewName(strings::StrCat(n->name(), "/Internal")));
    };

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    TensorShapeProto shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));

    std::vector<const Edge*> data_edges, control_edges;
    for (const Edge* input_edge : n->in_edges()) {
      if (input_edge->IsControlEdge()) {
        control_edges.push_back(input_edge);
      } else {
        data_edges.push_back(input_edge);
      }
    }

    // Create the following ops to replace the AccumulateNV2 placeholder:
    Node* create_accumulator = nullptr;            // TemporaryVariable op
    Node* initial_val = nullptr;                   // Const op
    Node* initialize_accumulator = nullptr;        // Assign op
    std::vector<Node*> add_values_to_accumulator;  // AssignAdd ops
    Node* clean_up_accumulator = nullptr;          // DestroyTemporaryVariable

    const string accumulator_name =
        strings::StrCat(n->name(), "/Internal/Accumulator");
    TF_RETURN_IF_ERROR(make_node("TemporaryVariable")
                           .Attr("shape", shape)
                           .Attr("dtype", dtype)
                           .Attr("var_name", accumulator_name)
                           .Finalize(g, &create_accumulator));
    TF_RETURN_IF_ERROR(make_node("Const")
                           .Attr("value", make_zeros(dtype, shape))
                           .Attr("dtype", dtype)
                           .Finalize(g, &initial_val));
    TF_RETURN_IF_ERROR(make_node("Assign")
                           .Attr("T", dtype)
                           .Input(create_accumulator)  // ref: Ref(T)
                           .Input(initial_val)         // value: T
                           .Finalize(g, &initialize_accumulator));
    for (int i = 0; i < data_edges.size(); ++i) {
      Node* assignAdd;
      TF_RETURN_IF_ERROR(make_node("AssignAdd")
                             .Attr("T", dtype)
                             .Attr("use_locking", true)
                             .Input(initialize_accumulator)  // ref: Ref(T)
                             .Input(data_edges[i]->src(),
                                    data_edges[i]->src_output())  // value: T
                             .Finalize(g, &assignAdd));

      add_values_to_accumulator.push_back(assignAdd);
    }

    // Note that we use the original placeholder op's name here
    TF_RETURN_IF_ERROR(base_make_node("DestroyTemporaryVariable", n->name())
                           .Attr("T", dtype)
                           .Attr("var_name", accumulator_name)
                           .Input(initialize_accumulator)
                           .Finalize(g, &clean_up_accumulator));

    // Add edges to the graph to ensure that operations occur in the right
    // order:
    // 1. Do anything that had a control edge to the AccumulateNV2 placeholder
    // 2. Initialize accumulator
    // 3. Add input values to accumulator (already handled by data edges
    //    added above)
    // 4. Reclaim the buffer that held the accumulator
    // 5. Do anything that depended on the AccumulateNV2 placeholder
    for (const Edge* control_edge : control_edges) {
      g->AddControlEdge(control_edge->src(), initialize_accumulator);
    }

    for (Node* assign_add : add_values_to_accumulator) {
      g->AddControlEdge(assign_add, clean_up_accumulator);
    }

    for (const Edge* out_edge : n->out_edges()) {
      if (out_edge->IsControlEdge()) {
        g->AddControlEdge(clean_up_accumulator, out_edge->dst());
      } else {
        g->AddEdge(clean_up_accumulator, 0, out_edge->dst(),
                   out_edge->dst_input());
      }
    }

    // Remove the original AccumulateNV2 placeholder op.
    // This removal modifies the op and must happen after we have finished
    // using its incoming/outgoing edge sets.
    g->RemoveNode(n);

    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      AccumulateNV2RemovePass);

}  // namespace
}  // namespace tensorflow
