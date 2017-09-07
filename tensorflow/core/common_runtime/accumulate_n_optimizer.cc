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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {
namespace {

const int kNotAnIndex = -1;

// Utility function to retrieve the index of a given named input of an op
int op_input_index(const char* op_name, const char* input_name) {
  const OpDef* op_def;
  if (!OpRegistry::Global()->LookUpOpDef(op_name, &op_def).ok()) {
    return kNotAnIndex;
  }

  // n.b. OpDef is defined in op_def.proto
  for (int i = 0; i < op_def->input_arg_size(); i++) {
    const OpDef::ArgDef& arg = op_def->input_arg(i);
    if (arg.name() == input_name) {
      return i;
    }
  }

  // Name not found
  return kNotAnIndex;
}

Tensor make_zeros(const DataType& dtype, const TensorShapeProto& shape) {
  Tensor tensor(dtype, TensorShape(shape));

  // Conveniently, all numeric data types have 0x0 == zero.  Otherwise we would
  // need a giant switch statement here.
  memset(const_cast<char*>(tensor.tensor_data().data()), 0,
         tensor.tensor_data().size());

  return tensor;
}

// Replaces occurrences of the AccumulateN stub operator with a graph of
// lower-level ops.  The graph is equivalent to what the following Python
// code used to produce:
//  with ops.name_scope(name, "AccumulateN", inputs) as name:
//    var = gen_state_ops._temporary_variable(
//        shape=tensor_shape.vector(0), dtype=tensor_dtype)
//    with ops.colocate_with(var):
//      zeros = array_ops.zeros_like(gen_control_flow_ops._merge(inputs)[0])
//      zeros.set_shape(shape)
//      ref = state_ops.assign(var, zeros, validate_shape=False)
//      update_ops = [
//          state_ops.assign_add(ref, input_tensor, use_locking=True)
//          for input_tensor in inputs
//      ]
//      with ops.control_dependencies(update_ops):
//        return gen_state_ops._destroy_temporary_variable(
//            ref, var_name=var.op.name, name=name)
class AccumulateNRemovePass : public GraphOptimizationPass {
 public:
  // Indices of named arguments to operators we generate.
  // Constants initialized in the class's constructor so that init code
  // runs after the OpRegistry is initialized.
  int kAssignAddRefIx, kAssignAddValueIx, kAssignRefIx, kAssignValueIx;

  AccumulateNRemovePass() :
    kAssignAddRefIx(op_input_index("AssignAdd", "ref")), 
    kAssignAddValueIx(op_input_index("AssignAdd", "value")), 
    kAssignRefIx(op_input_index("Assign", "ref")), 
    kAssignValueIx(op_input_index("Assign", "value"))
  { }

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
          "AccumulateN removal should happen before partitioning and a "
          "graph should be available.");
    }

    if (kNotAnIndex == kAssignAddRefIx) {
      // i.e. we couldn't figure out what input of the Assign op is called
      // "ref"
      return Status(error::INTERNAL, "Failed to query operator registry");
    }

    // Build up a todo list of ops to replace, *then* modify the graph
    gtl::InlinedVector<Node*, 2> matches;
    for (Node* n : g->op_nodes()) {
      if (n->type_string() == "AccumulateN") {
        matches.push_back(n);
      }
    }
    for (Node* n : matches) {
      TF_RETURN_IF_ERROR(rewriteNode(n, g));
    }
    return Status::OK();
  }

  Status rewriteNode(Node *n, Graph* g) {
    AttrSlice n_attrs = n->attrs();
    auto base_make_node = [n, g, &n_attrs](const string& op,
                                           const string& name) {
      NodeBuilder node_builder(name, op);

      // The pieces of AccumulateN should all be on the same node.
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

    // Create graph nodes.
    Node* tmp_var = nullptr;
    Node* initial_val = nullptr;
    Node* assign = nullptr;
    std::vector<Node*> add_nodes(data_edges.size(), nullptr);
    Node* destroy_var = nullptr;
    {
      const string accumulator_name =
          strings::StrCat(n->name(), "/Internal/Accumulator");
      TF_RETURN_IF_ERROR(make_node("TemporaryVariable")
                             .Attr("shape", shape)
                             .Attr("dtype", dtype)
                             .Attr("var_name", accumulator_name)
                             .Finalize(g, &tmp_var));
      TF_RETURN_IF_ERROR(make_node("Const")
                             .Attr("value", make_zeros(dtype, shape))
                             .Attr("dtype", dtype)
                             .Finalize(g, &initial_val));
      TF_RETURN_IF_ERROR(
          make_node("Assign").Attr("T", dtype).Finalize(g, &assign));
      for (int i = 0; i < data_edges.size(); ++i) {
        Node* assignAdd;
        TF_RETURN_IF_ERROR(make_node("AssignAdd")
                               .Attr("T", dtype)
                               .Attr("use_locking", true)
                               .Finalize(g, &assignAdd));

        add_nodes.push_back(assignAdd);
      }
      TF_RETURN_IF_ERROR(make_node("DestroyTemporaryVariable")
                             .Attr("T", dtype)
                             .Attr("var_name", accumulator_name)
                             .Finalize(g, &destroy_var));
    }

    // Add edges for data dependencies.
    {
      for (int i = 0; i < data_edges.size(); i++) {
        // Input[i] --> AssignAdd[i].value
        g->AddEdge(data_edges[i]->src(), data_edges[i]->src_output(),
                   add_nodes[i], kAssignAddValueIx);

        // Assign --> AssignAdd[1...n]
        g->AddEdge(assign, 0, add_nodes[i], kAssignAddRefIx);
      }

      // TempVar --> Assign
      g->AddEdge(tmp_var, 0, assign, kAssignValueIx);

      // Const --> Assign
      g->AddEdge(initial_val, 0, assign, kAssignRefIx);

      // Assign --> Destroy
      g->AddEdge(assign, 0, destroy_var, 0);

      // Redirect original outgoing data edges
      for (const Edge* out_edge : n->out_edges()) {
        if (!out_edge->IsControlEdge()) {
          g->AddEdge(destroy_var, 0, out_edge->dst(), out_edge->dst_input());
        }
      }
    }

    // Add edges for control dependencies.
    {
      for (const Edge* data_edge : data_edges) {
        // Inputs --> Assign
        g->AddControlEdge(data_edge->src(), assign);
      }
      for (const Edge* control_edge : control_edges) {
        // Original incoming control edges --> Assign
        g->AddControlEdge(control_edge->src(), assign);
      }
      for (Node* assign_add : add_nodes) {
        // AssignAdds --> Destroy
        g->AddControlEdge(assign_add, destroy_var);
      }
      // Redirect original outgoing control edges
      for (const Edge* out_edge : n->out_edges()) {
        if (out_edge->IsControlEdge()) {
          g->AddControlEdge(destroy_var, out_edge->dst());
        }
      }
    }

    // Remove the original placeholder.
    g->RemoveNode(n);

    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      AccumulateNRemovePass);

}  // namespace
}  // namespace tensorflow
