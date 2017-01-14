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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {
namespace {

// Replaces occurrences of parallel_concat with the implementation based on
// unsafe ops. Sets removed_any to true if any parallel_concats were removed;
// leaves it untouched otherwise.
// TODO(apassos) Use NodeBuilder.
Status RemoveParallelConcat(bool* removed_any, Graph* g) {
  gtl::InlinedVector<Node*, 2> matches;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "ParallelConcat") {
      matches.push_back(n);
    }
  }
  for (Node* n : matches) {
    AttrSlice n_attrs(n->def());
    auto make_node = [n, g, &n_attrs](string op) {
      NodeDef node;
      node.set_op(op);
      node.set_name(g->NewName(n->name()));
      node.set_device(n->def().device());
      string colo;
      if (GetNodeAttr(n_attrs, "_class", &colo).ok()) {
        AddNodeAttr("_class", colo, &node);
      }
      return node;
    };
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    TensorShapeProto shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));
    // Add the constant shape input to the start node.
    NodeDef shape_node_def = make_node("Const");
    AddNodeAttr("dtype", DT_INT32, &shape_node_def);
    TensorProto shape_tensor;
    shape_tensor.set_dtype(DT_INT32);
    shape_tensor.mutable_tensor_shape()->add_dim()->set_size(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i) {
      shape_tensor.add_int_val(shape.dim(i).size());
    }
    AddNodeAttr("value", shape_tensor, &shape_node_def);
    Status status = Status::OK();
    Node* shape_node = g->AddNode(shape_node_def, &status);
    if (!status.ok()) return status;

    // Add the start node
    NodeDef start_def = make_node("_ParallelConcatStart");
    AddNodeAttr("dtype", dtype, &start_def);
    AddNodeAttr("Tshape", DT_INT32, &start_def);
    AddNodeAttr("init", false, &start_def);
    start_def.add_input(shape_node_def.name());
    Node* start = g->AddNode(start_def, &status);
    if (!status.ok()) return status;
    // TODO(apassos): make the shape an attr of _ParallelStackBegin.
    g->AddEdge(shape_node, 0, start, 0);

    // Add all the inplace_updates.
    std::vector<string> control_dependencies;
    std::vector<Node*> control_nodes;
    int i = 0;
    for (const Edge* input_edge : n->in_edges()) {
      if (input_edge->IsControlEdge()) {
        g->AddControlEdge(input_edge->src(), start);
        continue;
      }
      // Constant index for the update node.
      // TODO(apassos): make _ParallelStackUpdate take this as an attr.
      NodeDef update_idx_def = make_node("Const");
      AddNodeAttr("dtype", DT_INT64, &update_idx_def);
      TensorProto index_tensor;
      index_tensor.set_dtype(DT_INT64);
      index_tensor.mutable_tensor_shape()->add_dim()->set_size(1);
      index_tensor.add_int64_val(i);
      AddNodeAttr("value", index_tensor, &update_idx_def);
      Node* index = g->AddNode(update_idx_def, &status);
      if (!status.ok()) return status;

      NodeDef update_def = make_node("_ParallelConcatUpdate");
      control_dependencies.push_back(update_def.name());
      AddNodeAttr("T", dtype, &update_def);
      AddNodeAttr("Tshape", DT_INT64, &update_def);
      update_def.add_input(start_def.name());
      update_def.add_input(update_idx_def.name());
      update_def.add_input(strings::StrCat(input_edge->src()->name(), ":",
                                           input_edge->src_output()));
      Node* update = g->AddNode(update_def, &status);
      if (!status.ok()) return status;
      g->AddEdge(start, 0, update, 0);
      g->AddEdge(index, 0, update, 1);
      g->AddEdge(input_edge->src(), input_edge->src_output(), update, 2);
      control_nodes.push_back(update);

      ++i;
    }

    // Add the final identity.
    NodeDef identity_def = make_node("Identity");
    AddNodeAttr("T", dtype, &identity_def);
    identity_def.add_input(start_def.name());
    for (const string& s : control_dependencies) {
      identity_def.add_input(strings::StrCat("^", s));
    }
    Node* identity_node = g->AddNode(identity_def, &status);
    if (!status.ok()) return status;
    g->AddEdge(start, 0, identity_node, 0);
    for (Node* inp : control_nodes) {
      g->AddControlEdge(inp, identity_node);
    }

    // Remove the node and redirect edges.
    for (auto* e : n->out_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(identity_node, e->dst());
      } else {
        g->AddEdge(identity_node, 0, e->dst(), e->dst_input());
      }
    }
    g->RemoveNode(n);
    *removed_any = true;
  }
  return Status::OK();
}
}

GraphOptimizer::GraphOptimizer(const OptimizerOptions& opts) : opts_(opts) {
  if (opts_.opt_level() >= OptimizerOptions::L1) {
    opts_.set_do_common_subexpression_elimination(true);
    opts_.set_do_constant_folding(true);
  }
}

GraphOptimizer::~GraphOptimizer() {}

void GraphOptimizer::Optimize(FunctionLibraryRuntime* runtime, Env* env,
                              Device* device, std::unique_ptr<Graph>* graph) {
  Graph* g = graph->get();
  DumpGraph("Initial", g);

  bool changed = true;
  const int kMaxRounds = 10;
  for (int rounds = 0; rounds < kMaxRounds; ++rounds) {
    changed = false;
    if (RemoveListArrayConverter(g)) {
      DumpGraph("RemoveListArrayConverter", g);
      changed = true;
    }
    auto s = RemoveParallelConcat(&changed, g);
    if (!s.ok()) {
      // TODO(apassos): figure out how to halt here.
      LOG(WARNING) << s;
    }
    if (opts_.do_function_inlining() && RemoveDeadNodes(g)) {
      DumpGraph("RemoveDeadNodes", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveIdentityNodes(g)) {
      DumpGraph("RemoveIdentityNodes", g);
      changed = true;
    }

    if (opts_.do_constant_folding()) {
      ConstantFoldingOptions cf_opts;
      if (DoConstantFolding(cf_opts, runtime, env, device, g)) {
        RemoveDeadNodes(g);
        DumpGraph("ConstFolding", g);
        changed = true;
      }
    }

    if (opts_.do_function_inlining() && FixupSourceAndSinkEdges(g)) {
      DumpGraph("FixupSourceAndSinkEdges", g);
      changed = true;
    }
    if (opts_.do_common_subexpression_elimination() &&
        OptimizeCSE(g, nullptr)) {
      DumpGraph("OptimizeCSE", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && ExpandInlineFunctions(runtime, g)) {
      DumpGraph("ExpandInlineFunctions", g);
      changed = true;
    }
    if (!changed) break;
  }

  std::unique_ptr<Graph> copy(new Graph(g->op_registry()));
  CopyGraph(*g, copy.get());
  graph->swap(copy);

  DumpGraph("ReCopy", graph->get());
}

}  // end namespace tensorflow
