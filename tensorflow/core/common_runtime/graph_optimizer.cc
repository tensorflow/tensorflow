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
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {
namespace {

// Replaces occurrences of parallel_concat with the implementation based on
// unsafe ops. Sets removed_any to true if any parallel_concats were removed;
// leaves it untouched otherwise.
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
      NodeBuilder node_builder(g->NewName(n->name()), op);
      node_builder.Device(n->def().device());
      string colo;
      if (GetNodeAttr(n_attrs, "_class", &colo).ok()) {
        node_builder.Attr("_class", colo);
      }
      return node_builder;
    };
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    TensorShapeProto shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));

    // Add the start node
    Node* start;
    TF_RETURN_IF_ERROR(make_node("_ParallelConcatStart")
                           .Attr("shape", shape)
                           .Attr("dtype", dtype)
                           .Finalize(g, &start));

    // Add all the inplace_updates.
    std::vector<Node*> control_nodes;
    int64 i = 0;
    for (const Edge* input_edge : n->in_edges()) {
      if (input_edge->IsControlEdge()) {
        g->AddControlEdge(input_edge->src(), start);
        continue;
      }

      Node* update;
      TF_RETURN_IF_ERROR(make_node("_ParallelConcatUpdate")
                             .Attr("loc", i)
                             .Input(start)
                             .Input(input_edge->src(), input_edge->src_output())
                             .Finalize(g, &update));
      control_nodes.push_back(update);

      ++i;
    }

    // Add the final identity.
    NodeBuilder identity_def = make_node("Identity");
    identity_def.Input(start, 0);
    for (Node* s : control_nodes) {
      identity_def.ControlInput(s);
    }
    Node* identity_node;
    TF_RETURN_IF_ERROR(identity_def.Finalize(g, &identity_node));

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
