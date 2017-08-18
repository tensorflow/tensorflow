/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/xlagen/xlagen.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace xlagen {

const char* const kArgOp = "_Arg";
const char* const kRetvalOp = "_Retval";
const char* const kPlaceholderOp = "Placeholder";
const char* const kFeedIdAttr = "_feed_id";
const char* const kFetchIdAttr = "_fetch_id";
const char* const kShapeAttr = "_shape";
const char* const kDebugNameAttr = "_debug_name";

namespace {

Status GetTensorShape(shape_inference::InferenceContext *ctx,
                      int output_idx, TensorShape *shape) {
  auto sh = ctx->output(output_idx);
  if (!ctx->FullyDefined(sh)) {
    return errors::InvalidArgument("Shapes should be fully defined");
  }

  *shape = TensorShape();
  for (int64 i = 0; i < ctx->Rank(sh); ++i) {
    shape->AddDim(ctx->Value(ctx->Dim(sh, i)));
  }
  return Status::OK();
}

Status AddArgAndRetvalNodes(const std::vector<string> &output_tensor_names,
                            Graph *graph) {
  ShapeRefiner refiner(graph->versions().producer(), graph->op_registry());
  for (const Node *n : graph->nodes()) {
    TF_RETURN_IF_ERROR(refiner.AddNode(n));
  }
  std::unordered_set<const Node*> retval_nodes;

  int arg_index = 0;
  int ret_index = 0;
  for (Node *n : graph->nodes()) {
    auto ctx = refiner.GetContext(n);
    TensorShape shape;

    if (n->type_string() == kPlaceholderOp) {
      if (std::find(output_tensor_names.begin(), output_tensor_names.end(),
                    n->name() + ":0") != output_tensor_names.end()) {
        return errors::InvalidArgument("Placeholder cannot be fetched");
      }
      TF_RETURN_IF_ERROR(GetTensorShape(ctx, 0, &shape));
      Node *arg_node = nullptr;
      TF_RETURN_IF_ERROR(
          NodeBuilder(strings::StrCat("_arg_", arg_index), kArgOp)
              .Attr("T", BaseType(n->output_type(0)))
              .Attr("index", arg_index)
              .Attr(kShapeAttr, shape)
              .Attr(kDebugNameAttr, n->name())
              .Finalize(graph, &arg_node));
      ++arg_index;

      std::vector<const Edge*> to_delete;
      for (const Edge* edge : n->out_edges()) {
         to_delete.push_back(edge);
      }
      for (const Edge *edge : to_delete) {
        graph->AddEdge(arg_node, 0, edge->dst(), edge->dst_input());
        graph->RemoveEdge(edge);
      }
    } else {
      for (int i = 0; i < n->num_outputs(); ++i) {
        auto tensor_name = n->name() + ":" + std::to_string(i);
        if (std::find(output_tensor_names.begin(), output_tensor_names.end(),
                      tensor_name) == output_tensor_names.end()) {
          continue;
        }
        TF_RETURN_IF_ERROR(GetTensorShape(ctx, i, &shape));
        Node* retval_node = nullptr;
        TF_RETURN_IF_ERROR(
            NodeBuilder(strings::StrCat("_retval_", ret_index), kRetvalOp)
                .Input(n, i)
                .Attr("T", BaseType(n->output_type(i)))
                .Attr("index", ret_index)
                .Attr(kShapeAttr, shape)
                .Attr(kDebugNameAttr, tensor_name)
                .Finalize(graph, &retval_node));
        ++ret_index;
        retval_nodes.insert(retval_node);
      }
    }
  }
  PruneForReverseReachability(graph, retval_nodes);
  return Status::OK();
}

// CollectArgNodes collects _Arg nodes from the graph, and performs basic
// sanity-checking to ensure the index and type attributes of each node are
// initialized correctly.
Status CollectArgNodes(const Graph& graph, std::vector<Node*>* arg_nodes) {
  std::map<int, Node*> indexed_arg_nodes;
  for (Node* n : graph.nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      auto insert_result = indexed_arg_nodes.insert({index, n});
      if (!insert_result.second) {
        const Node* dup = insert_result.first->second;
        return errors::InvalidArgument(
            "Multiple ", kArgOp, " nodes with index ", index, ", ",
            n->DebugString(), " and ", dup->DebugString());
      }
    }
  }
  arg_nodes->clear();
  for (const auto& index_node : indexed_arg_nodes) {
    if (index_node.first != arg_nodes->size()) {
      return errors::InvalidArgument("Expected ", kArgOp, " node with index ",
                                     arg_nodes->size(), ", but got index ",
                                     index_node.first);
    }
    arg_nodes->push_back(index_node.second);
  }
  return Status::OK();
}

Status CreateXlaArgs(const Graph* graph,
                     std::vector<XlaCompiler::Argument>* xla_args) {
  std::vector<Node*> arg_nodes;
  TF_RETURN_IF_ERROR(CollectArgNodes(*graph, &arg_nodes));
  for (const Node* node : arg_nodes) {
    XlaCompiler::Argument arg;
    arg.kind = XlaCompiler::Argument::kParameter;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &arg.type));
    TensorShape shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kShapeAttr, &shape));
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(arg.type, shape, &arg.shape));
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kDebugNameAttr, &arg.name));
    xla_args->push_back(arg);
  }
  return Status::OK();
}

}  // end namespace

// Convert a tf graph to a xla session module
xla::StatusOr<std::unique_ptr<xla::SessionModule>>
GraphDefToXlaSessionModule(const std::vector<string> &output_tensor_names,
                           const GraphDef &graphdef) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graphdef.library());
  std::unique_ptr<Graph> g(new Graph(flib_def));

  ShapeRefiner refiner(graphdef.versions().producer(), g->op_registry());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graphdef, g.get()));

  TF_RETURN_IF_ERROR(AddArgAndRetvalNodes(output_tensor_names, g.get()));

  FixupSourceAndSinkEdges(g.get());

  for (Node* node : g->nodes()) {
    node->set_assigned_device_name(DEVICE_CPU_XLA_JIT);
  }
  std::vector<XlaCompiler::Argument> args;
  TF_RETURN_IF_ERROR(CreateXlaArgs(g.get(), &args));

  xla::Client *client = xla::ClientLibrary::LocalClientOrDie();
  XlaOpRegistry::RegisterCompilationKernels();

  // Compile the graph into an XLA computation.
  XlaCompiler::Options compiler_options;
  compiler_options.client = client;
  DeviceType device_type(DEVICE_CPU_XLA_JIT);
  compiler_options.device_type = &device_type;
  compiler_options.flib_def = &g->flib_def();
  compiler_options.graph_def_version = g->versions().producer();
  XlaCompiler compiler(compiler_options);

  XlaCompiler::CompilationResult result;
  TF_CHECK_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "xla_graph",
                                    std::move(g), args, &result));

  if (!result.computation) {
    return errors::InvalidArgument("Empty computation");
  }
  return result.computation->Snapshot();
}

} // end namespace xlagen
} // end namespace tensorflow
