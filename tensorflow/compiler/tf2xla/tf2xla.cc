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

#include "tensorflow/compiler/tf2xla/tf2xla.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

const char* const kArgOp = "_Arg";
const char* const kRetvalOp = "_Retval";
const char* const kFeedIdAttr = "_feed_id";
const char* const kFetchIdAttr = "_fetch_id";
const char* const kShapeAttr = "_shape";
const char* const kDebugNameAttr = "_debug_name";

namespace {

typedef std::unordered_map<string, Node*> NodeMap;

// Each feed id identifies the positional output of some node, which may consist
// of multiple edges. AddPlaceholdersForFeeds has already replaced each fed
// tensor with a placeholder.  For each feed tensor, replaces all edges so they
// point from a new _Arg node instead.
Status AddArgNodes(Graph* graph, const NodeMap& node_map,
                   const protobuf::RepeatedPtrField<tf2xla::Feed>& feeds,
                   const std::unordered_map<string, string>& feed_remapping) {
  for (int arg_index = 0; arg_index < feeds.size(); ++arg_index) {
    const tf2xla::Feed& feed = feeds[arg_index];
    // All feeds have been replaced by placeholders.
    const int output_index = 0;

    const string key = TensorIdToString(feed.id());
    const auto remap_it = feed_remapping.find(key);
    auto node_it = node_map.find(remap_it->second);
    if (node_it == node_map.end()) {
      // Strip off the aot_feed_#/ prefix.
      StringPiece name(remap_it->second);
      const auto index = name.find('/');
      if (index > 0) name.remove_prefix(index + 1);
      return errors::InvalidArgument(
          "Node is fed but not needed for fetching: ", name);
    }
    const Node* feed_node = node_it->second;

    // TODO(toddw): Invoke shape inference in AddPlaceholdersForFeeds and add a
    // "_shape" attr if we can determine it.  That way the graph will be
    // initialized with whatever shapes we can infer, while the user can still
    // explicitly specify or override them.
    Node* arg_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_arg_", arg_index), kArgOp)
            .Attr("T", BaseType(feed_node->output_type(output_index)))
            .Attr("index", arg_index)
            .Attr(kFeedIdAttr, TensorIdToString(feed.id()))
            .Attr(kShapeAttr, TensorShape(feed.shape()))
            .Attr(kDebugNameAttr, feed.name())
            .Finalize(graph, &arg_node));

    // Collects out-edges from the feed node that have a matching edge index;
    // these will be replaced with edges from the arg node instead.
    //
    // We must collect the edges first and process them in a second pass, since
    // removing the edge from the graph invalidates feed_node->out_edges.
    std::vector<const Edge*> feed_edges;
    for (const Edge* edge : feed_node->out_edges()) {
      if (edge->src_output() == output_index) {
        feed_edges.push_back(edge);
      }
    }
    for (const Edge* edge : feed_edges) {
      graph->AddEdge(arg_node, 0, edge->dst(), edge->dst_input());
      graph->RemoveEdge(edge);
    }
  }
  return Status::OK();
}

// Each fetch id identifies the positional output of some node.  For each fetch
// node, adds a new _Retval node instead, and adds the node to `retval_nodes`.
Status AddRetvalNodes(Graph* graph, const NodeMap& node_map,
                      const protobuf::RepeatedPtrField<tf2xla::Fetch>& fetches,
                      std::unordered_set<const Node*>* retval_nodes) {
  for (int ret_index = 0; ret_index < fetches.size(); ++ret_index) {
    const tf2xla::TensorId& id = fetches[ret_index].id();
    auto it = node_map.find(id.node_name());
    if (it == node_map.end()) {
      return errors::NotFound("Can't find fetch id: ", TensorIdToString(id));
    }
    Node* fetch_node = it->second;
    if (id.output_index() >= fetch_node->num_outputs()) {
      return errors::InvalidArgument("Invalid fetch id: ", TensorIdToString(id),
                                     ", output index should be < ",
                                     fetch_node->num_outputs());
    }
    // Connects fetch_node -> retval_node.
    Node* retval_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_retval_", ret_index), kRetvalOp)
            .Input(fetch_node, id.output_index())
            .Attr("T", BaseType(fetch_node->output_type(id.output_index())))
            .Attr("index", ret_index)
            .Attr(kFetchIdAttr, TensorIdToString(id))
            .Finalize(graph, &retval_node));
    retval_nodes->insert(retval_node);
  }
  return Status::OK();
}

// RewriteAndPruneGraph identifies input and output edges (named by the feed and
// fetch ids respectively), and rewrites the edges so that inputs flow from _Arg
// nodes, and outputs flow to _Retval nodes.  This allows the symbolic graph
// execution to know the input and output args for the generated function.
Status RewriteAndPruneGraph(
    Graph* graph, const tf2xla::Config& config,
    const std::unordered_map<string, string>& feed_remapping) {
  NodeMap node_map;
  for (Node* n : graph->nodes()) {
    node_map[n->name()] = n;
  }
  TF_RETURN_IF_ERROR(
      AddArgNodes(graph, node_map, config.feed(), feed_remapping));
  std::unordered_set<const Node*> retval_nodes;
  TF_RETURN_IF_ERROR(
      AddRetvalNodes(graph, node_map, config.fetch(), &retval_nodes));
  VLOG(2) << "Post rewrite: "
          << dump_graph::DumpGraphToFile("tf2xla_post_rewrite", *graph);
  PruneForReverseReachability(graph, retval_nodes);
  FixupSourceAndSinkEdges(graph);
  VLOG(2) << "Post prune: "
          << dump_graph::DumpGraphToFile("tfcompile_post_prune", *graph);
  // Sanity-check, to make sure the feeds and fetches still exist post-pruning.
  std::set<string> missing_feeds, missing_fetches;
  for (const tf2xla::Feed& feed : config.feed()) {
    missing_feeds.insert(TensorIdToString(feed.id()));
  }
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    missing_fetches.insert(TensorIdToString(fetch.id()));
  }
  for (const Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      string feed_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kFeedIdAttr, &feed_id));
      if (missing_feeds.erase(feed_id) == 0) {
        return errors::Aborted(kArgOp,
                               " node found with unknown feed id: ", feed_id);
      }
    } else if (n->type_string() == kRetvalOp) {
      string fetch_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kFetchIdAttr, &fetch_id));
      if (missing_fetches.erase(fetch_id) == 0) {
        return errors::Aborted(kRetvalOp,
                               " node found with unknown fetch id: ", fetch_id);
      }
    }
  }
  if (!missing_feeds.empty() || !missing_fetches.empty()) {
    return errors::Aborted(
        "Post graph-pruning",
        ", missing feeds: ", str_util::Join(missing_feeds, ", "),
        ", missing fetches: ", str_util::Join(missing_fetches, ", "));
  }
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

// Fills in xla_args from the corresponding _Arg nodes in the graph.
Status CreateXlaArgs(const Graph& graph,
                     std::vector<XlaCompiler::Argument>* xla_args) {
  std::vector<Node*> arg_nodes;
  TF_RETURN_IF_ERROR(CollectArgNodes(graph, &arg_nodes));
  for (const Node* node : arg_nodes) {
    XlaCompiler::Argument arg;
    arg.kind = XlaCompiler::Argument::kParameter;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &arg.type));
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kShapeAttr, &arg.shape));
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kDebugNameAttr, &arg.name));
    xla_args->push_back(arg);
  }
  return Status::OK();
}

// Converts the TensorFlow graph into an XLA computation, by executing the
// graph symbolically, with each op building up the XLA HLO.
Status ConvertGraphToXla(std::unique_ptr<Graph> graph, xla::Client* client,
                         xla::Computation* computation) {
  XlaOpRegistry::RegisterCompilationKernels();
  for (Node* node : graph->nodes()) {
    node->set_assigned_device_name(
        strings::StrCat("/device:", DEVICE_CPU_XLA_JIT));
  }
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(CreateXlaArgs(*graph, &xla_args));

  // Compile the graph into an XLA computation.
  XlaCompiler::Options compiler_options;
  compiler_options.client = client;
  DeviceType device_type(DEVICE_CPU_XLA_JIT);
  compiler_options.device_type = &device_type;
  compiler_options.flib_def = &graph->flib_def();
  compiler_options.graph_def_version = graph->versions().producer();
  compiler_options.allow_cpu_custom_calls = true;
  XlaCompiler compiler(compiler_options);

  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                           "tfcompile", std::move(graph),
                                           xla_args, &result));
  *computation = std::move(*result.computation);

  int num_const_results = 0;
  for (int i = 0; i < result.outputs.size(); ++i) {
    // Ending up with const results (i.e. output args) is an error, since it
    // means that one or more fetches that the user specified will be dropped
    // from the generated function.  It's most likely a configuration error,
    // since the user shouldn't be asking for output args that end up as consts.
    //
    // TODO(toddw): Provide a way for the user to access const output args,
    // e.g. perhaps hard-coded into the header, or somehow copied into the
    // output buffers.
    if (result.outputs[i].is_constant) {
      ++num_const_results;
      LOG(ERROR) << "ConstRetVal index:" << i
                 << " value:" << result.outputs[i].constant_value.DebugString();
    }
  }
  if (num_const_results > 0) {
    return errors::Unimplemented(
        "Conversion from TensorFlow graph to XLA resulted in ",
        num_const_results,
        " constant results.  The configuration of "
        "the output args (i.e. fetch ids) is probably wrong.");
  }
  return Status::OK();
}

// InitGraph creates a graph based on the graph_def, that may then be converted
// to an xla::Computation via ConvertGraphToXla.
//
// The graph is rewritten with _Arg and _Retval nodes, representing the inputs
// and outputs of the function that will be compiled.  Each feed id causes a new
// _Arg node to be created, where we first collect all existing edges pointing
// from the named node's output index, and then rewrite them to point from that
// _Arg node instead.  Each fetch id causes a new _Retval node to be created,
// with a new edge pointing from the named node's output index to that _Retval
// node.
Status InitGraph(const GraphDef& graph_def, const tf2xla::Config& config,
                 std::unique_ptr<Graph>* graph) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graph_def.library());
  std::unique_ptr<Graph> g(new Graph(flib_def));

  // Replace references to fed tensors with references to newly added
  // placeholders.
  GraphDef first_copy_def = graph_def;

  // Maps from name:port of a feed to the name:port of the placeholder to use.
  std::unordered_map<string, string> feed_remapping;
  TF_RETURN_IF_ERROR(AddPlaceholdersForFeeds(config, g->op_registry(),
                                             &feed_remapping, &first_copy_def));

  // Prune the GraphDef first so that unknown ops that we aren't compiling get
  // filtered out.
  GraphDef second_copy_def;
  TF_RETURN_IF_ERROR(
      PruneGraphDefInto(config, first_copy_def, &second_copy_def));

  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(
      &second_copy_def, *g->op_registry(), /*node_offset=*/0));

  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(GraphConstructorOptions(),
                                            second_copy_def, g.get()));
  TF_RETURN_IF_ERROR(RewriteAndPruneGraph(g.get(), config, feed_remapping));
  *graph = std::move(g);
  return Status::OK();
}

}  // namespace

Status ConvertGraphDefToXla(const GraphDef& graph_def,
                            const tf2xla::Config& config, xla::Client* client,
                            xla::Computation* computation) {
  std::unique_ptr<Graph> graph;
  TF_RETURN_IF_ERROR(InitGraph(graph_def, config, &graph));
  TF_RETURN_IF_ERROR(ConvertGraphToXla(std::move(graph), client, computation));
  return Status::OK();
}

}  // namespace tensorflow
