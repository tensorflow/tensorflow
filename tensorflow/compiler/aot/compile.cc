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

#include "tensorflow/compiler/aot/compile.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/aot/tfcompile_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace tfcompile {

const char* const kArgOp = "_Arg";
const char* const kRetvalOp = "_Retval";
const char* const kFeedIdAttr = "_feed_id";
const char* const kFetchIdAttr = "_fetch_id";
const char* const kShapeAttr = "_shape";
const char* const kDebugNameAttr = "_debug_name";

namespace {

Status DumpGraph(const MainFlags& flags, const string& name,
                 const Graph& graph) {
  if (flags.debug_dir.empty()) {
    return Status::OK();
  }
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  string file = io::JoinPath(flags.debug_dir, name + ".pbtxt");
  return WriteTextProto(Env::Default(), file, graph_def);
}

string TensorIdToString(const TensorId& id) {
  return strings::StrCat(id.node_name(), ":", id.output_index());
}

typedef std::unordered_map<string, Node*> NodeMap;

// Each feed id identifies the positional output of some node, which may consist
// of multiple edges.  For each feed node, replaces all matching edges so that
// they point from a new _Arg node instead.
Status AddArgNodes(Graph* graph, const NodeMap& node_map,
                   const protobuf::RepeatedPtrField<Feed>& feeds) {
  for (int arg_index = 0; arg_index < feeds.size(); ++arg_index) {
    const Feed& feed = feeds[arg_index];
    const TensorId& id = feed.id();
    auto it = node_map.find(id.node_name());
    if (it == node_map.end()) {
      return errors::NotFound("Can't find feed id: ", TensorIdToString(id));
    }
    const Node* feed_node = it->second;
    if (id.output_index() >= feed_node->num_outputs()) {
      return errors::InvalidArgument("Invalid feed id: ", TensorIdToString(id),
                                     ", output index should be < ",
                                     feed_node->num_outputs());
    }
    // TODO(toddw): Invoke shape inference on the graph and add a "_shape" attr
    // if we can determine it.  That way the graph will be initialized with
    // whatever shapes we can infer, while the user can still explicitly specify
    // or override them.
    Node* arg_node = nullptr;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_arg_", arg_index), kArgOp)
            .Attr("T", BaseType(feed_node->output_type(id.output_index())))
            .Attr("index", arg_index)
            .Attr(kFeedIdAttr, TensorIdToString(id))
            .Attr(kShapeAttr, TensorShape(feed.shape()))
            .Attr(kDebugNameAttr, feed.name())
            .Finalize(graph, &arg_node));
    // Collects out-edges from the feed node that have a matching edge index;
    // these will be replaced with edges from the arg node instead.  Also
    // replaces all control edges from Placeholder feed nodes; similar code
    // exists in subgraph::RewriteGraphForExecution.
    // TODO(toddw): Why only replace control edges from Placeholder?
    //
    // We must collect the edges first and process them in a second pass, since
    // removing the edge from the graph invalidates feed_node->out_edges.
    std::vector<const Edge*> feed_edges;
    for (const Edge* edge : feed_node->out_edges()) {
      if (edge->src_output() == id.output_index() ||
          (edge->src_output() == Graph::kControlSlot &&
           feed_node->type_string() == "Placeholder")) {
        feed_edges.push_back(edge);
      }
    }
    for (const Edge* edge : feed_edges) {
      if (edge->src_output() == id.output_index()) {
        graph->AddEdge(arg_node, 0, edge->dst(), edge->dst_input());
      } else {
        CHECK_EQ(edge->src_output(), Graph::kControlSlot);
        graph->AddControlEdge(arg_node, edge->dst());
      }
      graph->RemoveEdge(edge);
    }
  }
  return Status::OK();
}

// Each fetch id identifies the positional output of some node.  For each fetch
// node, adds a new _Retval node instead, and adds the node to `retval_nodes`.
Status AddRetvalNodes(Graph* graph, const NodeMap& node_map,
                      const protobuf::RepeatedPtrField<Fetch>& fetches,
                      std::unordered_set<const Node*>* retval_nodes) {
  for (int ret_index = 0; ret_index < fetches.size(); ++ret_index) {
    const TensorId& id = fetches[ret_index].id();
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
Status RewriteAndPruneGraph(Graph* graph, const Config& config,
                            const MainFlags& flags) {
  NodeMap node_map;
  for (Node* n : graph->nodes()) {
    node_map[n->name()] = n;
  }
  TF_RETURN_IF_ERROR(AddArgNodes(graph, node_map, config.feed()));
  std::unordered_set<const Node*> retval_nodes;
  TF_RETURN_IF_ERROR(
      AddRetvalNodes(graph, node_map, config.fetch(), &retval_nodes));
  TF_RETURN_IF_ERROR(DumpGraph(flags, "tfcompile_post_rewrite", *graph));
  PruneForReverseReachability(graph, retval_nodes);
  FixupSourceAndSinkEdges(graph);
  TF_RETURN_IF_ERROR(DumpGraph(flags, "tfcompile_post_prune", *graph));
  // Sanity-check, to make sure the feeds and fetches still exist post-pruning.
  std::set<string> missing_feeds, missing_fetches;
  for (const Feed& feed : config.feed()) {
    missing_feeds.insert(TensorIdToString(feed.id()));
  }
  for (const Fetch& fetch : config.fetch()) {
    missing_fetches.insert(TensorIdToString(fetch.id()));
  }
  for (const Node* n : graph->nodes()) {
    if (n->type_string() == kArgOp) {
      string feed_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), kFeedIdAttr, &feed_id));
      if (missing_feeds.erase(feed_id) == 0) {
        return errors::Aborted(kArgOp,
                               " node found with unknown feed id: ", feed_id);
      }
    } else if (n->type_string() == kRetvalOp) {
      string fetch_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), kFetchIdAttr, &fetch_id));
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
      TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
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
    TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "T", &arg.type));
    TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), kShapeAttr, &arg.shape));
    TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), kDebugNameAttr, &arg.name));
    xla_args->push_back(arg);
  }
  return Status::OK();
}

// Converts the TensorFlow graph into an XLA computation, by executing the
// graph symbolically, with each op building up the XLA HLO.
Status ConvertGraphToXla(xla::LocalClient* client, std::unique_ptr<Graph> graph,
                         const FunctionLibraryDefinition* flib_def,
                         xla::Computation* computation, bool* has_context_arg) {
  // Create a device and context to convert the graph into an XLA computation.
  XlaOpRegistry::RegisterCompilationKernels();
  // Populate the context with args from the graph.
  for (Node* node : graph->nodes()) {
    node->set_assigned_device_name(DEVICE_CPU_XLA_JIT);
  }
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(CreateXlaArgs(*graph, &xla_args));

  // Compile the graph into an XLA computation.
  XlaCompiler::Options compiler_options;
  compiler_options.client = client;
  compiler_options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
  compiler_options.allow_cpu_custom_calls = true;
  XlaCompiler compiler(compiler_options);

  std::unique_ptr<FunctionLibraryRuntime> flib_run(NewFunctionLibraryRuntime(
      compiler.device_mgr(), Env::Default(), compiler.device(),
      graph->versions().producer(), flib_def, OptimizerOptions()));
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(compiler.CompileGraph("tfcompile", std::move(graph),
                                           flib_run.get(), xla_args, &result));
  *has_context_arg = result.requires_runtime_context;
  *computation = std::move(result.computation);

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
  if (computation->IsNull()) {
    return errors::Aborted(
        "Conversion from TensorFlow graph to XLA resulted in an empty "
        "computation.");
  }
  return Status::OK();
}

// Compiles the XLA computation into executable code.
Status CompileXla(xla::LocalClient* client, const xla::Computation& computation,
                  const xla::cpu::CpuAotCompilationOptions& aot_opts,
                  CompileResult* compile_result) {
  // Retrieves arg and result layouts from the computation.
  // TODO(toddw): Should we let the user choose the major/minor ordering?
  xla::StatusOr<std::unique_ptr<xla::ProgramShape>> pshape_or =
      client->GetComputationShape(computation);
  if (!pshape_or.ok()) {
    return errors::Unknown("Couldn't get XLA program shape: ",
                           pshape_or.status().error_message());
  }
  compile_result->program_shape = *pshape_or.ValueOrDie();
  xla::ProgramShape* pshape = &compile_result->program_shape;
  std::vector<const xla::Shape*> arg_layouts;
  for (int i = 0; i < pshape->parameters_size(); ++i) {
    arg_layouts.push_back(pshape->mutable_parameters(i));
  }
  xla::LocalClient::AheadOfTimeComputationInstance instance;
  instance.computation = &computation;
  instance.argument_layouts = std::move(arg_layouts);
  instance.result_layout = &pshape->result();
  xla::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>>
      aot_or = client->CompileAheadOfTime({instance}, aot_opts);
  if (!aot_or.ok()) {
    return errors::Unknown("XLA compilation failed: ",
                           aot_or.status().error_message());
  }
  compile_result->aot =
      xla::unique_ptr_static_cast<xla::cpu::CpuAotCompilationResult>(
          std::move(aot_or.ValueOrDie().back()));
  compile_result->entry_point = aot_opts.entry_point_name();
  compile_result->pointer_size =
      xla::LocalClient::PointerSizeForTriple(aot_opts.triple());
  return Status::OK();
}

}  // namespace

Status InitGraph(const GraphDef& graph_def, const Config& config,
                 const MainFlags& flags, const FunctionLibraryDefinition* flib,
                 std::unique_ptr<Graph>* graph) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  std::unique_ptr<Graph> g(new Graph(flib));
  GraphDef copy_def(graph_def);
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&copy_def, *g->op_registry(),
                                               0 /*node_offset*/));
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), copy_def, g.get()));
  TF_RETURN_IF_ERROR(RewriteAndPruneGraph(g.get(), config, flags));
  *graph = std::move(g);
  return Status::OK();
}

Status CompileGraph(std::unique_ptr<Graph> graph, const MainFlags& flags,
                    const FunctionLibraryDefinition* flib,
                    CompileResult* compile_result) {
  // Converts the graph into an XLA computation, and compiles the
  // computation.
  // TODO(toddw): Should we let the user pick the XLA cpu vs. gpu client?
  namespace gpu = perftools::gputools;
  gpu::Platform* cpu_platform =
      gpu::MultiPlatformManager::PlatformWithName("Host").ValueOrDie();
  xla::LocalClient* client =
      xla::ClientLibrary::GetOrCreateLocalClient(cpu_platform).ValueOrDie();
  xla::Computation computation;
  TF_RETURN_IF_ERROR(ConvertGraphToXla(client, std::move(graph), flib,
                                       &computation,
                                       &compile_result->has_context_arg));
  if (!flags.debug_dir.empty()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::SessionModule> module,
                        computation.Snapshot());
    string file = io::JoinPath(flags.debug_dir, "tfcompile_xla_module.pb");
    TF_RETURN_IF_ERROR(WriteBinaryProto(Env::Default(), file, *module));
  }
  xla::cpu::CpuAotCompilationOptions aot_opts(
      flags.target_triple, flags.target_cpu, flags.target_features,
      flags.entry_point,
      xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic);
  return CompileXla(client, computation, aot_opts, compile_result);
}

}  // namespace tfcompile
}  // namespace tensorflow
