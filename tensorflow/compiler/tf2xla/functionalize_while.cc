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

#include "tensorflow/compiler/tf2xla/functionalize_while.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_cond.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

using xla::StatusOr;

// Information about a loop argument.
struct Arg {
  // Every loop argument has an Enter node.
  Node* enter;

  // Is the loop argument a loop-invariant value? Taken from the `is_constant`
  // attribute on the Enter node.
  bool is_loop_invariant;

  // If 'is_loop_invariant' is true, the following are all nullptr. Non-constant
  // arguments must have all of the following nodes:
  Node* merge = nullptr;
  Node* switch_node = nullptr;
  Node* next_iteration = nullptr;
  Node* exit = nullptr;
};

// Information about a loop frame.
struct Frame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  Frame* parent = nullptr;
  int num_children = 0;

  // Arguments to this loop.
  std::vector<Arg> args;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  Node* loop_cond = nullptr;

  // Set of nodes that belong to the loop frame.
  std::unordered_set<Node*> nodes;
};

// Copies a subgraph from `graph` to `output` by performing a reverse DFS
// starting at nodes in vector `stack`.
// `node_map` is a vector indexed by source node ID to dest nodes.
// Does not traverse into nodes in `node_map`, so by adding nodes to `node_map`
// before the traversal clients can cut the graph. If a frame is provided (frame
// != nullptr), then this functions will return an error if the
// traversal leaves 'frame'; the client must add enough nodes to `node_map` to
// cut the graph and prevent the traversal from escaping.
//
// `squash_src_outputs` contains a bool for each source node ID. If true, then
// the source output on that node will be replaced by zero when copied. This is
// used when replacing a Switch node with an _Arg node. The output we are
// taking from the Switch node was not necessarily the first output, but _Arg
// nodes only have one output. By adding the Switch node to `squash_src_outputs`
// we rewrite the src_output of the corresponding edge to be 0.
Status CopySubgraph(const Graph& graph, const Frame* frame,
                    std::vector<Node*> stack,
                    const std::vector<bool>& squash_src_outputs,
                    std::vector<Node*>* node_map, Graph* output) {
  VLOG(3) << "Stack: " << NodesToString(stack);
  std::vector<bool> visited(graph.num_node_ids(), false);
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();

    VLOG(5) << "Copying node " << n->name();

    if (visited[n->id()]) continue;
    visited[n->id()] = true;

    for (const Edge* e : n->in_edges()) {
      Node* src = e->src();
      if (frame != nullptr && frame->nodes.find(src) == frame->nodes.end()) {
        // We traversed out of the loop frame, without encountering a cut node.
        return errors::Internal("Graph traversal of loop frame ", frame->name,
                                " escaped frame at ", src->name(),
                                " without encountering an argument node.");
      }
      if ((*node_map)[src->id()] == nullptr) {
        (*node_map)[src->id()] = output->CopyNode(src);
        stack.push_back(src);
      }
      Node* src_copy = (*node_map)[e->src()->id()];
      int src_output = squash_src_outputs[e->src()->id()] && !e->IsControlEdge()
                           ? 0
                           : e->src_output();
      Node* dst_copy = (*node_map)[e->dst()->id()];
      output->AddEdge(src_copy, src_output, dst_copy, e->dst_input());
    }
  }
  return Status::OK();
}

StatusOr<Node*> BuildArgNode(Graph* graph, DataType type, int index) {
  const char* const kArgOp = "_Arg";
  NodeDef arg_def;
  NodeDefBuilder builder(absl::StrCat(kArgOp, index), kArgOp);
  builder.Attr("T", type);
  builder.Attr("index", index);
  TF_RETURN_IF_ERROR(builder.Finalize(&arg_def));
  return AddNodeDefToGraph(arg_def, graph);
}

// Builds a graph for the loop condition.
Status BuildLoopCondition(const Graph& graph, Frame* frame,
                          std::unique_ptr<Graph>* cond_output) {
  VLOG(2) << "Building loop condition for " << frame->name;
  *cond_output = absl::make_unique<Graph>(graph.op_registry());
  Graph* output = cond_output->get();

  // Map from nodes in the original graph to the condition graph.
  std::vector<Node*> node_map(graph.num_node_ids(), nullptr);
  std::vector<bool> squash_src_outputs(graph.num_node_ids(), false);

  // Build one _Arg node for each Enter node.
  for (int i = 0; i < frame->args.size(); ++i) {
    const Arg& arg = frame->args[i];

    TF_ASSIGN_OR_RETURN(Node * arg_node,
                        BuildArgNode(output, arg.enter->input_type(0), i));
    if (arg.is_loop_invariant) {
      node_map[arg.enter->id()] = arg_node;
    } else {
      node_map[arg.merge->id()] = arg_node;
    }
  }

  // Build a Retval node for the loop condition. The LoopCond nodes are always
  // boolean because of the type constraints on the LoopCond op.
  TF_ASSIGN_OR_RETURN(node_map[frame->loop_cond->id()],
                      BuildRetvalNode(output, DT_BOOL, 0));

  // Performs a reverse DFS, copying nodes and edges to the output graph.
  // The _Arg and _Retval nodes were added unconditionally above, so we are
  // guaranteed to get the correct function signature.
  return CopySubgraph(graph, frame, {frame->loop_cond}, squash_src_outputs,
                      &node_map, output);
}

// Builds a graph for the loop body.
Status BuildLoopBody(const Graph& graph, Frame* frame,
                     DataTypeVector* arg_types,
                     std::unique_ptr<Graph>* body_output) {
  VLOG(2) << "Building loop body for " << frame->name;
  *body_output = absl::make_unique<Graph>(graph.op_registry());
  Graph* output = body_output->get();

  // Map from nodes in the original graph to the condition graph.
  std::vector<Node*> node_map(graph.num_node_ids(), nullptr);
  std::vector<bool> squash_src_outputs(graph.num_node_ids(), false);

  // Build one _Arg node for each Enter node.
  std::vector<Node*> next_iterations;
  next_iterations.reserve(frame->args.size());
  arg_types->reserve(frame->args.size());
  for (int i = 0; i < frame->args.size(); ++i) {
    const Arg& arg = frame->args[i];

    DataType dtype = arg.enter->input_type(0);
    arg_types->push_back(dtype);

    TF_ASSIGN_OR_RETURN(Node * arg_node, BuildArgNode(output, dtype, i));

    if (dtype == DT_RESOURCE) {
      // The convention of the XLA bridge is that resource variable arguments
      // are only inputs to the loop body and have no corresponding output.
      // TODO(b/37741920): change the convention so that DT_RESOURCE variables
      // are both inputs and outputs, and then remove this case.
      TF_RET_CHECK(arg.is_loop_invariant);
      node_map[arg.enter->id()] = arg_node;
    } else {
      TF_ASSIGN_OR_RETURN(Node * retval_node,
                          BuildRetvalNode(output, dtype, i));

      if (arg.is_loop_invariant) {
        // Argument is loop-invariant. Forward it from the Arg to the Retval.
        node_map[arg.enter->id()] = arg_node;
        output->AddEdge(arg_node, 0, retval_node, 0);
      } else {
        // Argument is loop-varying.
        node_map[arg.switch_node->id()] = arg_node;
        // The Switch node has two outputs, but _Arg only has one. This tells
        // the CopySubgraph function to rewrite the output number of edges from
        // the _Arg node to be 0 rather than copying the output number from the
        // Switch node.
        squash_src_outputs[arg.switch_node->id()] = true;
        node_map[arg.next_iteration->id()] = retval_node;
        next_iterations.push_back(arg.next_iteration);
      }
    }
  }

  // Performs a reverse DFS, copying nodes and edges to the output graph.
  // The _Arg and _Retval nodes were added unconditionally above, so we are
  // guaranteed to get the correct function signature.
  TF_RETURN_IF_ERROR(CopySubgraph(graph, frame, std::move(next_iterations),
                                  squash_src_outputs, &node_map, output));

  return Status::OK();
}

// Copy the FunctionDef of given function from lookup_library to library, if
// it can be found in lookup_library but is missing from library.
Status AddMissingFunctionByName(const string& function_name,
                                const FunctionLibraryDefinition* lookup_library,
                                FunctionLibraryDefinition* library) {
  if (!library->Find(function_name) && lookup_library->Find(function_name)) {
    return library->AddFunctionDef(*lookup_library->Find(function_name));
  }
  return Status::OK();
}

// Iterate over all functions that the given fdef refers to. Copy the missing
// FunctionDefs from lookup_library to library.
Status AddMissingFunctionDef(const FunctionDef& fdef,
                             const FunctionLibraryDefinition* lookup_library,
                             FunctionLibraryDefinition* library) {
  TF_RET_CHECK(lookup_library);
  for (const NodeDef& node : fdef.node_def()) {
    if (library->Find(node.op())) {
      continue;
    }
    // The function referred by 'SymbolicGradient' node is specified in its
    // attribute 'f'.
    if (node.op() == FunctionLibraryDefinition::kGradientOp) {
      const AttrValue* attr =
          AttrSlice(&node.attr()).Find(FunctionLibraryDefinition::kFuncAttr);
      if (!attr) {
        return errors::InvalidArgument("SymbolicGradient is missing attr: f");
      }
      const string& func_name = attr->func().name();
      TF_RETURN_IF_ERROR(
          AddMissingFunctionByName(func_name, lookup_library, library));
      // Copy the user-defined gradient function if it exists.
      const string grad_name = lookup_library->FindGradient(func_name);
      if (!grad_name.empty() && library->FindGradient(func_name).empty()) {
        TF_RETURN_IF_ERROR(
            AddMissingFunctionByName(grad_name, lookup_library, library));
        GradientDef grad_def;
        grad_def.set_function_name(func_name);
        grad_def.set_gradient_func(grad_name);
        TF_RETURN_IF_ERROR(library->AddGradientDef(grad_def));
      }
    } else if (lookup_library->Find(node.op())) {
      TF_RETURN_IF_ERROR(
          library->AddFunctionDef(*lookup_library->Find(node.op())));
    }
  }
  return Status::OK();
}

Status FunctionalizeLoop(const FunctionLibraryDefinition* lookup_library,
                         Graph* graph, Frame* frame,
                         FunctionLibraryDefinition* library) {
  VLOG(2) << "Frame " << frame->name << " before: "
          << dump_graph::DumpGraphToFile("functionalize_before", *graph,
                                         library);

  // Split loop-varying Enter nodes with multiple successors. If the same
  // Tensor is fed as input to multiple loop arguments, we may end up with a
  // shared Enter node. We clone Enter nodes with multiple successors to
  // maintain the invariant of a unique Enter node per argument of the final
  // loop.
  std::vector<Arg> args;
  for (const Arg& arg : frame->args) {
    if (arg.is_loop_invariant) {
      args.push_back(arg);
    } else {
      std::vector<const Edge*> edges(arg.enter->out_edges().begin(),
                                     arg.enter->out_edges().end());
      for (int i = 0; i < edges.size(); ++i) {
        if (edges[i]->IsControlEdge() && edges[i]->dst()->IsSink()) {
          continue;
        }
        TF_RET_CHECK(!edges[i]->IsControlEdge()) << edges[i]->src()->name();
        Arg new_arg;
        new_arg.is_loop_invariant = false;
        if (i == 0) {
          new_arg.enter = arg.enter;
        } else {
          new_arg.enter = graph->CopyNode(arg.enter);
          frame->nodes.insert(new_arg.enter);
          for (Edge const* e : arg.enter->in_edges()) {
            graph->AddEdge(e->src(), e->src_output(), new_arg.enter,
                           e->IsControlEdge() ? Graph::kControlSlot : 0);
          }
          Node* dst = edges[i]->dst();
          int dst_input = edges[i]->dst_input();
          graph->RemoveEdge(edges[i]);
          graph->AddEdge(new_arg.enter, 0, dst, dst_input);
        }
        args.push_back(new_arg);
      }
    }
  }
  frame->args = std::move(args);

  std::sort(frame->args.begin(), frame->args.end(),
            [](const Arg& a, const Arg& b) {
              return NodeCmpByNameResourcesLast()(a.enter, b.enter);
            });

  if (frame->loop_cond == nullptr) {
    return errors::InvalidArgument("Loop ", frame->name,
                                   " has no LoopCond node");
  }

  // Find the set of Switch nodes that are successors of the LoopCond.
  std::unordered_set<Node*> switches;
  for (const Edge* edge : frame->loop_cond->out_edges()) {
    if (!edge->IsControlEdge() && IsSwitch(edge->dst()) &&
        edge->dst_input() == 1) {
      switches.insert(edge->dst());
    }
  }

  // For each non-constant argument, looks for the following pattern of nodes:
  // Enter ----> Merge  -------->  Switch  --> Exit
  //               ^                  ^
  //               |                  |
  //         NextIteration         LoopCond
  //               ^                  ^
  //               |                  |
  //              ...                ...
  for (Arg& arg : frame->args) {
    if (!arg.is_loop_invariant) {
      // Follow the edge from the Enter to Merge.
      const Edge* enter_merge = nullptr;
      for (const Edge* e : arg.enter->out_edges()) {
        // Ignore control-edges to the sink node. These are allowed by the
        // graph invariants, although probably they should have been stripped
        // off earlier.
        if (e->IsControlEdge() && e->dst()->IsSink()) {
          continue;
        }
        if (enter_merge != nullptr) {
          return errors::Internal("Enter node for loop-varying argument ",
                                  FormatNodeForError(*arg.enter),
                                  " has multiple successors: ",
                                  FormatNodeForError(*enter_merge->dst()),
                                  " and ", FormatNodeForError(*e->dst()));
        }
        enter_merge = e;
      }
      if (enter_merge == nullptr) {
        return errors::Internal("Enter node for loop-varying argument ",
                                FormatNodeForError(*arg.enter),
                                " has zero successors");
      }
      arg.merge = enter_merge->dst();
      if (!IsMerge(arg.merge)) {
        return errors::InvalidArgument(
            "Successor of Enter node for loop-varying argument ",
            FormatNodeForError(*arg.merge),
            " is not a Merge node; got: ", arg.merge->type_string());
      }

      // Find the NextIteration from the merge. There should be two inputs to
      // the Merge and the NextIteration should be the other input.
      if (arg.merge->input_types().size() != 2) {
        return errors::InvalidArgument(
            "Unexpected number of inputs to Merge node for loop-varying "
            "argument ",
            FormatNodeForError(*arg.merge), "; expected 2, got ",
            arg.merge->input_types().size());
      }
      TF_RETURN_IF_ERROR(arg.merge->input_node(1 - enter_merge->dst_input(),
                                               &arg.next_iteration));
      if (!IsNextIteration(arg.next_iteration)) {
        return errors::InvalidArgument(
            "Expected NextIteration node as input to Merge node; got node ",
            FormatNodeForError(*arg.next_iteration), " with kind ",
            arg.next_iteration->type_string());
      }

      // Find the Switch successor of the Merge. There should be exactly one
      // Switch node that is a successor of both the Merge and the LoopCond.
      for (const Edge* edge : arg.merge->out_edges()) {
        if (edge->dst_input() == 0 && IsSwitch(edge->dst()) &&
            switches.find(edge->dst()) != switches.end()) {
          if (arg.switch_node != nullptr) {
            return errors::InvalidArgument("Duplicate Switch successors to ",
                                           FormatNodeForError(*arg.merge));
          }
          arg.switch_node = edge->dst();
        }
      }
      if (arg.switch_node == nullptr) {
        return errors::InvalidArgument("Missing Switch successor to ",
                                       FormatNodeForError(*arg.merge));
      }

      // Update the device on the Identity outputs of the switch to match their
      // target. These Identity outputs do not

      // Loop over the switch node's output to:
      // - Find the Exit successor.
      // - Set the sharding on all Identity outputs of the switch. These
      //   identity nodes are values used by the loop body or condition.
      //   The Identity node may have the wrong device so copy the device from
      //   one of its outputs instead.
      std::deque<const Edge*> possible_exit;
      for (const Edge* edge : arg.switch_node->out_edges()) {
        if (edge->src_output() == 0) {
          possible_exit.push_back(edge);
        }
        if (IsIdentity(edge->dst())) {
          TF_RETURN_IF_ERROR(
              SetNodeShardingFromNeighbors(edge->dst(), /*out_edges=*/true));
        }
      }
      // TODO(b/67425339): Allow general graph between switch and exit.
      while (!possible_exit.empty()) {
        const Edge* edge = possible_exit.front();
        possible_exit.pop_front();
        if (IsExit(edge->dst())) {
          if (arg.exit != nullptr) {
            return errors::InvalidArgument(
                "Duplicate Exit successors to ",
                FormatNodeForError(*arg.switch_node));
          }
          arg.exit = edge->dst();
        } else {
          if (!IsIdentity(edge->dst())) {
            return errors::Unimplemented("General graph between switch (",
                                         FormatNodeForError(*arg.switch_node),
                                         ") and exit node of frame ",
                                         frame->name, " not supported yet.");
          }
          for (const Edge* out : edge->dst()->out_edges()) {
            possible_exit.push_back(out);
          }
        }
      }
    }
  }

  // Builds the condition and body functions. Notice that we call
  // FunctionalizeCond() on cond_graph and body_graph because we might have
  // unfunctionalized "if" in cond_graph and body_graph. Functionalize them
  // before they are encapsulated in FunctionDef.
  std::unique_ptr<Graph> cond_graph;
  TF_RETURN_IF_ERROR(BuildLoopCondition(*graph, frame, &cond_graph));
  FixupSourceAndSinkEdges(cond_graph.get());
  TF_RETURN_IF_ERROR(FunctionalizeCond(cond_graph.get(), library));
  DataTypeVector arg_types;
  std::unique_ptr<Graph> body_graph;
  TF_RETURN_IF_ERROR(BuildLoopBody(*graph, frame, &arg_types, &body_graph));
  FixupSourceAndSinkEdges(body_graph.get());
  TF_RETURN_IF_ERROR(FunctionalizeCond(body_graph.get(), library));

  VLOG(2) << "Frame " << frame->name << " condition: "
          << dump_graph::DumpGraphToFile("loop_condition", *cond_graph, library)
          << " body: " << dump_graph::DumpGraphToFile("loop_body", *body_graph);

  static std::atomic<int64> sequence_num(0LL);
  int64 id = ++sequence_num;
  NameAttrList cond_name;
  cond_name.set_name(absl::StrCat("_functionalize_cond_", id));
  NameAttrList body_name;
  body_name.set_name(absl::StrCat("_functionalize_body_", id));
  FunctionDef cond_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*cond_graph, cond_name.name(), &cond_fdef));
  FunctionDef body_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*body_graph, body_name.name(), &body_fdef));

  TF_RETURN_IF_ERROR(library->AddFunctionDef(cond_fdef));
  TF_RETURN_IF_ERROR(library->AddFunctionDef(body_fdef));
  if (lookup_library) {
    // Copy missing FunctionDefs from lookup_library to library to make library
    // self-contained.
    TF_RETURN_IF_ERROR(
        AddMissingFunctionDef(cond_fdef, lookup_library, library));
    TF_RETURN_IF_ERROR(
        AddMissingFunctionDef(body_fdef, lookup_library, library));
  }

  // Builds a While operator.
  NodeDef while_def;
  NodeDefBuilder builder(frame->loop_cond->name(), "While", library);
  builder.Attr("T", arg_types);
  builder.Attr("cond", cond_name);
  builder.Attr("body", body_name);
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (int i = 0; i < frame->args.size(); ++i) {
    const Arg& arg = frame->args[i];
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(arg.enter->input_edge(0, &in_edge));
    if (in_edge->IsControlEdge()) {
      builder.ControlInput(in_edge->src()->name());
    } else {
      inputs.push_back(NodeDefBuilder::NodeOut(
          in_edge->src()->name(), in_edge->src_output(), arg_types[i]));
    }
  }
  builder.Input(inputs);
  TF_RETURN_IF_ERROR(builder.Finalize(&while_def));
  TF_ASSIGN_OR_RETURN(Node * while_node, AddNodeDefToGraph(while_def, graph));

  // Copies edges to the Enter nodes and from the Exit nodes onto the While.
  for (int i = 0; i < frame->args.size(); ++i) {
    const Arg& arg = frame->args[i];
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(arg.enter->input_edge(0, &in_edge));
    if (in_edge->IsControlEdge()) {
      graph->AddControlEdge(in_edge->src(), while_node);
    } else {
      graph->AddEdge(in_edge->src(), in_edge->src_output(), while_node, i);
    }

    if (!arg.is_loop_invariant) {
      // Add output edges if the output of the loop is consumed.
      if (arg.exit != nullptr) {
        std::vector<const Edge*> edges(arg.exit->out_edges().begin(),
                                       arg.exit->out_edges().end());
        for (const Edge* edge : edges) {
          Node* dst = edge->dst();
          int dst_input = edge->dst_input();
          graph->RemoveEdge(edge);

          if (dst_input == Graph::kControlSlot) {
            graph->AddControlEdge(while_node, dst);
          } else {
            graph->AddEdge(while_node, i, dst, dst_input);
          }
        }
      }
    }
  }

  // Remove the old nodes from the graph, and add the while node to the parent
  // frame.
  for (Node* node : frame->nodes) {
    graph->RemoveNode(node);
  }
  frame->nodes.clear();
  frame->parent->nodes.insert(while_node);

  VLOG(2) << "Frame " << frame->name << " after: "
          << dump_graph::DumpGraphToFile("functionalize_after", *graph,
                                         library);

  return Status::OK();
}
}  // namespace

Status FunctionalizeWhileLoop(const FunctionLibraryDefinition* lookup_library,
                              Graph* graph,
                              FunctionLibraryDefinition* library) {
  // Note: BuildControlFlowInfo() requires that the graph's source node is
  // connected to all source nodes in the graph. Many graphs violate this
  // invariant.
  std::vector<ControlFlowInfo> cf_info;
  std::vector<string> unreachable_nodes;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &cf_info, &unreachable_nodes));
  if (!unreachable_nodes.empty()) {
    return errors::InvalidArgument(
        "The following nodes are unreachable from the source in the graph: ",
        errors::FormatNodeNamesForError(unreachable_nodes));
  }

  // Builds Frames, indexed by name.
  std::unordered_map<string, Frame> frames;
  for (Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];

    VLOG(2) << "node: " << node->name() << " (" << node->id()
            << ") frame_name: " << cf.frame_name
            << " frame: " << (cf.frame ? cf.frame->name() : "---")
            << " parent_frame: "
            << (cf.parent_frame ? cf.parent_frame->name() : "---");
    TF_RET_CHECK(cf.frame != nullptr && cf.parent_frame != nullptr);

    Frame& frame = frames[cf.frame_name];
    Frame* parent = &frames[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
      ++parent->num_children;
    }

    if (IsEnter(node)) {
      Arg arg;
      arg.enter = node;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg.enter->attrs(), "is_constant",
                                     &arg.is_loop_invariant));
      frame.args.push_back(arg);
    } else if (IsLoopCond(node)) {
      frame.loop_cond = node;
    }
    frame.nodes.insert(node);
  }

  // Adds frames with no children (i.e., the innermost frames) to a worklist.
  std::deque<Frame*> worklist;
  for (auto& frame : frames) {
    if (frame.second.num_children == 0) {
      worklist.push_back(&frame.second);
    }
  }

  // Eliminate loops from innermost to outermost.
  while (!worklist.empty()) {
    Frame* frame = worklist.front();
    worklist.pop_front();
    if (frame->parent == frame) {
      // Skip the root frame.
      continue;
    }

    TF_RETURN_IF_ERROR(
        FunctionalizeLoop(lookup_library, graph, frame, library));

    // If the parent has no remaining children, add it to the worklist.
    --frame->parent->num_children;
    if (frame->parent->num_children == 0) {
      worklist.push_back(frame->parent);
    }
  }

  // There should be no cycle at this point, since while loops have been removed
  // from graph.
  // Check that the newly added While nodes don't feed into themselves.
  for (const Node* node : graph->op_nodes()) {
    if (node->def().op() == "While") {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          CheckNodeNotInCycle(node, graph->num_node_ids()),
          "Functionalizing loop failed.");
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
