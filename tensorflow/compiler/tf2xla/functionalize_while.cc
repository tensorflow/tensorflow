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
#include "tensorflow/compiler/tf2xla/frontend_attributes_util.h"
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
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

using xla::StatusOr;

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
Status CopySubgraph(const Graph& graph, const WhileLoopFrame* frame,
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

    // Sort "n->in_edges()" to make sure nodes are copied in a deterministic
    // order.
    std::vector<const Edge*> sorted_edges(n->in_edges().begin(),
                                          n->in_edges().end());
    std::sort(sorted_edges.begin(), sorted_edges.end(),
              [](const Edge* a, const Edge* b) {
                int a_src_output = a->src_output(),
                    b_src_output = b->src_output();
                StringPiece a_name(a->src()->name()), b_name(b->src()->name());
                return std::tie(a_src_output, a_name) <
                       std::tie(b_src_output, b_name);
              });
    for (const Edge* e : sorted_edges) {
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
Status BuildLoopCondition(const Graph& graph, WhileLoopFrame* frame,
                          std::unique_ptr<Graph>* cond_output) {
  VLOG(2) << "Building loop condition for " << frame->name;
  *cond_output = absl::make_unique<Graph>(graph.op_registry());
  Graph* output = cond_output->get();

  // Map from nodes in the original graph to the condition graph.
  std::vector<Node*> node_map(graph.num_node_ids(), nullptr);
  std::vector<bool> squash_src_outputs(graph.num_node_ids(), false);

  // Build one _Arg node for each Enter node.
  for (int i = 0; i < frame->args.size(); ++i) {
    const WhileLoopArg& arg = frame->args[i];

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
Status BuildLoopBody(const Graph& graph, WhileLoopFrame* frame,
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
    const WhileLoopArg& arg = frame->args[i];

    DataType dtype = arg.enter->input_type(0);
    arg_types->push_back(dtype);

    TF_ASSIGN_OR_RETURN(Node * arg_node, BuildArgNode(output, dtype, i));
    TF_ASSIGN_OR_RETURN(Node * retval_node, BuildRetvalNode(output, dtype, i));
    if (arg.is_loop_invariant) {
      // Argument is loop-invariant. Forward it from the Arg to the Retval.
      node_map[arg.enter->id()] = arg_node;
      output->AddEdge(arg_node, 0, retval_node, 0);
    } else {
      // Argument is loop-varying.
      if (dtype == DT_RESOURCE) {
        // DT_RESOURCE arguments should always be loop-invariant in the graphs
        // generated from TF.
        return errors::Unimplemented("Loop-varying DT_RESOURCE Enter node ",
                                     arg.enter->name(), " is currently not",
                                     " supported.");
      }
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

  // Performs a reverse DFS, copying nodes and edges to the output graph.
  // The _Arg and _Retval nodes were added unconditionally above, so we are
  // guaranteed to get the correct function signature.
  TF_RETURN_IF_ERROR(CopySubgraph(graph, frame, std::move(next_iterations),
                                  squash_src_outputs, &node_map, output));

  return Status::OK();
}

Status FunctionalizeLoop(Graph* graph, WhileLoopFrame* frame,
                         FunctionLibraryDefinition* library) {
  VLOG(2) << "Frame " << frame->name << " before: "
          << DumpGraphToFile("functionalize_before", *graph, library);

  // Split loop-varying Enter nodes with multiple successors. If the same
  // Tensor is fed as input to multiple loop arguments, we may end up with a
  // shared Enter node. We clone Enter nodes with multiple successors to
  // maintain the invariant of a unique Enter node per argument of the final
  // loop.
  std::vector<WhileLoopArg> args;
  for (const WhileLoopArg& arg : frame->args) {
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
        WhileLoopArg new_arg;
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
            [](const WhileLoopArg& a, const WhileLoopArg& b) {
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
  for (WhileLoopArg& arg : frame->args) {
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
          << DumpGraphToFile("loop_condition", *cond_graph, library)
          << " body: " << DumpGraphToFile("loop_body", *body_graph);

  NameAttrList cond_name;
  cond_name.set_name(library->UniqueFunctionName("_functionalize_cond_"));
  NameAttrList body_name;
  body_name.set_name(library->UniqueFunctionName("_functionalize_body_"));
  FunctionDef cond_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*cond_graph, cond_name.name(), &cond_fdef));
  FunctionDef body_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*body_graph, body_name.name(), &body_fdef));

  TF_RETURN_IF_ERROR(library->AddFunctionDef(cond_fdef));
  TF_RETURN_IF_ERROR(library->AddFunctionDef(body_fdef));

  // Builds a While operator.
  NodeDef while_def;
  NodeDefBuilder builder(frame->loop_cond->name(), "While", library);
  builder.Attr("T", arg_types);
  builder.Attr("cond", cond_name);
  builder.Attr("body", body_name);
  string outside_compilation;
  string frontend_attributes;
  if (GetNodeAttr(frame->loop_cond->def(), kXlaFrontendAttributesAttrName,
                  &frontend_attributes)
          .ok()) {
    builder.Attr(kXlaFrontendAttributesAttrName, frontend_attributes);
  }
  if (GetNodeAttr(frame->loop_cond->def(), kXlaOutsideCompilationAttrName,
                  &outside_compilation)
          .ok()) {
    builder.Attr(kXlaOutsideCompilationAttrName, outside_compilation);
  }
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (int i = 0; i < frame->args.size(); ++i) {
    const WhileLoopArg& arg = frame->args[i];
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
    const WhileLoopArg& arg = frame->args[i];
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
          << DumpGraphToFile("functionalize_after", *graph, library);

  return Status::OK();
}
}  // namespace

Status FunctionalizeWhileLoop(Graph* graph,
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
  std::unordered_map<string, WhileLoopFrame> frames;
  TF_RETURN_IF_ERROR(ExtractWhileLoopFrames(cf_info, graph, &frames));

  // Adds frames with no children (i.e., the innermost frames) to a worklist.
  std::deque<WhileLoopFrame*> worklist;
  for (auto& frame : frames) {
    if (frame.second.num_children == 0) {
      worklist.push_back(&frame.second);
    }
  }

  // Eliminate loops from innermost to outermost.
  while (!worklist.empty()) {
    WhileLoopFrame* frame = worklist.front();
    worklist.pop_front();
    if (frame->parent == frame) {
      // Skip the root frame.
      continue;
    }

    TF_RETURN_IF_ERROR(FunctionalizeLoop(graph, frame, library));

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
