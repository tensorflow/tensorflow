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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/optional.h"

namespace tensorflow {

namespace {

using xla::StatusOr;

const char* const kArgOp = "_Arg";
const char* const kRetValOp = "_Retval";

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

// Comparison function used for sorting nodes consistently.
// a) resource variables are last, and
// b) sort lexicographically by name (for deterministic output).
struct NodeCmp {
  bool operator()(const Node* lhs, const Node* rhs) const {
    bool lhs_is_resource =
        lhs->num_inputs() > 0 ? (lhs->input_type(0) == DT_RESOURCE) : false;
    bool rhs_is_resource =
        rhs->num_inputs() > 0 ? (rhs->input_type(0) == DT_RESOURCE) : false;
    return std::tie(lhs_is_resource, lhs->name()) <
           std::tie(rhs_is_resource, rhs->name());
  }
};

// Returns a textual representation of the names of the nodes in the input.
template <typename T>
string NodesToString(const T& nodes) {
  return strings::StrCat("{",
                         str_util::Join(nodes, ",",
                                        [](string* output, const Node* node) {
                                          strings::StrAppend(output,
                                                             node->name());
                                        }),
                         "}");
}

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

StatusOr<Node*> AddNode(const NodeDef& node_def, Graph* graph) {
  Status status;
  Node* inserted_node = graph->AddNode(node_def, &status);
  if (!status.ok()) {
    return status;
  }
  return inserted_node;
}

StatusOr<Node*> BuildArgNode(Graph* graph, DataType type, int index) {
  NodeDef arg_def;
  NodeDefBuilder builder(strings::StrCat(kArgOp, index), kArgOp);
  builder.Attr("T", type);
  builder.Attr("index", index);
  TF_RETURN_IF_ERROR(builder.Finalize(&arg_def));
  return AddNode(arg_def, graph);
}

StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index) {
  NodeDef ret_def;
  ret_def.set_op(kRetValOp);
  ret_def.set_name(strings::StrCat(kRetValOp, index));
  AddNodeAttr("T", type, &ret_def);
  AddNodeAttr("index", index, &ret_def);
  return AddNode(ret_def, graph);
}

// Builds a graph for the loop condition.
Status BuildLoopCondition(const Graph& graph, Frame* frame,
                          std::unique_ptr<Graph>* cond_output) {
  VLOG(2) << "Building loop condition for " << frame->name;
  *cond_output = xla::MakeUnique<Graph>(graph.op_registry());
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
  *body_output = xla::MakeUnique<Graph>(graph.op_registry());
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
    // The function refered by 'SymbolicGradient' node is specified in its
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

  std::sort(
      frame->args.begin(), frame->args.end(),
      [](const Arg& a, const Arg& b) { return NodeCmp()(a.enter, b.enter); });

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
          return errors::Internal(
              "Enter node for loop-varying argument ", arg.enter->name(),
              " has multiple successors: ", enter_merge->dst()->name(), " and ",
              e->dst()->name());
        }
        enter_merge = e;
      }
      if (enter_merge == nullptr) {
        return errors::Internal("Enter node for loop-varying argument ",
                                arg.enter->name(), " has zero successors");
      }
      arg.merge = enter_merge->dst();
      if (!IsMerge(arg.merge)) {
        return errors::InvalidArgument(
            "Successor of Enter node for loop-varying argument ",
            arg.merge->name(),
            " is not a Merge node; got: ", arg.merge->type_string());
      }

      // Find the NextIteration from the merge. There should be two inputs to
      // the Merge and the NextIteration should be the other input.
      if (arg.merge->input_types().size() != 2) {
        return errors::InvalidArgument(
            "Unexpected number of inputs to Merge node for loop-varying "
            "argument ",
            arg.merge->name(), "; expected 2, got ",
            arg.merge->input_types().size());
      }
      TF_RETURN_IF_ERROR(arg.merge->input_node(1 - enter_merge->dst_input(),
                                               &arg.next_iteration));
      if (!IsNextIteration(arg.next_iteration)) {
        return errors::InvalidArgument(
            "Expected NextIteration node as input to Merge node; got node ",
            arg.next_iteration->name(), " with kind ",
            arg.next_iteration->type_string());
      }

      // Find the Switch successor of the Merge. There should be exactly one
      // Switch node that is a successor of both the Merge and the LoopCond.
      for (const Edge* edge : arg.merge->out_edges()) {
        if (edge->dst_input() == 0 && IsSwitch(edge->dst()) &&
            switches.find(edge->dst()) != switches.end()) {
          if (arg.switch_node != nullptr) {
            return errors::InvalidArgument("Duplicate Switch successors to ",
                                           arg.merge->name());
          }
          arg.switch_node = edge->dst();
        }
      }
      if (arg.switch_node == nullptr) {
        return errors::InvalidArgument("Missing Switch successor to ",
                                       arg.merge->name());
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
            return errors::InvalidArgument("Duplicate Exit successors to ",
                                           arg.switch_node->name());
          }
          arg.exit = edge->dst();
        } else {
          if (!IsIdentity(edge->dst())) {
            return errors::Unimplemented("General graph between switch (",
                                         arg.switch_node->name(),
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

  // Builds the condition and body functions.
  std::unique_ptr<Graph> cond_graph;
  TF_RETURN_IF_ERROR(BuildLoopCondition(*graph, frame, &cond_graph));
  DataTypeVector arg_types;
  std::unique_ptr<Graph> body_graph;
  TF_RETURN_IF_ERROR(BuildLoopBody(*graph, frame, &arg_types, &body_graph));

  VLOG(2) << "Frame " << frame->name << " condition: "
          << dump_graph::DumpGraphToFile("loop_condition", *cond_graph, library)
          << " body: " << dump_graph::DumpGraphToFile("loop_body", *body_graph);

  static std::atomic<int64> sequence_num(0LL);
  int64 id = ++sequence_num;
  NameAttrList cond_name;
  cond_name.set_name(strings::StrCat("_functionalize_cond_", id));
  NameAttrList body_name;
  body_name.set_name(strings::StrCat("_functionalize_body_", id));
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
  NodeDefBuilder builder(frame->loop_cond->name(), "XlaWhile");
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
  TF_ASSIGN_OR_RETURN(Node * while_node, AddNode(while_def, graph));

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

class FunctionalizeCond {
 public:
  // All nodes are assumed to be either in no branch, then branch, else branch,
  // or both branches (such as merge nodes).
  enum Branch {
    kElseBranch = 0,
    kThenBranch = 1,
    kBoth = 2,
    kNeither = 3,
    kNumBranchTypes = 4
  };

  // Returns a textual representation of the Branch b.
  static string Branch_Name(FunctionalizeCond::Branch b);

  // Functionalize all the switch-merge nodes of a loop-free graph into XlaIf
  // nodes. That is, attempt to transform every remaining switch and merge nodes
  // in the graph into XlaIf nodes.
  // Precondition: All while loops have been removed from graph.
  static Status Functionalize(Graph* graph, FunctionLibraryDefinition* library);

 private:
  // CondArgNode represents a input to the conditional and its corresponding
  // switch nodes.
  struct CondArgNode {
    explicit CondArgNode(Node* src, int src_output)
        : src(src), src_output(src_output) {}
    string ToString() const {
      return strings::StrCat("src=", src->name(), ":", src_output,
                             " switches=", NodesToString(switches));
    }

    Node* src;
    int src_output;
    std::vector<Node*> switches;
  };
  using CondArgNodes = std::vector<CondArgNode>;

  struct ForwardFlowNode {
    explicit ForwardFlowNode(Branch branch = Branch::kNeither)
        : branch(branch), count(0) {}
    string ToString() const {
      return strings::StrCat("branch=", Branch_Name(branch), " count=", count);
    }
    Branch branch;
    int count;
  };

  // Group of switch nodes that will be part of the same XlaIf.
  struct SwitchCluster {
    explicit SwitchCluster(const Edge* predicate_edge)
        : predicate_edge(predicate_edge) {}
    string ToString() const {
      return strings::StrCat(name, " predicate=", predicate_edge->src()->name(),
                             " switches=", NodesToString(switches));
    }

    string name;
    const Edge* predicate_edge;
    std::vector<Node*> switches;
  };

  FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library,
                    bool dump_graphs)
      : library_(library), graph_(graph), dump_graphs_(dump_graphs) {}

  // Perform the actual cond functionalization. Iterate over groups of switch
  // nodes (linked by common predicate), from innermost to outermost, and
  // extract into XlaIf nodes.
  Status FunctionalizeInternal();

  // Determines the branch_map (mapping from node to branch of cond) and
  // frontier (the nodes where the cond ends).
  StatusOr<std::pair<std::unordered_map<Node*, ForwardFlowNode>,
                     std::unordered_set<Node*>>>
  DetermineBranchMapAndFrontier(const SwitchCluster& switch_cluster);

  // Returns XlaIf node created from subgraph of merge and switch nodes. This
  // encapsulates the process of extracting the bodies needed for the then and
  // else branch, creates a XlaIf node, removing the nodes of the branches from
  // the graph and replacing the merge node with a XlaIf.
  StatusOr<Node*> ConvertToXlaIf(const CondArgNodes& cond_arg_nodes,
                                 const SwitchCluster& switch_cluster,
                                 const std::vector<Node*>& switches);

  // Builds a XlaIfOp to replace the Switch-Graph-Merge cluster with.
  StatusOr<Node*> BuildAndAddXlaIfOp(const CondArgNodes& cond_arg_nodes,
                                     const SwitchCluster& switch_cluster,
                                     const std::vector<Node*>& merge_nodes);

  // Extracts a function body corresponding to the given input edge of the merge
  // node.
  Status ExtractBody(const CondArgNodes& cond_arg_nodes,
                     const std::vector<Node*>& switches,
                     const std::vector<Node*>& merge_nodes, int input_edge,
                     Graph* body);

  // Adds all the input edges to `if_node` corresponding to the arguments.
  Status AddInputEdges(const CondArgNodes& cond_arg_nodes,
                       const Edge* predicate_edge, Node* if_node);

  // Adds all output edges from the `if_node`.
  Status AddOutputEdges(const std::vector<Node*>& outputs, Node* if_node);

  // Returns the switch clusters of graph_ in postorder. Dead switch nodes are
  // skipped and removed from the graph.
  StatusOr<std::vector<SwitchCluster>> DeterminePredicateSwitchOrder();

  // Update the state for destination based on the state of source and the node
  // being updated.
  Status Join(const ForwardFlowNode& src_state, const Node* dst,
              ForwardFlowNode* dst_state);

  // Ensure that all nodes in the branch_map are dominated by the switch
  // nodes. Returns nodes that are not dominated by the switches but are a
  // control dependency of a node in the cond, and remove such control
  // dependencies.
  StatusOr<std::vector<Node*>> EnsureDominanceAndReturnNonDominatedControlNodes(
      const std::unordered_map<Node*, ForwardFlowNode>& branch_map,
      const std::vector<Node*>& switches);

  // Validates that the frontier of nodes for the conditional
  // section are as expected.
  Status ValidateFrontier(
      const std::unordered_map<Node*, ForwardFlowNode>& branch_map,
      const std::unordered_set<Node*>& frontier);

  FunctionLibraryDefinition* library_;
  Graph* graph_;
  bool dump_graphs_;
};

bool IsDeadSwitch(const Node* node) {
  for (const Edge* e : node->out_edges()) {
    const Node* dst = e->dst();
    if (!dst->IsIdentity()) {
      return false;
    }
    for (const Edge* ee : dst->out_edges()) {
      if (!ee->IsControlEdge() || !ee->dst()->IsSink()) {
        return false;
      }
    }
  }
  return true;
}

string FunctionalizeCond::Branch_Name(FunctionalizeCond::Branch b) {
  const string branch_name[FunctionalizeCond::kNumBranchTypes + 1] = {
      "else", "then", "both", "neither", "count"};
  return branch_name[b];
}

Status FunctionalizeCond::ValidateFrontier(
    const std::unordered_map<Node*, FunctionalizeCond::ForwardFlowNode>&
        branch_map,
    const std::unordered_set<Node*>& frontier) {
  std::unordered_set<const Node*> pending[kNumBranchTypes];
  for (Node* n : frontier) {
    pending[branch_map.at(n).branch].insert(n);
  }
  TF_RET_CHECK(pending[kNeither].empty()) << NodesToString(pending[kNeither]);
  for (const Node* n : pending[kBoth]) {
    TF_RET_CHECK(IsMerge(n)) << n->DebugString();
    // Merge nodes may be in then or else branch too
  }
  int index = (pending[kThenBranch].size() <= pending[kElseBranch].size())
                  ? kThenBranch
                  : kElseBranch;
  int other = 1 - index;
  for (const Node* n : pending[index]) {
    if (pending[other].find(n) != pending[other].end()) {
      return errors::Internal(
          "Node (", n->DebugString().c_str(),
          ") in both Else and Then branch should be in Both.");
    }
  }
  // An empty frontier indicates a dead switch. Above we attempt to remove dead
  // switch nodes, but not all are removed so don't treat it as an error yet.
  // TODO(jpienaar): Find out why dead switch nodes remain.
  // if (pending[kBoth].empty() && pending[kThenBranch].empty() &&
  //     pending[kElseBranch].empty()) {
  //   return errors::Internal("Unexpected empty frontier for switch nodes");
  // }
  return Status::OK();
}

Status FunctionalizeCond::Join(const ForwardFlowNode& src_state,
                               const Node* dst, ForwardFlowNode* dst_state) {
  TF_RET_CHECK(dst_state->branch != Branch::kBoth &&
               dst_state->branch != Branch::kNumBranchTypes)
      << "Unexpected/Invalid branch type: Merging "
      << Branch_Name(src_state.branch) << " with "
      << Branch_Name(dst_state->branch);
  if (dst_state->branch == Branch::kNeither) {
    dst_state->branch = src_state.branch;
  } else if (src_state.branch != dst_state->branch &&
             src_state.branch != Branch::kNeither) {
    if (IsMerge(dst)) {
      dst_state->branch = Branch::kBoth;
    } else {
      return errors::Internal("Illegal merge:\n", src_state.ToString(),
                              " with ", dst_state->ToString(), " for\n",
                              dst->DebugString());
    }
  }
  ++dst_state->count;
  return Status::OK();
}

StatusOr<std::vector<FunctionalizeCond::SwitchCluster>>
FunctionalizeCond::DeterminePredicateSwitchOrder() {
  struct Cluster {
    bool operator==(const Cluster& other) const {
      return representative == other.representative;
    }
    int representative = -1;
  };

  // Perform a DFS over the graph and
  // * Determine the reverse topological order of the nodes (there should be no
  //   cycles at this point so the post-order numbering corresponds to the
  //   reverse topological sorting);
  // * Identify dead switches;
  // * Initialize the cluster's representative;
  std::vector<UnionFind<Cluster>> clusters(graph_->num_node_ids());
  std::vector<Node*> dead_switches;
  std::vector<Node*> switch_order;
  std::vector<Node*> rev_topo_sorted_nodes;
  DFS(*graph_, nullptr, [&](Node* n) {
    clusters[n->id()].Get().representative = n->id();
    if (IsSwitch(n)) {
      if (IsDeadSwitch(n)) {
        dead_switches.push_back(n);
      } else {
        rev_topo_sorted_nodes.push_back(n);
        switch_order.push_back(n);
      }
    } else if (n->IsOp()) {
      // Exclude src and sink nodes from further consideration.
      rev_topo_sorted_nodes.push_back(n);
    }
  });

  std::vector<SwitchCluster> switch_clusters;
  // Return early if there are no switches in the graph.
  if (switch_order.empty()) {
    return switch_clusters;
  }

  // Remove all dead switch nodes.
  for (Node* n : dead_switches) {
    VLOG(2) << "Removing dead switch: " << n->DebugString();
    graph_->RemoveNode(n);
  }

  // Identify switch nodes that are part of the same control flow context by
  // considering the operands of operations: an operation is part of the same
  // control context as its operands unless the operation is a switch. Control
  // dependencies are considered part of the same control flow context if the
  // switch depth is the same (see comment below).

  // entry_cluster records the input cluster to a switch node. This is used when
  // merging with a merge node where the dst's cluster is merged with the entry
  // cluster of the merge node's cluster (which corresponds to a switch cluster
  // and so has an entry cluster).
  std::unordered_map<int, UnionFind<Cluster>*> entry_cluster;

  // Returns the output cluster of a node. Where the output cluster is cluster
  // where the output of the node is used. For non-merge nodes this is simply
  // the cluster they are part of, while for merge nodes it is the entry cluster
  // of the cluster they are part of (this will correspond to the entry node of
  // a switch node that dominates the merge).
  auto find_output_cluster = [&](Node* n) {
    UnionFind<Cluster>* cluster = &clusters[n->id()];
    if (!IsMerge(n)) return cluster;
    auto it = entry_cluster.find(clusters[n->id()].Get().representative);
    // If the cluster is not found in the entry_cluster map then an
    // instruction not dominated by a switch node has been merged into the
    // cluster of the merge. This indicates a failure of the clustering.
    CHECK(it != entry_cluster.end())
        << "Unable to find entry for n=" << n->id() << " ("
        << cluster->Get().representative << ")";
    return it->second;
  };

  // TODO(jpienaar): This could be combined with DetermineBranchMapAndFrontier.
  std::vector<int> switch_depth(graph_->num_node_ids());
  for (auto it = rev_topo_sorted_nodes.rbegin();
       it != rev_topo_sorted_nodes.rend(); ++it) {
    Node* n = *it;

    // Compute switch depth.
    int new_switch_depth = 0;
    for (const Edge* e : n->in_edges()) {
      Node* src = e->src();
      new_switch_depth = std::max(
          new_switch_depth, switch_depth[src->id()] - (IsMerge(src) ? 1 : 0));
    }
    switch_depth[n->id()] = new_switch_depth + (IsSwitch(n) ? 1 : 0);

    // Only merge the input operands of a switch. The switch's clustering itself
    // is determined by the interaction of the switch's outputs.
    if (IsSwitch(n)) {
      Node* input;
      TF_CHECK_OK(n->input_node(0, &input));
      entry_cluster[n->id()] = find_output_cluster(input);
      UnionFind<Cluster>* cluster = entry_cluster[n->id()];
      int cluster_depth = switch_depth[cluster->Get().representative];
      // Merge the inputs of the switch node with one another. This results in
      // predicates and control input residing in the same cluster.
      for (const Edge* e : n->in_edges()) {
        // Only consider the data inputs to the Switch node.
        if (e->IsControlEdge()) continue;

        Node* src = e->src();
        UnionFind<Cluster>* src_cluster = find_output_cluster(src);
        int src_cluster_depth = switch_depth[src_cluster->Get().representative];
        if (cluster_depth != src_cluster_depth) {
          return errors::InvalidArgument(
              "Unable to functionalize control flow in graph: Switch ('",
              n->name(), "') has operands ('", input->name(), "' and '",
              src->name(), "') that have different switch depths (",
              cluster_depth, " != ", src_cluster_depth, ")");
        }
        cluster->Merge(src_cluster);
      }
      continue;
    }

    for (const Edge* e : n->in_edges()) {
      Node* src = e->src();
      if (!src->IsOp()) continue;
      UnionFind<Cluster>* cluster = find_output_cluster(src);
      // Merge a node with its data operands and with its control operands if
      // the src and dst are in the same ControlContext. The ControlContext is
      // not explicitly available here, and instead the switch depth is used as
      // a proxy here. Due to the invariant that control edges can only be from
      // a containing scope to an inner scope or from the inner scope to its
      // containing scope (for exit nodes), the switch depth will only match if
      // the src and dst are in the same ControlContext. Control edges between
      // ControlContexts are handled during the extraction.
      int src_id = cluster->Get().representative;
      int src_depth = switch_depth[src_id];
      if (!e->IsControlEdge() || new_switch_depth == src_depth) {
        if (src_depth != new_switch_depth) {
          // TODO(b/77601805) remove this when outside_compilation supports
          // control flow.
          if (str_util::StrContains(src->name(), "outside_compilation") ||
              str_util::StrContains(n->name(), "outside_compilation")) {
            return errors::InvalidArgument(
                "outside_compilation is not yet supported within TensorFlow "
                "control flow constructs b/77601805");
          }
          return errors::InvalidArgument(
              "Unable to functionalize control flow in graph: Operand ('",
              src->name(), "') and operator ('", n->name(),
              "') have different switch depths (", src_depth,
              " != ", new_switch_depth, ")");
        }
        cluster->Merge(&clusters[n->id()]);
      }
    }
  }

  if (dump_graphs_) {
    // Mark the switch cluster each node is part of.
    for (Node* n : graph_->nodes()) {
      n->ClearAttr("_XlaFunctionalizeSwitchGroup");
      n->AddAttr("_XlaFunctionalizeSwitchGroup",
                 clusters[n->id()].Get().representative);
    }
    LOG(INFO) << "FunctionalizeControlFlow (with_clusters): "
              << dump_graph::DumpGraphToFile("functionalize_clustered", *graph_,
                                             library_);
  }

  // Verify all the nodes of a cluster are at the same depth.
  std::unordered_map<int, std::pair<int, Node*>> cluster_to_depth_node;
  for (Node* n : graph_->nodes()) {
    int depth = switch_depth[n->id()];
    int cluster_rep = clusters[n->id()].Get().representative;
    auto it = cluster_to_depth_node.find(cluster_rep);
    if (it == cluster_to_depth_node.end()) {
      cluster_to_depth_node[cluster_rep] = std::make_pair(depth, n);
    } else {
      if (it->second.first != depth) {
        return errors::Internal(
            "Illegal clustering created, mismatch in depths:", "\n\t",
            n->DebugString(), "(", clusters[n->id()].Get().representative,
            ") at depth=", depth, " vs\n\t", it->second.second->DebugString(),
            "(", clusters[n->id()].Get().representative, ") at depth ",
            it->second.first);
      }
    }
  }

  struct Hash {
    size_t operator()(const std::pair<Node*, Cluster>& item) const {
      return Hash64Combine(hash<Node*>()(item.first),
                           std::hash<int>()(item.second.representative));
    }
  };

  // Merge Switch nodes with common predicate.
  std::unordered_map<std::pair<Node*, Cluster>, int, Hash> predicate_index;
  // The nodes in switch_order are in reverse topological order, but the
  // clustered switches need not be (i.e., when considered as a cluster one
  // element of a cluster may be later in the topological order than another
  // node whose cluster is later in the topological order of clustered
  // switches).
  for (auto it = switch_order.rbegin(); it != switch_order.rend(); ++it) {
    const Edge* pred_edge;
    TF_CHECK_OK((*it)->input_edge(1, &pred_edge));
    // The predicate can be preceded by a identity node. Look through identity
    // nodes to predicate.
    while (pred_edge->src()->IsIdentity()) {
      TF_CHECK_OK(pred_edge->src()->input_edge(0, &pred_edge));
    }
    auto repr = std::make_pair(pred_edge->src(), clusters[(*it)->id()].Get());
    if (predicate_index.find(repr) == predicate_index.end()) {
      predicate_index[repr] = switch_clusters.size();
      switch_clusters.emplace_back(pred_edge);
      // Generate a name by concatenating with the cluster representative as
      // there could be multiple switch clusters with the same predicate.
      switch_clusters[predicate_index[repr]].name = strings::StrCat(
          pred_edge->src()->name(), "_", repr.second.representative, "_If");
    }
    switch_clusters[predicate_index[repr]].switches.push_back(*it);
  }

  return switch_clusters;
}

StatusOr<std::vector<Node*>>
FunctionalizeCond::EnsureDominanceAndReturnNonDominatedControlNodes(
    const std::unordered_map<Node*, ForwardFlowNode>& branch_map,
    const std::vector<Node*>& switches) {
  std::vector<Node*> old_control_nodes;
  for (const auto& kv : branch_map) {
    if (kv.second.count != kv.first->in_edges().size()) {
      std::vector<const Edge*> delete_edges;
      for (const Edge* in : kv.first->in_edges()) {
        auto it = branch_map.find(in->src());
        if (it == branch_map.end()) {
          if (in->IsControlEdge()) {
            old_control_nodes.push_back(in->src());
            delete_edges.push_back(in);
          } else {
            if (IsSwitch(in->src())) {
              if (std::find(switches.begin(), switches.end(), in->src()) ==
                  switches.end()) {
                return errors::Internal(
                    "Unexpected switch node found during flow forward: ",
                    in->src()->DebugString());
              }
              continue;
            }
            return errors::InvalidArgument(
                "Value ", kv.first->name(), "'s input, ", in->src()->name(),
                ", is not dominated by switch nodes ", NodesToString(switches));
          }
        }
      }
      // Remove control edges from nodes that are not dominated by the switch
      // nodes. New control dependencies will be added between these nodes and
      // the XlaIf node inserted.
      for (const Edge* e : delete_edges) {
        graph_->RemoveEdge(e);
      }
    }
  }
  return old_control_nodes;
}

StatusOr<
    std::pair<std::unordered_map<Node*, FunctionalizeCond::ForwardFlowNode>,
              std::unordered_set<Node*>>>
FunctionalizeCond::DetermineBranchMapAndFrontier(
    const SwitchCluster& switch_cluster) {
  std::unordered_map<Node*, ForwardFlowNode> branch_map;
  std::unordered_set<Node*> frontier;
  std::vector<Node*> stack = switch_cluster.switches;
  std::vector<bool> visited(graph_->num_node_ids(), false);
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();

    if (visited[n->id()]) {
      continue;
    }
    visited[n->id()] = true;

    // Propagate branch state along each edge of a switch node.
    bool sink_only = true;
    for (const Edge* e : n->out_edges()) {
      Node* out = e->dst();
      if (!out->IsOp()) {
        continue;
      }
      sink_only = false;
      // Propagate branch information.
      ForwardFlowNode& ffn = branch_map[out];
      if (IsSwitch(n)) {
        int index = e->IsControlEdge() ? Branch::kNeither : e->src_output();
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            Join(ForwardFlowNode(Branch(index)), out, &ffn), " when joining ",
            e->DebugString());
      } else {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(Join(branch_map[n], out, &ffn),
                                        " when joining ", e->DebugString());
      }
      if (IsMerge(out)) {
        if (out->in_edges().size() == ffn.count) {
          frontier.insert(out);
        }
      } else if (!visited[out->id()]) {
        stack.push_back(out);
      }
    }
    if (sink_only) {
      if (!IsIdentity(n)) {
        VLOG(1) << "Feeding into sink: " << n->DebugString();
      }
    }
  }

  if (dump_graphs_) {
    for (const auto& kv : branch_map) {
      // Append attribute to the graph if running with logging to make the
      // changes clearer in the visualization.
      kv.first->AddAttr("_XlaFunctionalizeBranch",
                        Branch_Name(kv.second.branch));
    }
  }
  return std::make_pair(std::move(branch_map), std::move(frontier));
}

Status FunctionalizeCond::FunctionalizeInternal() {
  TF_ASSIGN_OR_RETURN(std::vector<SwitchCluster> predicate_switch_order,
                      DeterminePredicateSwitchOrder());

  // Iterate from innermost set of clustered switches to outermost, replacing
  // matching switch->merge subgraphs with single XlaIf nodes.
  for (auto it = predicate_switch_order.rbegin();
       it != predicate_switch_order.rend(); ++it) {
    auto& ps = *it;
    VLOG(3) << "Flow down from: " << ps.ToString();

    std::unordered_map<Node*, ForwardFlowNode> branch_map;
    std::unordered_set<Node*> frontier;
    TF_ASSIGN_OR_RETURN(std::tie(branch_map, frontier),
                        DetermineBranchMapAndFrontier(ps));

    if (dump_graphs_)
      LOG(INFO) << "FunctionalizeControlFlow (before XlaIf conversion): "
                << dump_graph::DumpGraphToFile("functionalize_bc", *graph_,
                                               library_);
    TF_RETURN_IF_ERROR(ValidateFrontier(branch_map, frontier));

    struct Hash {
      size_t operator()(const std::pair<Node*, int>& item) const {
        return Hash64Combine(hash<Node*>()(item.first),
                             std::hash<int>()(item.second));
      }
    };

    // Sort the merge and switch nodes using NodeCmp. The switch-nodes are
    // further grouped (post sorting) by input to the switch node as in the
    // functionalized form each input will be passed in only once. This grouping
    // should retain the sorted order.
    CondArgNodes cond_arg_nodes;
    std::sort(ps.switches.begin(), ps.switches.end(), NodeCmp());
    std::unordered_map<std::pair<Node*, int>, int, Hash> input_index;
    for (Node* switch_node : ps.switches) {
      const Edge* e;
      TF_RETURN_IF_ERROR(switch_node->input_edge(0, &e));
      std::pair<Node*, int> key = std::make_pair(e->src(), e->src_output());
      if (input_index.find(key) == input_index.end()) {
        input_index[key] = cond_arg_nodes.size();
        cond_arg_nodes.emplace_back(key.first, key.second);
      }
      cond_arg_nodes.at(input_index.at(key)).switches.push_back(switch_node);
    }
    std::vector<Node*> merge_nodes(frontier.begin(), frontier.end());
    std::sort(merge_nodes.begin(), merge_nodes.end(), NodeCmp());

    TF_ASSIGN_OR_RETURN(std::vector<Node*> old_control_nodes,
                        EnsureDominanceAndReturnNonDominatedControlNodes(
                            branch_map, ps.switches));

    TF_ASSIGN_OR_RETURN(Node * if_node,
                        ConvertToXlaIf(cond_arg_nodes, ps, merge_nodes));
    for (Node* old : old_control_nodes) {
      graph_->AddControlEdge(old, if_node);
    }

    for (auto& del_kv : branch_map) {
      graph_->RemoveNode(del_kv.first);
    }
    for (auto& kv : cond_arg_nodes) {
      for (Node* node : kv.switches) {
        graph_->RemoveNode(node);
      }
    }
    if (dump_graphs_)
      LOG(INFO) << "FunctionalizeControlFlow (after XlaIf conversion): "
                << dump_graph::DumpGraphToFile("functionalize_ac", *graph_,
                                               library_);
  }
  return Status::OK();
}

StatusOr<Node*> FunctionalizeCond::BuildAndAddXlaIfOp(
    const CondArgNodes& cond_arg_nodes, const SwitchCluster& switch_cluster,
    const std::vector<Node*>& merge_nodes) {
  VLOG(2) << "Build if op for " << switch_cluster.name;

  NodeDef if_def;
  // Create a new If node using the name of the merge node.
  NodeDefBuilder builder(switch_cluster.name, "XlaIf");
  string branch[] = {"else_branch", "then_branch"};
  for (int i = 0; i < 2; ++i) {
    static std::atomic<int64> sequence_num(0LL);
    int64 id = ++sequence_num;

    NameAttrList body_name;
    body_name.set_name(
        strings::StrCat("_functionalize_if_", branch[i], "_", id));
    auto body = xla::MakeUnique<Graph>(graph_->op_registry());
    TF_RETURN_IF_ERROR(ExtractBody(cond_arg_nodes, switch_cluster.switches,
                                   merge_nodes, i, body.get()));
    VLOG(3) << "Body " << branch[i] << ": " << DebugString(body.get());
    FunctionDef body_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*body, body_name.name(), &body_fdef));
    TF_RETURN_IF_ERROR(library_->AddFunctionDef(body_fdef));
    builder.Attr(branch[i], body_name);
  }

  // Build input type.
  std::vector<NodeDefBuilder::NodeOut> inputs;
  DataTypeVector in_arg_types;
  for (auto& kv : cond_arg_nodes) {
    bool inserted = false;
    for (const Node* arg : kv.switches) {
      const Edge* in_edge;
      TF_RETURN_IF_ERROR(arg->input_edge(0, &in_edge));
      if (in_edge->IsControlEdge()) {
        builder.ControlInput(in_edge->src()->name());
      } else {
        if (!inserted) {
          DataType dtype = arg->input_type(0);
          inputs.emplace_back(NodeDefBuilder::NodeOut(
              in_edge->src()->name(), in_edge->src_output(), dtype));
          in_arg_types.push_back(dtype);
          inserted = true;
        }
      }
    }
  }
  builder.Attr("Tin", in_arg_types);

  // Build output type.
  DataTypeVector out_type;
  for (const Node* merge : merge_nodes) {
    DataType dtype = merge->output_type(0);
    out_type.push_back(dtype);
  }
  builder.Attr("Tout", out_type);

  builder.Attr("Tcond", DT_BOOL);
  builder.Device(switch_cluster.predicate_edge->src()->assigned_device_name());
  // Conditional should be the first input ...
  builder.Input(NodeDefBuilder::NodeOut(
      switch_cluster.predicate_edge->src()->name(),
      switch_cluster.predicate_edge->src_output(),
      switch_cluster.predicate_edge->src()->output_type(0)));
  // ... followed by the other inputs.
  builder.Input(inputs);

  TF_RETURN_IF_ERROR(builder.Finalize(&if_def));
  TF_ASSIGN_OR_RETURN(Node * if_node, AddNode(if_def, graph_));
  return if_node;
}

Status FunctionalizeCond::ExtractBody(const CondArgNodes& cond_arg_nodes,
                                      const std::vector<Node*>& switches,
                                      const std::vector<Node*>& merge_nodes,
                                      int input_edge, Graph* body) {
  VLOG(2) << "ExtractBody for " << NodesToString(merge_nodes) << " along edge "
          << input_edge;
  std::vector<bool> squash_src_outputs(graph_->num_node_ids(), false);
  std::vector<Node*> node_map(graph_->num_node_ids(), nullptr);
  int arg_count = 0;
  for (auto& kv : cond_arg_nodes) {
    Node* arg_node = nullptr;
    for (const auto* arg : kv.switches) {
      DataType dtype = arg->input_type(0);
      if (arg_node == nullptr) {
        TF_ASSIGN_OR_RETURN(arg_node, BuildArgNode(body, dtype, arg_count++));
      }
      node_map.at(arg->id()) = arg_node;
      squash_src_outputs.at(arg->id()) = true;
    }
  }

  std::vector<Node*> stack;
  stack.reserve(merge_nodes.size());
  for (int j = 0; j < merge_nodes.size(); ++j) {
    Node* node = merge_nodes[j];
    TF_ASSIGN_OR_RETURN(node_map.at(node->id()),
                        BuildRetvalNode(body, node->output_type(0),
                                        /*index=*/j));
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(node->input_edge(input_edge, &in_edge));
    Node* in = in_edge->src();
    if (node_map.at(in->id()) == nullptr) {
      node_map.at(in->id()) = body->CopyNode(in);
    }

    if (std::find(switches.begin(), switches.end(), in) == switches.end()) {
      body->AddEdge(node_map.at(in->id()), in_edge->src_output(),
                    node_map.at(node->id()), 0);
    } else {
      body->AddEdge(node_map.at(in->id()), 0, node_map.at(node->id()), 0);
      // Don't include input nodes that are already just returned in stack.
      continue;
    }
    stack.push_back(in);
  }

  return CopySubgraph(*graph_, nullptr, stack, squash_src_outputs, &node_map,
                      body);
}

Status FunctionalizeCond::AddInputEdges(const CondArgNodes& cond_arg_nodes,
                                        const Edge* predicate_edge,
                                        Node* if_node) {
  VLOG(3) << "AddInputEdges for " << if_node->name();
  int index = 0;
  graph_->AddEdge(predicate_edge->src(), predicate_edge->src_output(), if_node,
                  index++);
  for (auto& arg : cond_arg_nodes) {
    if (arg.src_output == Graph::kControlSlot) {
      graph_->AddControlEdge(arg.src, if_node);
    } else {
      graph_->AddEdge(arg.src, arg.src_output, if_node, index++);
    }
  }
  return Status::OK();
}

Status FunctionalizeCond::AddOutputEdges(const std::vector<Node*>& outputs,
                                         Node* if_node) {
  VLOG(3) << "AddOutputEdges for " << if_node->name();
  for (int i = 0; i < outputs.size(); ++i) {
    Node* node = outputs[i];
    std::vector<const Edge*> edges(node->out_edges().begin(),
                                   node->out_edges().end());
    for (const Edge* edge : edges) {
      Node* dst = edge->dst();
      int dst_input = edge->dst_input();

      if (edge->src_output() > 0) {
        return errors::Unimplemented("Output of index (", edge->src_output(),
                                     ") of merge node ", node->name());
      }

      int src_output =
          dst_input == Graph::kControlSlot ? Graph::kControlSlot : i;
      graph_->RemoveEdge(edge);
      graph_->AddEdge(if_node, src_output, dst, dst_input);
    }
  }
  return Status::OK();
}

StatusOr<Node*> FunctionalizeCond::ConvertToXlaIf(
    const CondArgNodes& cond_arg_nodes, const SwitchCluster& switch_cluster,
    const std::vector<Node*>& merge_nodes) {
  VLOG(1) << "ConvertToXlaIf for " << switch_cluster.ToString() << " -> "
          << NodesToString(merge_nodes);

  // Extract bodies and builds a If operator.
  TF_ASSIGN_OR_RETURN(
      Node * if_node,
      BuildAndAddXlaIfOp(cond_arg_nodes, switch_cluster, merge_nodes));
  TF_RETURN_IF_ERROR(
      AddInputEdges(cond_arg_nodes, switch_cluster.predicate_edge, if_node));
  TF_RETURN_IF_ERROR(AddOutputEdges(merge_nodes, if_node));

  return if_node;
}

Status FunctionalizeCond::Functionalize(Graph* graph,
                                        FunctionLibraryDefinition* library) {
  VLOG(1) << "FunctionalizeCond::Functionalize";
  FunctionalizeCond fc(graph, library, /*dump_graphs=*/VLOG_IS_ON(2));
  return fc.FunctionalizeInternal();
}

}  // namespace

// Transformation that converts TensorFlow's graph control flow constructs into
// functional equivalents.
Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library) {
  return FunctionalizeControlFlow(/*lookup_library=*/nullptr, graph, library);
}

Status FunctionalizeControlFlow(const FunctionLibraryDefinition* lookup_library,
                                Graph* graph,
                                FunctionLibraryDefinition* library) {
  VLOG(2) << "FunctionalizeControlFlow (initial): "
          << dump_graph::DumpGraphToFile("functionalize_initial", *graph,
                                         library);

  // Note: BuildControlFlowInfo() requires that the graph's source node is
  // connected to all source nodes in the graph. Many graphs violate this
  // invariant.
  std::vector<ControlFlowInfo> cf_info;
  std::vector<string> unreachable_nodes;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      BuildControlFlowInfo(graph, &cf_info, &unreachable_nodes),
      "FunctionalizeControlFlow failed");
  if (!unreachable_nodes.empty()) {
    return errors::InvalidArgument(
        "The following nodes are unreachable from the source in the graph: ",
        tensorflow::str_util::Join(unreachable_nodes, ", "));
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

  // FunctionalizeControlFlow is invoked for every function, so the loops's
  // bodies and conditionals that were extracted into functions will be handled
  // in successive invocations.
  TF_RETURN_IF_ERROR(FunctionalizeCond::Functionalize(graph, library));

  VLOG(2) << "FunctionalizeControlFlow (final): "
          << dump_graph::DumpGraphToFile("functionalize_final", *graph,
                                         library);
  return Status::OK();
}

}  // namespace tensorflow
