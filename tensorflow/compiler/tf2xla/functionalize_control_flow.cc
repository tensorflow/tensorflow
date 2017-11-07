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

#include "tensorflow/compiler/jit/graph_to_functiondef.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/optional.h"

namespace tensorflow {

namespace {

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

xla::StatusOr<Node*> AddNode(const NodeDef& node_def, Graph* graph) {
  Status status;
  Node* inserted_node = graph->AddNode(node_def, &status);
  if (!status.ok()) {
    return status;
  }
  return inserted_node;
}

xla::StatusOr<Node*> BuildArgNode(Graph* graph, DataType type, int index) {
  NodeDef arg_def;
  NodeDefBuilder builder(strings::StrCat(kArgOp, index), kArgOp);
  builder.Attr("T", type);
  builder.Attr("index", index);
  TF_RETURN_IF_ERROR(builder.Finalize(&arg_def));
  return AddNode(arg_def, graph);
}

xla::StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index) {
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

Status FunctionalizeLoop(Graph* graph, Frame* frame,
                         FunctionLibraryDefinition* library) {
  VLOG(2) << "Frame " << frame->name << " before: "
          << dump_graph::DumpGraphToFile("functionalize_before", *graph);

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

  // Order the arguments so that:
  // a) resource variables are last, and
  // b) sort lexicographically by name (for deterministic output).
  std::sort(frame->args.begin(), frame->args.end(),
            [](const Arg& a, const Arg& b) {
              bool a_is_resource = (a.enter->input_type(0) == DT_RESOURCE);
              bool b_is_resource = (b.enter->input_type(0) == DT_RESOURCE);
              return std::tie(a_is_resource, a.enter->name()) <
                     std::tie(b_is_resource, b.enter->name());
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
      for (const Edge* edge : arg.switch_node->out_edges()) {
        if (edge->src_output() == 0 && IsExit(edge->dst())) {
          if (arg.exit != nullptr) {
            return errors::InvalidArgument("Duplicate Exit successors to ",
                                           arg.switch_node->name());
          }
          arg.exit = edge->dst();
        } else if (StringPiece(edge->dst()->type_string()) == "Identity") {
          TF_RETURN_IF_ERROR(
              SetNodeShardingFromNeighbors(edge->dst(), /*out_edges=*/true));
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
          << dump_graph::DumpGraphToFile("loop_condition", *cond_graph)
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
          << dump_graph::DumpGraphToFile("functionalize_after", *graph);

  return Status::OK();
}

class FunctionalizeCond {
 public:
  // Identifies the connected parts of the tf.Cond.
  struct ClusterHandle {
    explicit ClusterHandle(int representative = -1)
        : representative(representative) {}

    bool operator==(const ClusterHandle& other) const {
      return representative == other.representative;
    }

    bool operator!=(const ClusterHandle& other) const {
      return !(*this == other);
    }

    bool operator<(const ClusterHandle& other) const {
      return representative < other.representative;
    }

    bool operator>(const ClusterHandle& other) const {
      return representative > other.representative;
    }

    string ToString() const {
      return strings::StrCat("Cluster_", representative);
    }

    // Vector of UnionFind<ClusterHandle> indexable by ClusterHandle and Node*.
    struct Vector {
      explicit Vector(size_t size) : clusters(size) {}

      UnionFind<ClusterHandle>& at(const ClusterHandle& cluster) {
        return clusters.at(cluster.representative);
      }

      UnionFind<ClusterHandle>& at(const Node* node) {
        return clusters.at(node->id());
      }

      UnionFind<ClusterHandle>& operator[](const Node* node) {
        return clusters.at(node->id());
      }

      size_t size() const { return clusters.size(); }

      void resize(size_t count) { return clusters.resize(count); }

     private:
      std::vector<UnionFind<ClusterHandle>> clusters;
    };

   private:
    int representative;
  };

  // Represents a node in the clustered graph consisting of switch_nodes,
  // merge_nodes as well as the edges into and out of this node to other
  // Clusters. Each Cluster corresponds to a ClusterHandle and has a
  // corresponding representative.
  struct Cluster {
    std::unordered_set<Node*> switch_nodes;
    std::unordered_set<Node*> merge_nodes;
    std::unordered_set<Cluster*> in_nodes;
    std::unordered_set<Cluster*> out_nodes;

    // A member of the ClusterHandle corresponding to this Cluster.
    ClusterHandle representative;
    bool visited = false;
  };

  // Represent the clustered graph as map from cluster representative to
  // Cluster.
  using ClusteredGraph = std::map<ClusterHandle, Cluster>;

  // The arguments and condition of a XlaIf. The arguments are ordered by node
  // id in the original graph.
  struct CondArgs {
    struct CondCmp {
      bool operator()(const Node* lhs, const Node* rhs) const {
        bool lhs_is_resource =
            lhs->num_inputs() > 0 ? (lhs->input_type(0) == DT_RESOURCE) : false;
        bool rhs_is_resource =
            rhs->num_inputs() > 0 ? (rhs->input_type(0) == DT_RESOURCE) : false;
        return std::tie(lhs_is_resource, lhs->name()) <
               std::tie(rhs_is_resource, rhs->name());
      }
    };
    Node* conditional = nullptr;
    std::set<Node*, CondCmp> args;
  };

  static Status Functionalize(Graph* graph, FunctionLibraryDefinition* library);

 private:
  FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library)
      : clusters_(graph->num_node_ids()), library_(library), graph_(graph) {}

  // Returns a vector of Merge nodes from the clustered graph where the nodes
  // are sorted by the number of switch nodes minus number of merge nodes
  // from a root of the clustered graph to the given Merge node, with ties
  // broken by the representative of the Cluster.
  std::vector<std::pair<int, Cluster*>> SortedMergeNodes();

  // Returns whether the graph has no conditionals.
  bool NoConditionals() const { return merge_nodes_.empty(); }

  // Construct the clustered graph by creating nodes for each cluster and the
  // connections between the clusters. Switch and Merge nodes partition
  // clusters, so iterate over those. Note: a Cluster may have neither a
  // Merge or Switch but will have an in/out edge from a Cluster that has.
  void CreateClusters();

  // Creates the clustered graph by identifying all the edges between different
  // clusters and collecting all switch and merge nodes that correspond to a
  // cluster.
  void CreateClusteredGraph();

  // If `from` and `to` correspond to different clusters, then merge the nodes
  // in the clustered graph corresponding to `from` and `to`.
  //
  // If `remove_from_graph` is specified then the `from` node is also removed
  // from the clustered graph post contracting the edge.
  void ContractEdge(Cluster* from, Cluster* to, bool remove_from_graph = false);

  // Converts a Merge node to a XlaIf. This encapsulates the process of
  // extracting the bodies needed for the then and else branch, creates a XlaIf
  // node, removing the nodes of the branches from the graph and replacing the
  // merge node with a XlaIf.
  Status ConvertMergeToXlaIf(Cluster* merge_cluster);

  // Removes a Switch cluster feeding directly into a Merge cluster by removing
  // the Switch and Merge nodes and collapsing into a single cluster.
  Status RemoveTrivialMerge(Cluster* merge_cluster);

  // Returns the switch cluster corresponding to the merge node. This function
  // only returns the switch cluster in the simple case where we have a switch
  // node is the entry of a diamond corresponding to a conditional:
  //
  //           Switch
  //          /      \
  //     Branch      Branch
  //          \      /
  //        merge_cluster
  //
  // Note: either of the branches may be empty. The case where both branches are
  // empty is handled by RemoveTrivialMerge.
  gtl::optional<Cluster*> GetSwitchCluster(const Cluster& merge_cluster);

  // Determines the arguments needed as input to the Merge cluster originating
  // from the Switch cluster.
  xla::StatusOr<CondArgs> DetermineCondArgs(const Cluster& merge_cluster,
                                            const Cluster& switch_cluster);

  // Builds a XlaIfOp to replace the Merge node with.
  xla::StatusOr<Node*> BuildAndAddXlaIfOp(const CondArgs& cond_args,
                                          const Cluster& merge_cluster,
                                          const std::vector<Node*>& outputs);

  // Extracts a function body corresponding to the given input edge of the merge
  // node.
  Status ExtractBody(const CondArgs& cond_args, const Cluster& merge_cluster,
                     const std::vector<Node*>& outputs, int input_edge,
                     Graph* body);

  // Adds all the input edges to `if_node` corresponding to the arguments.
  Status AddInputEdges(const CondArgs& cond_args, Node* if_node);

  // Adds all output edges from the `if_node`.
  Status AddOutputEdges(const std::vector<Node*>& outputs, Node* if_node);

  // Removes all nodes from the graph that are part of cluster.
  void RemoveClusterNodes(Cluster* cluster);

  // Removes all argument nodes that are unused.
  template <class T>
  void RemoveUnusedArgs(const T& args);

  // Removes all Merge nodes in merge_cluster.
  void RemoveMergeNodes(Cluster* merge_cluster);

  // Returns the representative member of the corresponding cluster.
  ClusterHandle Representative(const Node* node) {
    return clusters_.at(node).Get();
  }

  ClusteredGraph clustered_graph_;
  ClusterHandle::Vector clusters_;
  std::unordered_set<Node*> merge_nodes_;
  std::unordered_set<Node*> switch_nodes_;
  FunctionLibraryDefinition* library_;
  Graph* graph_;
};

std::ostream& operator<<(std::ostream& os,
                         const FunctionalizeCond::ClusterHandle& c) {
  os << c.ToString();
  return os;
}

// Returns a dot representation of the clustered graph showing the connections
// between the nodes and the nodes in each cluster.
string DebugString(const Graph& graph,
                   FunctionalizeCond::ClusterHandle::Vector* clusters) {
  string ret = "digraph {\ncompound=true;labeljust=\"r\";ranksep=0.24\n";
  std::map<FunctionalizeCond::ClusterHandle, string> subgraphs;
  for (Node* n : graph.nodes()) {
    if (n->IsOp()) {
      strings::StrAppend(&subgraphs[clusters->at(n).Get()], n->id(),
                         " [label=\"", n->name(), "\"];\n");
    }
  }
  for (auto kv : subgraphs) {
    strings::StrAppend(&ret, "subgraph cluster_", kv.first.ToString(), " {\n",
                       "style=filled; color=lightgrey;", "label = \"",
                       kv.first.ToString(), "\";\n", kv.second, "}\n");
  }
  for (Node* n : graph.nodes()) {
    if (!n->IsOp()) {
      continue;
    }
    for (Node* in : n->in_nodes()) {
      if (in->IsOp()) {
        strings::StrAppend(&ret, in->id(), " -> ", n->id(), ";\n");
      }
    }
  }
  return strings::StrCat(ret, "}");
}

string DebugString(const FunctionalizeCond::ClusteredGraph& clustered_graph) {
  string ret = "digraph {\ncompound=true;labeljust=\"r\";\n";
  auto name = [](const FunctionalizeCond::Cluster& cluster) {
    return cluster.representative.ToString();
  };
  for (auto kv : clustered_graph) {
    strings::StrAppend(&ret, kv.first.ToString(), " [label=\"", name(kv.second),
                       " (", kv.second.switch_nodes.size(), ", ",
                       kv.second.merge_nodes.size(), ")\"];\n");
  }
  for (auto kv : clustered_graph) {
    for (auto in : kv.second.in_nodes) {
      strings::StrAppend(&ret, name(*in), " -> ", name(kv.second), ";\n");
    }
  }
  return strings::StrCat(ret, "}");
}

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

void FunctionalizeCond::CreateClusters() {
  for (Node* node : graph_->nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    if (IsSwitch(node)) {
      switch_nodes_.insert(node);
    } else if (IsMerge(node)) {
      merge_nodes_.insert(node);
    }
    ClusterHandle& cluster = clusters_.at(node).Get();
    cluster = ClusterHandle(node->id());
  }

  // If there are no Merge nodes, then terminate.
  if (merge_nodes_.empty()) {
    return;
  }

  // Remove all dead Switch nodes.
  RemoveUnusedArgs(switch_nodes_);

  // All parent_'s are still nullptr so clusters_ may still be resized. Resize
  // conservatively assuming all merge nodes become XlaIf nodes.
  clusters_.resize(clusters_.size() + merge_nodes_.size());

  // Merge a cluster with its input, unless the input is a Switch node or
  // the node is a Merge node.
  for (const Node* node : graph_->nodes()) {
    if (IsMerge(node) || IsSwitch(node) || !node->IsOp()) {
      continue;
    }
    for (const Node* in : node->in_nodes()) {
      if (in->IsOp() && !IsSwitch(in) && !IsMerge(in)) {
        clusters_.at(node).Merge(&clusters_.at(in));
      }
    }
  }
}

void FunctionalizeCond::ContractEdge(Cluster* from, Cluster* to,
                                     bool remove_from_graph) {
  VLOG(3) << "ContractEdge from = " << from->representative
          << " to = " << to->representative;
  if (from->representative == to->representative) {
    return;
  }
  to->merge_nodes.insert(from->merge_nodes.begin(), from->merge_nodes.end());
  from->merge_nodes.clear();
  to->switch_nodes.insert(from->switch_nodes.begin(), from->switch_nodes.end());
  from->switch_nodes.clear();

  for (Cluster* from_out : from->out_nodes) {
    from_out->in_nodes.erase(from);
    if (from_out->representative != to->representative) {
      from_out->in_nodes.insert(to);
      to->out_nodes.insert(from_out);
    }
  }
  from->out_nodes.clear();

  for (Cluster* from_in : from->in_nodes) {
    from_in->out_nodes.erase(from);
    if (from_in->representative != to->representative) {
      from_in->out_nodes.insert(to);
      to->in_nodes.insert(from_in);
    }
  }
  from->in_nodes.clear();

  to->in_nodes.erase(from);
  to->out_nodes.erase(from);
  clusters_.at(to->representative).Merge(&clusters_.at(from->representative));
  from->visited = true;

  if (remove_from_graph) {
    clustered_graph_.erase(from->representative);
  }
}

void FunctionalizeCond::CreateClusteredGraph() {
  auto update_cluster_for_node = [this](Node* node) -> Cluster& {
    ClusterHandle repr = Representative(node);
    Cluster& cluster_node = clustered_graph_[repr];
    cluster_node.representative = repr;
    for (const Node* in : node->in_nodes()) {
      ClusterHandle other_repr = Representative(in);
      // Skip source, sink and internal edges.
      if (!in->IsOp() || other_repr == repr) {
        continue;
      }
      Cluster& cluster_node_in = clustered_graph_[other_repr];
      cluster_node.in_nodes.insert(&cluster_node_in);
      cluster_node_in.out_nodes.insert(&cluster_node);
      cluster_node_in.representative = other_repr;
    }
    for (const Node* out : node->out_nodes()) {
      ClusterHandle other_repr = Representative(out);
      // Skip source, sink and internal edges.
      if (!out->IsOp() || other_repr == repr) {
        continue;
      }
      Cluster& cluster_node_out = clustered_graph_[other_repr];
      cluster_node.out_nodes.insert(&cluster_node_out);
      cluster_node_out.in_nodes.insert(&cluster_node);
      cluster_node_out.representative = other_repr;
    }
    return cluster_node;
  };
  for (Node* node : switch_nodes_) {
    update_cluster_for_node(node).switch_nodes.insert(node);
  }
  for (Node* node : merge_nodes_) {
    update_cluster_for_node(node).merge_nodes.insert(node);
  }

  // Merge Switch nodes with common predicate.
  std::unordered_map<Node*, std::vector<Node*>> predicate_to_switch;
  for (Node* node : switch_nodes_) {
    Node* tmp;
    TF_CHECK_OK(node->input_node(1, &tmp));
    predicate_to_switch[tmp].push_back(node);
  }
  for (auto kv : predicate_to_switch) {
    Cluster& first = clustered_graph_.at(Representative(kv.second.front()));
    for (Node* switch_node : kv.second) {
      ClusterHandle handle = Representative(switch_node);
      Cluster& cluster = clustered_graph_.at(handle);
      ContractEdge(&cluster, &first, /*remove_from_graph=*/true);
    }
  }

  // Merge Merge nodes with common input together.
  for (Node* node : merge_nodes_) {
    Cluster& cluster = clustered_graph_.at(Representative(node));
    for (const Node* in : node->in_nodes()) {
      if (!in->IsOp()) {
        continue;
      }
      Cluster& cluster_node_in = clustered_graph_.at(Representative(in));
      // ContractEdge can modify out_nodes of cluster_node_in, so traverse
      // over out_nodes assuming it does.
      for (auto it = cluster_node_in.out_nodes.begin();
           it != cluster_node_in.out_nodes.end();) {
        if (!(*it)->merge_nodes.empty()) {
          ContractEdge(*it++, &cluster, /*remove_from_graph=*/true);
        } else {
          ++it;
        }
      }
    }
  }

  VLOG(3) << "Graph with clusters: " << DebugString(*graph_, &clusters_);
  VLOG(3) << "ClusteredGraph: " << DebugString(clustered_graph_);
}

gtl::optional<FunctionalizeCond::Cluster*> FunctionalizeCond::GetSwitchCluster(
    const Cluster& merge_cluster) {
  VLOG(3) << "GetSwitchCluster for " << merge_cluster.representative;
  gtl::optional<Cluster*> switch_cluster;
  if (merge_cluster.in_nodes.size() > 2) {
    return gtl::nullopt;
  }
  for (Cluster* in : merge_cluster.in_nodes) {
    Cluster* cluster = in;
    if (in->switch_nodes.empty()) {
      if (in->in_nodes.size() != 1) {
        return gtl::nullopt;
      }
      // There is only a single `in` cluster.
      cluster = *in->in_nodes.begin();
    }
    if (cluster->switch_nodes.empty()) {
      return gtl::nullopt;
    }

    if (switch_cluster.has_value() && *switch_cluster != cluster) {
      return gtl::nullopt;
    } else {
      switch_cluster = cluster;
    }
  }
  return switch_cluster;
}

xla::StatusOr<FunctionalizeCond::CondArgs> FunctionalizeCond::DetermineCondArgs(
    const Cluster& merge_cluster, const Cluster& switch_cluster) {
  VLOG(2) << "DetermineCondArgs for " << merge_cluster.representative
          << " with switch cluster " << switch_cluster.representative;
  CondArgs ret;
  auto feeds_into_branch_cluster = [&](Node* switch_cluster) {
    for (Node* out : switch_cluster->out_nodes()) {
      ClusterHandle repr = Representative(out);
      if (repr == merge_cluster.representative) {
        return true;
      }
      for (Cluster* in : merge_cluster.in_nodes) {
        if (repr == in->representative) {
          return true;
        }
      }
    }
    return false;
  };
  for (Node* switch_cluster_node : switch_cluster.switch_nodes) {
    if (!feeds_into_branch_cluster(switch_cluster_node)) {
      continue;
    }

    Node* tmp;
    TF_RETURN_IF_ERROR(switch_cluster_node->input_node(1, &tmp));
    if (ret.conditional == nullptr) {
      ret.conditional = tmp;
    } else if (ret.conditional != tmp) {
      return errors::Unimplemented(
          "Switch statements with different conditionals cannot be "
          "converted into functional conditional.");
    }
    ret.args.insert(switch_cluster_node);
  }
  return ret;
}

xla::StatusOr<Node*> FunctionalizeCond::BuildAndAddXlaIfOp(
    const CondArgs& cond_args, const Cluster& merge_cluster,
    const std::vector<Node*>& outputs) {
  VLOG(2) << "Build if op for " << NodesToString(merge_cluster.merge_nodes)
          << " with input " << NodesToString(cond_args.args);

  NodeDef if_def;
  // Create a new If node using the name of the merge node.
  NodeDefBuilder builder(
      strings::StrCat((*merge_cluster.merge_nodes.begin())->name(), "_If"),
      "XlaIf");
  string branch[] = {"else_branch", "then_branch"};
  for (int i = 0; i < 2; ++i) {
    static std::atomic<int64> sequence_num(0LL);
    int64 id = ++sequence_num;

    NameAttrList body_name;
    body_name.set_name(
        strings::StrCat("_functionalize_if_", branch[i], "_", id));
    auto body = xla::MakeUnique<Graph>(graph_->op_registry());
    TF_RETURN_IF_ERROR(
        ExtractBody(cond_args, merge_cluster, outputs, i, body.get()));
    VLOG(3) << "Body " << branch[i] << ": " << DebugString(body.get());
    FunctionDef body_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*body, body_name.name(), &body_fdef));
    TF_RETURN_IF_ERROR(library_->AddFunctionDef(body_fdef));
    builder.Attr(branch[i], body_name);
  }

  // Build input type.
  std::vector<NodeDefBuilder::NodeOut> inputs;
  DataTypeVector in_arg_types;
  for (const Node* arg : cond_args.args) {
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(arg->input_edge(0, &in_edge));
    if (in_edge->IsControlEdge()) {
      builder.ControlInput(in_edge->src()->name());
    } else {
      DataType dtype = arg->input_type(0);
      inputs.emplace_back(NodeDefBuilder::NodeOut(
          in_edge->src()->name(), in_edge->src_output(), dtype));
      in_arg_types.push_back(dtype);
    }
  }
  builder.Attr("Tin", in_arg_types);

  // Build output type.
  DataTypeVector out_type;
  for (const Node* merge : merge_cluster.merge_nodes) {
    DataType dtype = merge->output_type(0);
    out_type.push_back(dtype);
  }
  builder.Attr("Tout", out_type);

  builder.Attr("Tcond", DT_BOOL);
  builder.Device(cond_args.conditional->assigned_device_name());
  // Conditional should be the first input ...
  builder.Input(NodeDefBuilder::NodeOut(cond_args.conditional->name(), 0,
                                        cond_args.conditional->output_type(0)));
  // ... followed by the other inputs.
  builder.Input(inputs);

  TF_RETURN_IF_ERROR(builder.Finalize(&if_def));
  TF_ASSIGN_OR_RETURN(Node * if_node, AddNode(if_def, graph_));
  return if_node;
}

void FunctionalizeCond::RemoveClusterNodes(Cluster* cluster) {
  VLOG(3) << "RemoveClusterNodes for " << cluster->representative;
  ClusterHandle repr = cluster->representative;
  std::deque<Node*> to_delete;
  for (Node* node : graph_->nodes()) {
    if (Representative(node) == repr) {
      to_delete.push_back(node);
    }
  }
  for (Node* n : to_delete) {
    graph_->RemoveNode(n);
  }
}

template <class T>
void FunctionalizeCond::RemoveUnusedArgs(const T& args) {
  VLOG(2) << "RemoveUnusedArgs among: " << NodesToString(args);

  std::deque<Node*> to_delete;
  for (Node* arg : args) {
    if (IsDeadSwitch(arg)) {
      to_delete.push_back(arg);
      for (Node* n : arg->out_nodes()) {
        to_delete.push_back(n);
      }
    }
  }
  for (Node* n : to_delete) {
    switch_nodes_.erase(n);
    auto it = clustered_graph_.find(Representative(n));
    if (it != clustered_graph_.end()) {
      it->second.switch_nodes.erase(n);
    }
    graph_->RemoveNode(n);
  }
}

Status FunctionalizeCond::ExtractBody(const CondArgs& cond_args,
                                      const Cluster& merge_cluster,
                                      const std::vector<Node*>& outputs,
                                      int input_edge, Graph* body) {
  VLOG(2) << "ExtractBody for " << merge_cluster.representative
          << " along edge " << input_edge;
  std::vector<bool> squash_src_outputs(graph_->num_node_ids(), false);
  std::vector<Node*> node_map(graph_->num_node_ids(), nullptr);
  int arg_count = 0;
  for (const auto* arg : cond_args.args) {
    DataType dtype = arg->input_type(0);
    TF_ASSIGN_OR_RETURN(Node * arg_node,
                        BuildArgNode(body, dtype, arg_count++));
    node_map.at(arg->id()) = arg_node;
    squash_src_outputs.at(arg->id()) = true;
  }

  std::vector<Node*> stack;
  stack.reserve(outputs.size());
  for (int j = 0; j < outputs.size(); ++j) {
    Node* node = outputs[j];
    TF_ASSIGN_OR_RETURN(node_map.at(node->id()),
                        BuildRetvalNode(body, node->output_type(0),
                                        /*index=*/j));
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(node->input_edge(input_edge, &in_edge));
    Node* in = in_edge->src();
    if (node_map.at(in->id()) == nullptr) {
      node_map.at(in->id()) = body->CopyNode(in);
    }

    if (cond_args.args.find(in) == cond_args.args.end()) {
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

Status FunctionalizeCond::AddInputEdges(const CondArgs& cond_args,
                                        Node* if_node) {
  VLOG(3) << "AddInputEdges for " << if_node->name();
  int i = 0;
  graph_->AddEdge(cond_args.conditional, 0, if_node, i++);
  for (const Node* arg : cond_args.args) {
    const Edge* in_edge;
    TF_RETURN_IF_ERROR(arg->input_edge(0, &in_edge));
    if (in_edge->IsControlEdge()) {
      graph_->AddControlEdge(in_edge->src(), if_node);
    } else {
      graph_->AddEdge(in_edge->src(), in_edge->src_output(), if_node, i++);
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
      graph_->RemoveEdge(edge);

      int src_output =
          dst_input == Graph::kControlSlot ? Graph::kControlSlot : i;
      graph_->AddEdge(if_node, src_output, dst, dst_input);
    }
  }
  return Status::OK();
}

void FunctionalizeCond::RemoveMergeNodes(Cluster* merge_cluster) {
  VLOG(3) << "RemoveMergeNodes for " << merge_cluster->representative;
  // Remove all merge nodes now dead post extraction of If.
  for (auto it = merge_cluster->merge_nodes.begin();
       it != merge_cluster->merge_nodes.end();) {
    Node* node = *it;
    graph_->RemoveNode(node);
    merge_cluster->merge_nodes.erase(*it++);
  }
}

Status FunctionalizeCond::RemoveTrivialMerge(Cluster* merge_cluster) {
  Cluster* switch_cluster = *merge_cluster->in_nodes.begin();
  if (switch_cluster->switch_nodes.empty()) {
    return errors::FailedPrecondition(
        "Not a trivial merge: no Switch node feeding into Merge node");
  }

  for (auto it = merge_cluster->merge_nodes.begin();
       it != merge_cluster->merge_nodes.end();) {
    // We have the following structure:
    //   Op -> Switch -> Merge -> Consumer
    // and we want to transform it to:
    //   Op -> Consumer
    Node* merge_node = *it;
    Node* switch_node;
    const Edge* in = nullptr;
    TF_RETURN_IF_ERROR(merge_node->input_node(0, &switch_node));
    TF_RETURN_IF_ERROR(switch_node->input_edge(0, &in));
    for (auto out : merge_node->out_edges()) {
      int src_output = out->dst_input() == Graph::kControlSlot
                           ? Graph::kControlSlot
                           : in->src_output();
      graph_->AddEdge(in->src(), src_output, out->dst(), out->dst_input());
    }
    graph_->RemoveNode(*it++);
  }
  RemoveUnusedArgs(switch_cluster->switch_nodes);

  return Status::OK();
}

Status FunctionalizeCond::ConvertMergeToXlaIf(Cluster* merge_cluster) {
  VLOG(1) << "ConvertMergeToXlaIf for " << merge_cluster->representative;
  gtl::optional<Cluster*> switch_cluster = GetSwitchCluster(*merge_cluster);
  if (!switch_cluster.has_value()) {
    return errors::FailedPrecondition(
        "Merge cluster was not part of a simple conditional in the clustered "
        "graph. Graph nodes in merge cluster ",
        NodesToString(merge_cluster->merge_nodes));
  }
  TF_ASSIGN_OR_RETURN(auto cond_args,
                      DetermineCondArgs(*merge_cluster, **switch_cluster));

  // Sort the outputs by ID to produce more stable output.
  std::vector<Node*> outputs(merge_cluster->merge_nodes.begin(),
                             merge_cluster->merge_nodes.end());
  std::sort(outputs.begin(), outputs.end(), CondArgs::CondCmp());

  // Extract bodies and builds a If operator.
  TF_ASSIGN_OR_RETURN(Node * if_node,
                      BuildAndAddXlaIfOp(cond_args, *merge_cluster, outputs));
  TF_RETURN_IF_ERROR(AddInputEdges(cond_args, if_node));
  TF_RETURN_IF_ERROR(AddOutputEdges(outputs, if_node));

  // Remove the old nodes from the graph_ and contract the edges of the
  // clustered graph.
  for (auto in : merge_cluster->in_nodes) {
    if (in != *switch_cluster) {
      RemoveClusterNodes(in);
    }
  }
  RemoveMergeNodes(merge_cluster);
  RemoveUnusedArgs(cond_args.args);
  auto in_nodes = merge_cluster->in_nodes;
  for (auto it = in_nodes.begin(); it != in_nodes.end();) {
    ContractEdge(*it++, merge_cluster);
  }
  ContractEdge(*switch_cluster, merge_cluster);
  clusters_[if_node].Get() = ClusterHandle(merge_cluster->representative);

  return Status::OK();
}

std::vector<std::pair<int, FunctionalizeCond::Cluster*>>
FunctionalizeCond::SortedMergeNodes() {
  VLOG(2) << "ProcessClusteredGraph";
  std::stack<std::pair<int, Cluster*>> stack;
  for (auto& c : clustered_graph_) {
    if (c.second.in_nodes.empty()) {
      stack.push({0, &c.second});
    }
  }

  // Perform a depth-first traversal of the clustered graph computing the
  // switch-merge depth.
  std::vector<std::pair<int, Cluster*>> queue;
  std::unordered_set<Cluster*> visited;
  while (!stack.empty()) {
    Cluster* n = stack.top().second;
    size_t depth = stack.top().first;
    stack.pop();

    auto inserted = visited.insert(n);
    if (!inserted.second) {
      continue;
    }

    size_t new_depth = depth;
    if (!n->merge_nodes.empty()) {
      queue.emplace_back(depth, n);
      --new_depth;
    }
    if (!n->switch_nodes.empty()) {
      ++new_depth;
    }
    for (Cluster* e : n->out_nodes) {
      stack.emplace(new_depth, e);
    }
  }

  // Sort in reverse order of switch-merge depth with ties broken by the
  // ClusterHandle.
  std::sort(queue.begin(), queue.end(),
            [](const std::pair<int, Cluster*>& lhs,
               const std::pair<int, Cluster*>& rhs) {
              return std::tie(lhs.first, lhs.second->representative) >
                     std::tie(rhs.first, rhs.second->representative);
            });

  return queue;
}

Status FunctionalizeCond::Functionalize(Graph* graph,
                                        FunctionLibraryDefinition* library) {
  VLOG(1) << "FunctionalizeCond::Functionalize";
  FunctionalizeCond fc(graph, library);
  fc.CreateClusters();
  if (fc.NoConditionals()) {
    return Status::OK();
  }
  fc.CreateClusteredGraph();

  auto queue = fc.SortedMergeNodes();
  for (auto it = queue.begin(); it != queue.end();) {
    Cluster* merge_cluster = (*it).second;
    ++it;
    if (merge_cluster->in_nodes.size() == 1) {
      TF_RETURN_IF_ERROR(fc.RemoveTrivialMerge(merge_cluster));
    } else {
      TF_RETURN_IF_ERROR(fc.ConvertMergeToXlaIf(merge_cluster));
    }

    // Contract newly Merge free merge_cluster with incoming nodes without
    // Switch or Merge nodes.
    std::vector<Cluster*> in_nodes(merge_cluster->in_nodes.begin(),
                                   merge_cluster->in_nodes.end());
    for (auto in : in_nodes) {
      if (in->merge_nodes.empty() && in->switch_nodes.empty()) {
        fc.ContractEdge(in, merge_cluster);
      }
    }
  }

  if (!fc.switch_nodes_.empty()) {
    return errors::Internal(
        "Failed to functionalize control flow with Switch nodes remaining: ",
        NodesToString(fc.switch_nodes_));
  }
  return Status::OK();
}

}  // namespace

// Transformation that converts Tensorflow's graph control flow constructs into
// functional equivalents.
Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library) {
  VLOG(2) << "FunctionalizeControlFlow (initial): "
          << dump_graph::DumpGraphToFile("functionalize_initial", *graph);
  // Note: BuildControlFlowInfo() requires that the graph's source node is
  // connected to all source nodes in the graph. Many graphs violate this
  // invariant.
  std::vector<ControlFlowInfo> cf_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &cf_info));

  // Builds Frames, indexed by name.
  std::unordered_map<string, Frame> frames;
  for (Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];

    VLOG(2) << "node: " << node->name() << " frame_name: " << cf.frame_name
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
    } else if (frame.parent != parent) {
      return errors::InvalidArgument("Mismatched parent frames for ",
                                     cf.frame->id(), ": ", parent->name, " vs ",
                                     frame.parent->name);
    }

    if (IsEnter(node)) {
      Arg arg;
      arg.enter = node;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg.enter->attrs(), "is_constant",
                                     &arg.is_loop_invariant));
      frame.args.push_back(arg);
    } else if (IsLoopCond(node)) {
      if (frame.loop_cond) {
        return errors::InvalidArgument(
            "Loop ", cf.frame_name,
            " has more than one LoopCond node: ", node->name(), " and ",
            frame.loop_cond->name());
      }
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

    TF_RETURN_IF_ERROR(FunctionalizeLoop(graph, frame, library));

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
          << dump_graph::DumpGraphToFile("functionalize_final", *graph);
  return Status::OK();
}

}  // namespace tensorflow
