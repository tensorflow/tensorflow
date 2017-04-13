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

#include "tensorflow/core/graph/execution_condition.h"

#include <vector>
#include <unordered_set>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

// Insert identity node to every data input edge of n.
Status InsertIdentities(Graph* g, Node* n) {
  const string& name = n->name();
  const string& device = n->def().device();

  std::vector<const Edge*> data_edges;
  data_edges.reserve(n->in_edges().size());
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge()) {
      data_edges.push_back(e);
    }
  }

  for (const Edge* e : data_edges) {
    Node* id_node;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name, "/Identity")), "Identity")
        .Input(e->src(), e->src_output())
        .Device(device)
        .Finalize(g, &id_node));

    int port = e->dst_input();
    g->RemoveEdge(e);
    g->AddEdge(id_node, 0, n, port);
  }

  return Status::OK();
}

// Create the indicator for n.
Status MergeConditions(const std::vector<Node*>& indicators,
                       Graph* g, Node* n, Node** indicator) {
  //LOG(INFO) << "MergeConditions " << n->name();

  // Collect indicators of output nodes.
  std::unordered_set<Node*> output_indicators;
  for (Node* o : n->out_nodes()) {
    Node* i = indicators[o->id()];
    if (!i) {
      // One of the output nodes is executed unconditionally,
      // so this node is also executed unconditionally.
      *indicator = nullptr;
      return Status::OK();
    }
    output_indicators.insert(i);
  }

  if (output_indicators.empty()) {
    // This node has no output node so it must be sink.
    DCHECK(n->IsSink());
    *indicator = nullptr;
    return Status::OK();
  }

  if (output_indicators.size() == 1) {
    // All of the output nodes has the same indicator.
    *indicator = *output_indicators.begin();
    return Status::OK();
  }

  // The output nodes has differenct indicators. We have to create a merge
  // node followed by an identity node as the indicator of this node.
  std::vector<NodeBuilder::NodeOut> inputs; // Inputs for the merge node.
  inputs.reserve(output_indicators.size());
  for (Node* i : output_indicators) {
    inputs.emplace_back(i, 0);
  }
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(n->name(), "/Merge")), "Merge")
      .Input(inputs)
      .Device(n->def().device())
      .Finalize(g, indicator));
  (*indicator)->set_assigned_device_name(n->assigned_device_name());

  return Status::OK();
}

// Create indicators for each branch of mux node.
Status CreateIndicators(Graph* g, Node* n, std::vector<Node*>& indicators) {
  DCHECK(n->IsMux());

  int N = n->num_inputs() - 1;
  const string& device = n->def().device();
  const string& assigned_device = n->assigned_device_name();
  const string& name = n->name();

  Node* const_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Const")), "Const")
      .Attr("value", Tensor())
      .Attr("dtype", DT_FLOAT)
      .Device(device)
      .Finalize(g, &const_node));
  const_node->set_assigned_device_name(assigned_device);

  Node* demux_node;
  const Edge* e;
  TF_RETURN_IF_ERROR(n->input_edge(0, &e));
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Demux")), "Demux")
      .Input(e->src(), e->src_output())
      .Input(const_node, 0)
      .Attr("N", N)
      .Device(device)
      .Finalize(g, &demux_node));
  demux_node->set_assigned_device_name(assigned_device);

  indicators.clear();
  indicators.reserve(N);
  for (int i = 0; i < N; i++) {
    Node* ind;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name, "/Identity")), "Identity")
        .Input(demux_node, i)
        .Device(device)
        .Finalize(g, &ind));
    ind->set_assigned_device_name(assigned_device);
    indicators.push_back(ind);
  }

  return Status::OK();
}

// Replace mux node with merge node.
Status ReplaceWithMerge(Graph* g, Node* n) {
  DCHECK(n->IsMux());

  // Backup input edges.
  std::vector<NodeBuilder::NodeOut> data_inputs;
  data_inputs.reserve(n->num_inputs());
  std::vector<Node*> control_inputs;
  control_inputs.reserve(n->in_edges().size());
  for (const Edge* e : n->in_edges()) {
    switch (e->dst_input()) {
      case Graph::kControlSlot:
        control_inputs.push_back(e->src());
        break;
      case 0:
        // The first input of mux is index.
        break;
      default:
        DCHECK(e->src()->IsIdentity());
        data_inputs.emplace_back(e->src(), e->src_output());
    }
  }

  struct NodeAndPort {
    Node* node;
    int port;
  };

  // Backup output edges.
  std::vector<NodeAndPort> data_outputs;
  std::vector<Node*> control_outputs;
  for (const Edge* e : n->out_edges()) {
    if (e->IsControlEdge()) {
      control_outputs.push_back(e->dst());
    } else {
      data_outputs.push_back({e->dst(), e->dst_input()});
    }
  }

  // Create a merge node with the same inputs as the mux node.
  Node* merge_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(n->name(), "/Merge")), "Merge")
      .Input(data_inputs)
      .ControlInputs(control_inputs)
      .Device(n->def().device())
      .Finalize(g, &merge_node));
  merge_node->set_assigned_device_name(n->assigned_device_name());

  g->RemoveNode(n);

  // Restore output edges.
  for (NodeAndPort node_and_port : data_outputs) {
    g->AddEdge(merge_node, 0, node_and_port.node, node_and_port.port);
  }
  for (Node* n : control_outputs) {
    g->AddControlEdge(merge_node, n);
  }

  return Status::OK();
}

/*
string DumpGraph(Graph* g) {
  GraphDef gd;
  g->ToGraphDef(&gd);
  string ret;
  protobuf::TextFormat::PrintToString(gd, &ret);
  return ret;
}

void AddInputToNodeDef(NodeDef& dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst.add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst.add_input(src_name.data(), src_name.size());
  } else {
    dst.add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

template<typename T> NodeDef NodeToDef(T node) {
  NodeDef node_def = node->def();

  // Use the node's assigned device, if any, instead of the device requested
  // in the NodeDef.
  if (!node->assigned_device_name().empty()) {
    node_def.set_device(node->assigned_device_name());
  }

  // Get the inputs for this Node.  We make sure control inputs are
  // after data inputs, as required by GraphDef.
  std::vector<const Edge*> inputs(node->num_inputs(), nullptr);
  for (const Edge* edge : node->in_edges()) {
    if (edge->IsControlEdge()) {
      inputs.push_back(edge);
    } else {
      CHECK(inputs[edge->dst_input()] == nullptr)
          << "Edge " << edge->src()->DebugString() << ":"
          << edge->dst()->DebugString() << " with dst_input "
          << edge->dst_input() << " and had pre-existing input edge "
          << inputs[edge->dst_input()]->src()->DebugString() << ":"
          << inputs[edge->dst_input()]->dst()->DebugString();

      inputs[edge->dst_input()] = edge;
    }
  }
  node_def.clear_input();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Edge* edge = inputs[i];
    if (edge == nullptr) {
      node_def.add_input(node->def().input(i));
    } else {
      const Node* src = edge->src();
      if (!src->IsOp()) continue;
      AddInputToNodeDef(node_def, src->name(), edge->src_output());
    }
  }

  return node_def;
}

template<typename T> string DumpNode(T n) {
  string ret;
  protobuf::TextFormat::PrintToString(NodeToDef(n), &ret);
  return ret;
}

template<typename T> string DumpNodes(T nodes) {
  GraphDef graph_def;
  for (auto n : nodes) {
    NodeDef* node_def = graph_def.add_node();
    *node_def = NodeToDef(n);
  }

  string ret;
  protobuf::TextFormat::PrintToString(graph_def, &ret);
  return ret;
}
*/

} // namespace

Status AddExecutionConditions(
    Graph* g, const std::unordered_set<Node*>& target_nodes) {

  /*
  LOG(INFO) << "AddExecutionConditions";
  LOG(INFO) << DumpGraph(g);
  LOG(INFO) << "target_nodes";
  LOG(INFO) << DumpNodes(target_nodes);
  */

  // Connect target nodes to sink so that we can treat sink as
  // the only target node.
  for (Node* n : target_nodes) {
    // This is just an equivalent of python for-else loop.
    [=]() {
      for (const Edge* e : n->out_edges()) {
        if (e->dst()->IsSink()) return;
      }
      g->AddControlEdge(n, g->sink_node());
    }();
  }

  // Make reverse execution order of the graph and collect mux nodes.
  std::vector<Node*> iter_order;
  std::vector<Node*> mux_nodes;
  {
    iter_order.reserve(g->num_nodes());

    // How many output nodes of a node has not been visited yet.
    // A node is ready when its count reaches zeros.
    std::vector<int> pending_counts(g->num_node_ids());
    for (Node* n : g->nodes()) {
      // Initalize pending counts to # of output nodes.
      pending_counts[n->id()] = n->out_edges().size();
    }

    // Sink is the only node with no output.
    iter_order.push_back(g->sink_node());

    // We reuse iter_order[i ... end] as a queue for ready nodes.
    for (int i = 0; i < iter_order.size(); i++) {
      Node* n = iter_order[i];

      if (n->IsMux()) {
        mux_nodes.push_back(n);
      }

      // Decrement pending count for each of its inputs.
      for (Node* m : n->in_nodes()) {
        pending_counts[m->id()]--;
        if (pending_counts[m->id()] == 0 && !m->IsSource()) {
          // Exclude source node, otherwise we might end up giving it
          // execution conditions.
          iter_order.push_back(m);
        }
      }
    }

    DCHECK_EQ(g->num_nodes() - 1, iter_order.size());
  }

  /*
  LOG(INFO) << "iter_order" << iter_order.size();
  LOG(INFO) << "\n" << DumpNodes(iter_order);
  */

  // The execution conditions are different for each branch of mux,
  // so we insert an identity node to store the condition for each branch.
  // We also insert identity to the first input (index) so that
  // CreateIndicators() does not add any output node to the first input nodes
  // (output nodes are added to the inserted identity instead),
  // otherwise MergeCondition() for the first input node would break.
  for (Node* n : mux_nodes) {
    TF_RETURN_IF_ERROR(InsertIdentities(g, n));
  }

  // We will add some nodes to the graph later. Backup the current node list.
  std::vector<Node*> non_indicator_nodes;
  non_indicator_nodes.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    non_indicator_nodes.push_back(n);
  }

  /*
  LOG(INFO) << "InsertIdentities";
  LOG(INFO) << "\n" << DumpGraph(g);
  */

  // A map from nodes to their indicators. An indicator is an identity
  // node that is executed iff the execution condition of a node is true.
  std::vector<Node*> indicators(g->num_node_ids(), nullptr);
  // This is used to receive return value from CreateIndicators(). We put
  // it outside the loop to reduce memory allocation.
  std::vector<Node*> ind_for_mux;
  // Traverse the graph in reverse execution order and calculate condition
  // for each node.
  for (Node* n : iter_order) {
    if (!n->IsSink()) {
      // Compute the execution condition of this node from the conditions of
      // its output nodes.
      TF_RETURN_IF_ERROR(
        MergeConditions(indicators, g, n, &indicators[n->id()]));
    }

    if (n->IsMux()) {
      // Create indicators for each branch of mux.
      ind_for_mux.clear();
      TF_RETURN_IF_ERROR(CreateIndicators(g, n, ind_for_mux));

      // Input nodes to mux are identities created by InsertIdentities().
      // These nodes are not in iter_order, so we have to set their indicators
      // here.
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;

        int port = e->dst_input();
        if (port == 0) {
          // The first input (index) has the same execution condition as mux.
          indicators[e->src()->id()] = indicators[n->id()];
        } else {
          indicators[e->src()->id()] = ind_for_mux[port - 1];
        }
      }
    }
  }

  /*{
    string s;
    for (Node* i : indicators) {
      if (i) {
        s.append(i->name()).append(" ");
      } else {
        s.append("nullptr ");
      }
    }
    LOG(INFO) << "indicators";
    LOG(INFO) << s;
    LOG(INFO) << "\n" << DumpGraph(g);
  }*/

  // Add control edge to each node from its indicator, if necessary.
  for (Node* n : non_indicator_nodes) {
    Node* i = indicators[n->id()];

    if (!i || n->IsMux()) {
      // Mux nodes will be replace with merge later. These merge
      // nodes do not need additional control edge either.
      continue;
    }

    [=, &indicators]() {
      for (const Edge* e : n->in_edges()) {
        if (indicators[e->src()->id()] == i) {
          // An input node has the same execution condition, so we can skip
          // adding control edge from the indicator.
          return;
        }
      }
      g->AddControlEdge(i, n);
    }();
  }

  /*
  LOG(INFO) << "AddControlEdge";
  LOG(INFO) << "\n" << DumpGraph(g);
  */

  for (Node* n : mux_nodes) {

    /*
    LOG(INFO) << n->name();
    {
      const Node* m;
      string s;
      for (int i = 0; i < 3; i++) {
        n->input_node(i, &m);
        s.append(m->name()).append(" ");
      }
      LOG(INFO) << s;
    }
    */

    TF_RETURN_IF_ERROR(ReplaceWithMerge(g, n));
  }

  //LOG(INFO) << "AddExecutionConditions done.";
  return Status::OK();
}

Status AddExecutionConditions(
    Graph* g, const std::unordered_set<const Node*>& target_nodes) {
  std::unordered_set<Node*> mutable_target_nodes;
  mutable_target_nodes.reserve(target_nodes.size());

  for (const Node* n : target_nodes) {
    mutable_target_nodes.insert(g->FindNodeId(n->id()));
  }

  return AddExecutionConditions(g, mutable_target_nodes);
}

}  // namespace tensorflow
