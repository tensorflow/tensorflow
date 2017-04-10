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

#include <deque>
#include <vector>
#include <unordered_set>
#include <string>

#include "tensorflow/core/graph/cond_rewrite.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

struct NodeInfo {
  // An identity node that is executed iff this node should be executed.
  Node* condition = nullptr;
  // When an input node has the same condition as this node, there is no need
  // to add control edge from condition to this node.
  bool satisfied = false;
};

struct NodeAndPort {
  Node* node;
  int port;
};

Status InsertIdentities(Graph* g, Node* cond) {
  const string& name = cond->name();
  const string& device = cond->def().device();

  for (int i = 0; i < 3; i++) {
    const Edge* e;
    TF_RETURN_IF_ERROR(cond->input_edge(i, &e));

    Node* n;
    TF_RETURN_IF_ERROR(
      NodeBuilder(g->NewName(strings::StrCat(name, "/Identity")), "Identity")
        .Input(e->src(), e->src_output())
        .Device(device)
        .Finalize(g, &n));
    g->RemoveEdge(e);
    g->AddEdge(n, 0, cond, i);
  }

  return Status::OK();
}

Status MergeConditions(std::vector<NodeInfo>& node_info,
                       Graph* g, Node* n, Node** merged) {
  LOG(INFO) << "MergeConditions " << n->name();

  auto ret = [=, &node_info](Node* m) {
    *merged = m;
    node_info[n->id()].condition = m;
    return Status::OK();
  };

  std::unordered_set<Node*> conditions;
  for (Node* m : n->out_nodes()) {
    Node* c = node_info[m->id()].condition;
    if (!c) {
      return ret(nullptr);
    }
    conditions.insert(c);
  }

  if (conditions.empty()) {
    if (n->IsSink()) {
      return ret(nullptr);
    }
    return errors::Internal("Encountered node without consumer.", n->name());
  }

  if (conditions.size() == 1) {
    for (Node* m : n->out_nodes()) {
      node_info[m->id()].satisfied = true;
    }
    return ret(*conditions.begin());
  }

  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(conditions.size());
  for (Node* c : conditions) {
    inputs.emplace_back(c, 0);
  }

  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(n->name(), "/Merge")), "Merge")
      .Input(inputs)
      .Device(n->def().device())
      .Finalize(g, merged));

  return ret(*merged);
}

Status CreateBranchConditions(Graph* g, Node* cond,
                              Node** condition_true,
                              Node** condition_false) {
  const string& device = cond->def().device();
  const string& name = cond->name();

  const Edge* e;
  TF_RETURN_IF_ERROR(cond->input_edge(0, &e));

  Node* const_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Const")), "Const")
      .Attr("value", Tensor())
      .Attr("dtype", DT_FLOAT)
      .Device(device)
      .Finalize(g, &const_node));

  Node* switch_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Switch")), "Switch")
      .Input(const_node, 0)
      .Input(e->src(), e->src_output())
      .Device(device)
      .Finalize(g, &switch_node));

  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Identity")), "Identity")
      .Input(switch_node, 0)
      .Device(device)
      .Finalize(g, condition_false));

  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(name, "/Identity")), "Identity")
      .Input(switch_node, 1)
      .Device(device)
      .Finalize(g, condition_true));

  return Status::OK();
}

Status ReplaceCondWithMerge(Graph* g, Node* cond) {
  std::vector<NodeBuilder::NodeOut> data_inputs;
  std::vector<Node*> control_inputs;
  for (const Edge* e : cond->in_edges()) {
    switch (e->dst_input()) {
      case Graph::kControlSlot:
        control_inputs.push_back(e->src());
        break;
      case 0:
        break;
      default:
        data_inputs.emplace_back(e->src(), e->src_output());
    }
  }

  std::vector<NodeAndPort> data_outputs;
  std::vector<Node*> control_outputs;
  for (const Edge* e : cond->out_edges()) {
    if (e->IsControlEdge()) {
      control_outputs.push_back(e->dst());
    } else {
      data_outputs.push_back({e->dst(), e->dst_input()});
    }
  }

  Node* merge_node;
  TF_RETURN_IF_ERROR(
    NodeBuilder(g->NewName(strings::StrCat(cond->name(), "/Merge")), "Merge")
      .Input(data_inputs)
      .ControlInputs(control_inputs)
      .Device(cond->def().device())
      .Finalize(g, &merge_node));

  g->RemoveNode(cond);

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

Status DoCondRewrite(Graph* g,
                     const std::unordered_set<Node*>& target_nodes) {
  /*
  LOG(INFO) << "CondRewrite";
  LOG(INFO) << DumpGraph(g);
  LOG(INFO) << "target_nodes";
  LOG(INFO) << DumpNodes(target_nodes);
  */
  std::vector<Node*> iter_order;
  std::vector<Node*> cond_nodes;
  {
    iter_order.reserve(g->num_nodes());

    std::vector<int> pending_counts(g->num_node_ids());
    for (Node* n : g->nodes()) {
      pending_counts[n->id()] = n->out_edges().size();
    }
    for (Node* n : target_nodes) {
      pending_counts[n->id()] = -1;
    }

    if (!target_nodes.count(g->sink_node())) {
      iter_order.push_back(g->sink_node());
    }

    for (Node* n : target_nodes) {
      iter_order.push_back(n);
    }

    for (int i = 0; i < iter_order.size(); i++) {
      Node* n = iter_order[i];
      if (n->IsCond()) {
        cond_nodes.push_back(n);
      }
      for (Node* m : n->in_nodes()) {
        pending_counts[m->id()]--;
        if (pending_counts[m->id()] == 0) {
          iter_order.push_back(m);
        }
      }
    }

    if (!iter_order.back()->IsSource()) {
      TF_RETURN_IF_ERROR(errors::Internal("Last node is not source"));
    }
    iter_order.pop_back();
  }
  /*
  LOG(INFO) << "iter_order" << iter_order.size();
  LOG(INFO) << "\n" << DumpNodes(iter_order);
  */
  for (Node* n : cond_nodes) {
    TF_RETURN_IF_ERROR(InsertIdentities(g, n));
  }
  /*
  LOG(INFO) << "InsertIdentities";
  LOG(INFO) << "\n" << DumpGraph(g);
  */
  std::vector<NodeInfo> node_info(g->num_node_ids());

  for (Node* n : iter_order) {
    Node* c = nullptr;
    if (!target_nodes.count(n)) {
      TF_RETURN_IF_ERROR(MergeConditions(node_info, g, n, &c));
    }

    if (n->IsCond()) {
      Node* conditions[3] = {c};
      TF_RETURN_IF_ERROR(
        CreateBranchConditions(g, n, &conditions[1], &conditions[2]));

      for (int i = 0; i < 3; i++) {
        const Node* input_node;
        TF_RETURN_IF_ERROR(n->input_node(i, &input_node));
        node_info[input_node->id()].condition = conditions[i];
      }
    }
  }
  /*
  {
    string s;
    for (NodeInfo i : node_info) {
      if (i.condition) {
        s.append(i.condition->name()).append(" ");
      } else {
        s.append("nullptr ");
      }
    }
    LOG(INFO) << "NodeInfo";
    LOG(INFO) << s;
  }
  LOG(INFO) << "CreateBranchCondtions";
  LOG(INFO) << "\n" << DumpGraph(g);
  */
  for (Node* n : g->nodes()) {
    int id = n->id();

    if (id >= node_info.size()) continue;

    Node* c = node_info[id].condition;
    if (c && !node_info[id].satisfied) {
      g->AddControlEdge(c, n);
    }
  }
  /*
  LOG(INFO) << "AddControlEdge";
  LOG(INFO) << "\n" << DumpGraph(g);
  */
  for (Node* n : cond_nodes) {
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
    TF_RETURN_IF_ERROR(ReplaceCondWithMerge(g, n));
  }

  //LOG(INFO) << "CondRewrite done.";
  return Status::OK();
}

Status DoCondRewrite(Graph* g,
                     const std::unordered_set<const Node*>& target_nodes) {
  std::unordered_set<Node*> mutable_target_nodes;
  mutable_target_nodes.reserve(target_nodes.size());

  for (const Node* n : target_nodes) {
    mutable_target_nodes.insert(g->FindNodeId(n->id()));
  }

  return DoCondRewrite(g, mutable_target_nodes);
}

}  // namespace tensorflow
