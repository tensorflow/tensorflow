/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "tensorflow/core/graph/auto_mixed_precision.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {

class AutoMixedPrecision {
 public:
  explicit AutoMixedPrecision(Graph* g) :
           g_(g) {
    f_.set_type(DT_FLOAT);
    h_.set_type(DT_HALF);
    false_.set_b(false);
  }

  bool Optimize();

 private:
  Graph* g_;

  static int num_matched_;

  AttrValue f_;
  AttrValue h_;
  AttrValue false_;

  bool HandleFusedBN(const string& op, Node* node);
  bool HandleFusedBNGrad(const string& op, Node* node);
  bool HandleConv2DBPFilter(const string& op, Node* node);
  bool HandleConv2DBPInput(const string& op, Node* node);
  bool HandleBinaryOp(const string& op, Node* node);
  bool HandleUnaryOp(const string& op, Node* node);
  bool HandleNInputsOp(const string& op, Node* node);
  bool SkipCasts();

  // helper functions for constructing new subgraph
  void AddInput(NodeDef& ndef, const Edge* iedge) {
    string input_name = strings::StrCat(iedge->src()->def().name(),
                            ":", iedge->src_output());
    ndef.add_input(input_name);
  }

  void AddIedge(Node* dst, int dst_input,
      const Edge* ori_edge, bool remove = true) {
    g_->AddEdge(ori_edge->src(), ori_edge->src_output(), dst, dst_input);
    if (remove) {
      g_->RemoveEdge(ori_edge);
    }
  }

  void GetInputs(
      std::map<int, const Edge*>& inputs,
      std::vector<Node*>& control_inputs,
      Node* node) {
    for (auto* iedge : node->in_edges()) {
      int input_port = iedge->dst_input();
      if (input_port == -1) {
        control_inputs.push_back(iedge->src());
      } else {
        inputs[input_port] = iedge;
      }
    }
  }

  NodeDef CreateNodeDefCastf2h(
      const string& name,
      const string& device,
      const Edge* input) {
    NodeDef cast_def;
    cast_def.set_op("Cast");
    cast_def.set_name(name);
    cast_def.set_device(device);
    AddInput(cast_def, input);
    cast_def.mutable_attr()->insert({"SrcT", f_});
    cast_def.mutable_attr()->insert({"DstT", h_});
    cast_def.mutable_attr()->insert({"Truncate", false_});
    return cast_def;
  }

  NodeDef CreateNodeDefCasth2f(
      const string& name,
      const string& device,
      const string& input) {
    NodeDef cast_def;
    cast_def.set_op("Cast");
    cast_def.set_name(name);
    cast_def.set_device(device);
    cast_def.add_input(input);
    cast_def.mutable_attr()->insert({"SrcT", h_});
    cast_def.mutable_attr()->insert({"DstT", f_});
    cast_def.mutable_attr()->insert({"Truncate", false_});
    return cast_def;
  }

  bool CreateNode(
      Node*& node,
      const NodeDef& node_def,
      const string& device_name,
      const int device_name_index) {
    Status status;
    node = g_->AddNode(node_def, &status);
    if (status != Status::OK()) {
      LOG(INFO) << status.error_message();
      return false;
    }
    node->set_assigned_device_name(device_name);
    node->set_assigned_device_name_index(device_name_index);
    return true;
  }
};

int AutoMixedPrecision::num_matched_ = 0;

bool AutoMixedPrecision::HandleFusedBN(const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef cast_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast",
        /*device=*/node->def().device(),
        /*input=*/inputs[0]);
    Node* cast_node;
    bool success = CreateNode(
        /*result_node=*/cast_node,
        /*node_def=*/cast_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast_node, 0, inputs[0]);

    NodeDef half_def;
    half_def.set_op("FusedBatchNormV2");
    half_def.set_name(name_prefix + "_BNV2");
    cast_def.set_device(node->def().device());
    half_def.add_input(name_prefix + "_cast");
    AddInput(half_def, inputs[1]);
    AddInput(half_def, inputs[2]);
    AddInput(half_def, inputs[3]);
    AddInput(half_def, inputs[4]);
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    half_def.mutable_attr()->insert({"U", f_});
    Node* half_node;
    success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(cast_node, 0, half_node, 0);
    AddIedge(half_node, 1, inputs[1]);
    AddIedge(half_node, 2, inputs[2]);
    AddIedge(half_node, 3, inputs[3]);
    AddIedge(half_node, 4, inputs[4]);

    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_BNV2");
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);
    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      } else if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      } else {
        g_->AddEdge(half_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }
  return changed;
}

bool AutoMixedPrecision::HandleFusedBNGrad(const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef cast_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast",
        /*device=*/node->def().device(),
        /*input=*/inputs[0]);
    Node* cast_node;
    bool success = CreateNode(
        /*result_node=*/cast_node,
        /*node_def=*/cast_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast_node, 0, inputs[0]);

    NodeDef cast1_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast1",
        /*device=*/node->def().device(),
        /*input=*/inputs[1]);
    Node* cast1_node;
    success = CreateNode(
        /*result_node=*/cast1_node,
        /*node_def=*/cast1_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast1_node, 0, inputs[1]);

    NodeDef half_def;
    half_def.set_op("FusedBatchNormGradV2");
    half_def.set_name(name_prefix + "_BNGradV2");
    half_def.set_device(node->def().device());
    half_def.add_input(name_prefix + "_cast");
    half_def.add_input(name_prefix + "_cast1");
    AddInput(half_def, inputs[2]);
    AddInput(half_def, inputs[3]);
    AddInput(half_def, inputs[4]);
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    half_def.mutable_attr()->insert({"U", f_});
    Node* half_node;
    success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    g_->AddEdge(cast_node, 0, half_node, 0);
    g_->AddEdge(cast1_node, 0, half_node, 1);
    AddIedge(half_node, 2, inputs[2]);
    AddIedge(half_node, 3, inputs[3]);
    AddIedge(half_node, 4, inputs[4]);
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_BNGradV2");
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);
    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      } else if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      } else {
        g_->AddEdge(half_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }
  return changed;
}

bool AutoMixedPrecision::HandleBinaryOp(const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef cast_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast",
        /*device=*/node->def().device(),
        /*input=*/inputs[0]);
    Node* cast_node;
    bool success = CreateNode(
        /*result_node=*/cast_node,
        /*node_def=*/cast_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast_node, 0, inputs[0]);

    NodeDef cast1_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast1",
        /*device=*/node->def().device(),
        /*input=*/inputs[1]);
    Node* cast1_node;
    success = CreateNode(
        /*result_node=*/cast1_node,
        /*node_def=*/cast1_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast1_node, 0, inputs[1]);

    NodeDef half_def;
    half_def.set_op(node->type_string());
    half_def.set_name(name_prefix + "_" + node->type_string());
    half_def.set_device(node->def().device());
    half_def.add_input(name_prefix + "_cast");
    half_def.add_input(name_prefix + "_cast1");
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    Node* half_node;
    success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(cast_node, 0, half_node, 0);
    g_->AddEdge(cast1_node, 0, half_node, 1);
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_" + node->type_string());
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);
    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
      if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }
  return changed;
}

bool AutoMixedPrecision::HandleConv2DBPFilter(
    const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef cast_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast",
        /*device=*/node->def().device(),
        /*input=*/inputs[0]);
    Node* cast_node;
    bool success = CreateNode(
        /*result_node=*/cast_node,
        /*node_def=*/cast_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast_node, 0, inputs[0]);

    NodeDef cast1_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast1",
        /*device=*/node->def().device(),
        /*input=*/inputs[2]);
    Node* cast1_node;
    success = CreateNode(
        /*result_node=*/cast1_node,
        /*node_def=*/cast1_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast1_node, 0, inputs[2]);

    NodeDef half_def;
    half_def.set_op(node->type_string());
    half_def.set_name(name_prefix + "_" + node->type_string());
    half_def.set_device(node->def().device());
    half_def.add_input(name_prefix + "_cast");
    AddInput(half_def, inputs[1]);
    half_def.add_input(name_prefix + "_cast1");
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    Node* half_node;
    success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(cast_node, 0, half_node, 0);
    AddIedge(half_node, 1, inputs[1]);
    g_->AddEdge(cast1_node, 0, half_node, 2);
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_" + node->type_string());
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);

    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
      if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }
  return changed;
}

bool AutoMixedPrecision::HandleConv2DBPInput(
    const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef cast_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast",
        /*device=*/node->def().device(),
        /*input=*/inputs[2]);
    Node* cast_node;
    bool success = CreateNode(
        /*result_node=*/cast_node,
        /*node_def=*/cast_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast_node, 0, inputs[2]);

    NodeDef cast1_def = CreateNodeDefCastf2h(
        /*name=*/name_prefix + "_cast1",
        /*device=*/node->def().device(),
        /*input=*/inputs[1]);
    Node* cast1_node;
    success = CreateNode(
        /*result_node=*/cast1_node,
        /*node_def=*/cast1_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }
    AddIedge(cast1_node, 0, inputs[1]);

    NodeDef half_def;
    half_def.set_op(node->type_string());
    half_def.set_name(name_prefix + "_" + node->type_string());
    half_def.set_device(node->def().device());
    AddInput(half_def, inputs[0]);
    half_def.add_input(name_prefix + "_cast1");
    half_def.add_input(name_prefix + "_cast");
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    Node* half_node;
    success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    AddIedge(half_node, 0, inputs[0]);
    g_->AddEdge(cast1_node, 0, half_node, 1);
    g_->AddEdge(cast_node, 0, half_node, 2);
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_" + node->type_string());
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);
    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
      if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }
  return changed;
}

bool AutoMixedPrecision::HandleUnaryOp(const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    const Node* pre_cast = inputs[0]->src();
    if (pre_cast->type_string() != "Cast") {
      return false;
    }
    if (pre_cast->def().attr().at("SrcT").type() != DT_HALF) {
      return false;
    }
    const Edge* cast_input = nullptr;
    for (auto* iedge : pre_cast->in_edges()) {
      int input_port = iedge->dst_input();
      if (input_port == -1) {
        control_input.push_back(iedge->src());
      } else {
        cast_input = iedge;
      }
    }

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef half_def;
    half_def.set_op(node->type_string());
    half_def.set_name(name_prefix + "_" + node->type_string());
    half_def.set_device(node->def().device());
    half_def.add_input(pre_cast->def().input(0));
    if (node->type_string() == "Reshape") {
      half_def.add_input(node->def().input(1));
    }
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    Node* half_node;
    bool success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(cast_input->src(), cast_input->src_output(), half_node, 0);
    if (node->type_string() == "Reshape") {
      AddIedge(half_node, 1, inputs[1]);
    }
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_" + node->type_string());
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);

    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        if (node->type_string() == "Shape") {
          g_->AddEdge(
            half_node, output_port, oedge->dst(), oedge->dst_input());
        } else {
          g_->AddEdge(
            cast2_node, output_port, oedge->dst(), oedge->dst_input());
        }
        g_->RemoveEdge(oedge);
      }
      if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }

  return changed;
}

bool AutoMixedPrecision::HandleNInputsOp(const string& op, Node* node) {
  bool changed = false;
  if (node->type_string() == op) {
    std::map<int, const Edge*> inputs;
    std::vector<Node*> control_input;
    GetInputs(inputs, control_input, node);

    bool skip = false;
    for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
      const Node* pre_cast = iter->second->src();
      if (pre_cast->type_string() != "Cast") {
        skip = true;
        break;
      }
      if (pre_cast->def().attr().at("SrcT").type() != DT_HALF) {
        skip = true;
        break;
      }
    }
    if (skip) {
      return false;
    }

    ++num_matched_;
    string name_prefix = strings::StrCat("mixed_precision_", num_matched_);

    NodeDef half_def;
    half_def.set_op(node->type_string());
    half_def.set_name(name_prefix + "_" + node->type_string());
    half_def.set_device(node->def().device());
    for (int i = 0; i < inputs.size(); ++i) {
      const Node* pre_cast = inputs.at(i)->src();
      half_def.add_input(pre_cast->def().input(0));
    }
    auto attr = node->def().attr();
    *half_def.mutable_attr() = attr;
    (*half_def.mutable_attr())["T"] = h_;
    Node* half_node;
    bool success = CreateNode(
        /*result_node=*/half_node,
        /*node_def=*/half_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    for (int i = 0; i < inputs.size(); ++i) {
      const Node* pre_cast = inputs.at(i)->src();
      const Edge* cast_input = nullptr;
      for (auto* iedge : pre_cast->in_edges()) {
        int input_port = iedge->dst_input();
        if (input_port == -1) {
          control_input.push_back(iedge->src());
        } else {
          cast_input = iedge;
        }
      }
      g_->AddEdge(cast_input->src(), cast_input->src_output(), half_node, i);
    }
    for (auto control_src : control_input) {
      g_->AddControlEdge(control_src, half_node);
    }

    NodeDef cast2_def = CreateNodeDefCasth2f(
        /*name=*/name_prefix + "_cast2",
        /*device=*/node->def().device(),
        /*input=*/name_prefix + "_" + node->type_string());
    Node* cast2_node;
    success = CreateNode(
        /*result_node=*/cast2_node,
        /*node_def=*/cast2_def,
        /*device_name=*/node->assigned_device_name(),
        /*device_name_index=*/node->assigned_device_name_index());
    if (!success) {
      return false;
    }

    g_->AddEdge(half_node, 0, cast2_node, 0);

    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast2_node, output_port, oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
      }
      if (output_port == -1) {
        g_->AddControlEdge(half_node, oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }

  return changed;
}

bool AutoMixedPrecision::SkipCasts() {
  bool changed = false;

  for (Node* node : g_->nodes()) {
    if (node->type_string() != "Cast") {
      continue;
    }
    std::vector<Node*> control_input;
    const Node* pre_cast = nullptr;
    for (auto* iedge : node->in_edges()) {
      int input_port = iedge->dst_input();
      if (input_port == -1) {
        control_input.push_back(iedge->src());
      } else {
        pre_cast = iedge->src();
      }
    }
    if (pre_cast->type_string() != "Cast") {
      continue;
    }
    if (pre_cast->def().attr().at("SrcT").type()
        != node->def().attr().at("DstT").type()) {
      continue;
    }

    const Edge* cast_input = nullptr;
    for (auto* iedge : pre_cast->in_edges()) {
      int input_port = iedge->dst_input();
      if (input_port == -1) {
        control_input.push_back(iedge->src());
      } else {
        cast_input = iedge;
      }
    }

    std::vector<Edge*> oedges;
    for (auto* oedge : node->out_edges()) {
      oedges.push_back(const_cast<Edge*>(oedge));
    }
    for (auto* oedge : oedges) {
      int output_port = oedge->src_output();
      if (output_port == 0) {
        g_->AddEdge(cast_input->src(), cast_input->src_output(),
                   oedge->dst(), oedge->dst_input());
        g_->RemoveEdge(oedge);
        for (auto control_src : control_input) {
          g_->AddControlEdge(control_src, oedge->dst());
        }
      }
      if (output_port == -1) {
        g_->AddControlEdge(cast_input->src(), oedge->dst());
        g_->RemoveEdge(oedge);
      }
    }
    changed = true;
  }

  return changed;
}

bool AutoMixedPrecision::Optimize() {
  bool changed = false;

  for (Node* node : g_->nodes()) {
    if (node->def().attr().count("T") == 0) {
      continue;
    }
    if (node->def().attr().at("T").type() != DT_FLOAT) {
      continue;
    }

    if (node->def().device().find("GPU") == -1) {
      continue;
    }

    changed = HandleFusedBN("FusedBatchNorm", node) || changed;
    changed = HandleFusedBNGrad("FusedBatchNormGrad", node) || changed;
    changed = HandleConv2DBPFilter("Conv2DBackpropFilter", node) || changed;
    changed = HandleConv2DBPInput("Conv2DBackpropInput", node) || changed;

    changed = HandleBinaryOp("Conv2D", node) || changed;
    changed = HandleBinaryOp("MatMul", node) || changed;
    // FP16 BatchMatMul is not supported yet

    changed = HandleUnaryOp("Relu", node) || changed;
    changed = HandleUnaryOp("Identity", node) || changed;
    changed = HandleUnaryOp("MaxPool", node) || changed;
    changed = HandleUnaryOp("Shape", node) || changed;
    // Reshape is not unary op, but only one input needs conversion
    changed = HandleUnaryOp("Reshape", node) || changed;

    changed = HandleNInputsOp("ReluGrad", node) || changed;
    changed = HandleNInputsOp("Add", node) || changed;
    changed = HandleNInputsOp("AddN", node) || changed;
  }

  // Skip 2 connected Cast Ops where firt one is from half to float
  // and the second one is from float to half
  changed = SkipCasts() || changed;

  return changed;
}

bool RunAutoMixedPrecision(Graph* g) {
  bool changed = false;
  std::unique_ptr<AutoMixedPrecision> opt(new AutoMixedPrecision(g));
  changed = opt->Optimize();
  return changed;
}
}  // namespace tensorflow
