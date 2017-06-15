/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/tools/tfprof/internal/tfprof_node_show.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tfprof {
namespace {}

ShowNode::ShowNode(const TFGraphNode* node) : node(node), account(false) {
  ReInit(-1);
}

void ShowNode::ReInit(int64 step) {
  mutable_proto()->set_name(name());
  mutable_proto()->clear_devices();
  if (!node->canonical_device().empty()) {
    mutable_proto()->add_devices(node->canonical_device());
  }
  mutable_proto()->set_exec_micros(node->kernel_exec_micros(step));
  mutable_proto()->set_requested_bytes(node->requested_bytes(step));
  mutable_proto()->set_float_ops(node->float_ops());

  mutable_proto()->clear_input_shapes();
  for (const auto& inp : node->input_shapes()) {
    (*mutable_proto()->mutable_input_shapes())[inp.first].MergeFrom(
        VecToShapeProto(inp.second));
  }

  proto_.clear_parameters();
  if (!node->shape().empty()) {
    int64 params = 1;
    bool complete_shape = true;
    for (int64 d : node->shape()) {
      // Sometimes parameters could be <0 when a dim is unknown.
      if (d < 0) {
        complete_shape = false;
        break;
      }
      params *= d;
    }
    if (complete_shape) {
      mutable_proto()->set_parameters(proto_.parameters() + params);
    } else {
      fprintf(stderr, "Incomplete shape.");
    }
  }
}

TFGraphNodeProto* ShowNode::mutable_proto() { return &proto_; }

const TFGraphNodeProto& ShowNode::proto() const { return proto_; }

void ShowNode::AggregateTotalStats(ShowNode* node) {
  TFGraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowNode::ResetTotalStats() {
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

ShowMultiNode::ShowMultiNode(TFMultiGraphNode* node)
    : node(node), account(false), show(false) {
  ReInit(-1, {".*"});
}

bool ShowMultiNode::ReInit(int64 step,
                           const std::vector<string>& type_regexes) {
  bool has_matched_type = node->SnapshotNodes(step, type_regexes);

  std::vector<ShowNode> snodes;
  mutable_proto()->mutable_graph_nodes()->Clear();
  for (auto it : node->graph_nodes()) {
    ShowNode snode(it.second);
    snodes.push_back(snode);
    snodes.back().ReInit(step);
    snodes.back().AddSelfToTotalStats();
    mutable_proto()->add_graph_nodes()->MergeFrom(snodes.back().proto());
  }

  mutable_proto()->set_name(name());
  mutable_proto()->set_exec_micros(node->kernel_exec_micros());
  mutable_proto()->set_requested_bytes(node->requested_bytes());
  mutable_proto()->set_float_ops(node->float_ops());

  mutable_proto()->clear_parameters();
  if (!node->shapes().empty()) {
    for (const std::vector<int64>& shape : node->shapes()) {
      int64 params = 1;
      bool complete_shape = true;
      for (int64 d : shape) {
        // Sometimes parameters could be <0 when a dim is unknown.
        if (d < 0) {
          complete_shape = false;
          break;
        }
        params *= d;
      }
      if (complete_shape) {
        mutable_proto()->set_parameters(proto().parameters() + params);
      } else {
        fprintf(stderr, "Incomplete shape.");
      }
    }
  }
  return has_matched_type;
}

TFMultiGraphNodeProto* ShowMultiNode::mutable_proto() { return &proto_; }

const TFMultiGraphNodeProto& ShowMultiNode::proto() const { return proto_; }

void ShowMultiNode::AggregateTotalStats(ShowMultiNode* node) {
  TFMultiGraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowMultiNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowMultiNode::ResetTotalStats() {
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

}  // namespace tfprof
}  // namespace tensorflow
