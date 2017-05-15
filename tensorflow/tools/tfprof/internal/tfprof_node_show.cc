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
ShowNode::ShowNode(const TFGraphNode* node) : node(node), account(true) {
  ReInit();
}

void ShowNode::ReInit() {
  mutable_proto()->set_name(name());
  for (const string& device : node->devices()) {
    *mutable_proto()->mutable_devices()->Add() = device;
  }
  mutable_proto()->set_exec_micros(node->kernel_exec_micros());
  mutable_proto()->set_requested_bytes(node->requested_bytes());
  mutable_proto()->set_float_ops(node->float_ops());

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

string ShowNode::Format(const Options& opts) {
  if (opts.select.empty()) {
    return name();
  }
  return strings::Printf("%s (%s)", name().c_str(), FormatMeta(opts).c_str());
}

string ShowNode::FormatMeta(const Options& opts) {
  std::vector<string> info;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    const string shape = FormatShapes(node->shape());
    if (!shape.empty()) {
      info.push_back(shape);
    }
    string params = FormatNumber(proto().total_parameters()) + " params";
    if (account) {
      params = FormatNumber(proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    info.push_back(params);
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(proto().total_float_ops()) + " flops";
    if (account) {
      fops = FormatNumber(proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    info.push_back(fops);
  }
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    string memory = FormatMemory(proto().total_requested_bytes());
    if (account) {
      memory = FormatMemory(proto().requested_bytes()) + "/" + memory;

    } else {
      memory = "--/" + memory;
    }
    info.push_back(memory);
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    string time = FormatTime(proto().total_exec_micros());
    if (account) {
      time = FormatTime(proto().exec_micros()) + "/" + time;
    } else {
      time = "--/" + time;
    }
    info.push_back(time);
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    if (proto().devices_size() > 0) {
      info.push_back(str_util::Join(proto().devices(), "|"));
    }
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    std::set<string> op_types = node->op_types();
    // Device is considered a type.
    if (proto().devices_size() > 0) {
      op_types.insert(str_util::Join(proto().devices(), "|"));
    }
    info.push_back(str_util::Join(op_types, "|"));
  }
  return str_util::Join(info, ", ");
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

ShowCodeNode::ShowCodeNode(const TFCodeNode* node) : node(node), account(true) {
  std::vector<ScopeNode> snodes;
  for (auto it : node->graph_nodes()) {
    ScopeNode snode(it.second);
    snodes.push_back(snode);
    snodes[snodes.size() - 1].AddSelfToTotalStats();
    *mutable_proto()->mutable_graph_nodes()->Add() =
        snodes[snodes.size() - 1].proto();
  }

  mutable_proto()->set_name(name());
  mutable_proto()->set_exec_micros(node->kernel_exec_micros());
  mutable_proto()->set_requested_bytes(node->requested_bytes());
  mutable_proto()->set_float_ops(node->float_ops());

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
}

string ShowCodeNode::Format(const Options& opts) {
  if (opts.select.empty()) {
    return name();
  }
  return strings::Printf("%s (%s)", name().c_str(), FormatMeta(opts).c_str());
}

string ShowCodeNode::FormatMeta(const Options& opts) {
  std::vector<string> info;
  std::vector<string> shapes;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    for (const std::vector<int64>& shape : node->shapes()) {
      if (!shape.empty()) {
        shapes.push_back(FormatShapes(shape));
      }
    }
    if (!shapes.empty()) {
      info.push_back(str_util::Join(shapes, "|"));
    }
    string params = FormatNumber(proto().total_parameters()) + " params";
    if (account) {
      params = FormatNumber(proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    info.push_back(params);
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(proto().total_float_ops()) + " flops";
    if (account) {
      fops = FormatNumber(proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    info.push_back(fops);
  }
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    string memory = FormatMemory(proto().total_requested_bytes());
    if (account) {
      memory = FormatMemory(proto().requested_bytes()) + "/" + memory;

    } else {
      memory = "--/" + memory;
    }
    info.push_back(memory);
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    string time = FormatTime(proto().total_exec_micros());
    if (account) {
      time = FormatTime(proto().exec_micros()) + "/" + time;
    } else {
      time = "--/" + time;
    }
    info.push_back(time);
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    if (!node->devices().empty()) {
      info.push_back(str_util::Join(node->devices(), "|"));
    }
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    std::set<string> op_types = node->op_types();
    // Device is considered a type.
    op_types.insert(node->devices().cbegin(), node->devices().cend());
    info.push_back(str_util::Join(op_types, "|"));
  }
  return str_util::Join(info, ", ");
}

TFCodeNodeProto* ShowCodeNode::mutable_proto() { return &proto_; }

const TFCodeNodeProto& ShowCodeNode::proto() const { return proto_; }

void ShowCodeNode::AggregateTotalStats(ShowCodeNode* node) {
  TFCodeNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowCodeNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowCodeNode::ResetTotalStats() {
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

}  // namespace tfprof
}  // namespace tensorflow
