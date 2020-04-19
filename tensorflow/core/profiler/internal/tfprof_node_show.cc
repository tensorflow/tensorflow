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
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"

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
  mutable_proto()->set_run_count(node->run_count(step));
  mutable_proto()->set_exec_micros(node->exec_micros(step));
  mutable_proto()->set_accelerator_exec_micros(
      node->accelerator_exec_micros(step));
  mutable_proto()->set_cpu_exec_micros(node->cpu_exec_micros(step));

  mutable_proto()->set_requested_bytes(node->requested_bytes(step));
  mutable_proto()->set_peak_bytes(node->peak_bytes(step));
  mutable_proto()->set_residual_bytes(node->residual_bytes(step));
  mutable_proto()->set_output_bytes(node->output_bytes(step));

  mutable_proto()->set_float_ops(node->float_ops(step));

  mutable_proto()->clear_input_shapes();
  for (const auto& inp : node->input_shapes()) {
    (*mutable_proto()->mutable_input_shapes())[inp.first].MergeFrom(
        VecToShapeProto(inp.second));
  }
  proto_.set_parameters(node->parameters());
}

GraphNodeProto* ShowNode::mutable_proto() { return &proto_; }

const GraphNodeProto& ShowNode::proto() const { return proto_; }

void ShowNode::AggregateTotalStats(ShowNode* node) {
  GraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_run_count(proto().total_run_count() +
                                       node_pb->total_run_count());
  mutable_proto()->set_total_definition_count(
      proto().total_definition_count() + node_pb->total_definition_count());
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      node_pb->total_accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             node_pb->total_cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        node_pb->total_peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            node_pb->total_residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          node_pb->total_output_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_definition_count(proto().total_definition_count() +
                                              1);
  mutable_proto()->set_total_run_count(proto().total_run_count() +
                                       proto().run_count());
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      proto().accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             proto().cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        proto().peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            proto().residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          proto().output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowNode::ResetTotalStats() {
  formatted_str.clear();

  mutable_proto()->set_total_definition_count(0);
  mutable_proto()->set_total_run_count(0);
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_accelerator_exec_micros(0);
  mutable_proto()->set_total_cpu_exec_micros(0);

  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_peak_bytes(0);
  mutable_proto()->set_total_residual_bytes(0);
  mutable_proto()->set_total_output_bytes(0);

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
  for (const auto& it : node->graph_nodes()) {
    ShowNode snode(it.second);
    snodes.push_back(snode);
    snodes.back().ReInit(step);
    snodes.back().AddSelfToTotalStats();
    mutable_proto()->add_graph_nodes()->MergeFrom(snodes.back().proto());
  }

  mutable_proto()->set_name(name());
  mutable_proto()->set_exec_micros(node->exec_micros());
  mutable_proto()->set_accelerator_exec_micros(node->accelerator_exec_micros());
  mutable_proto()->set_cpu_exec_micros(node->cpu_exec_micros());

  mutable_proto()->set_requested_bytes(node->requested_bytes());
  mutable_proto()->set_peak_bytes(node->peak_bytes());
  mutable_proto()->set_residual_bytes(node->residual_bytes());
  mutable_proto()->set_output_bytes(node->output_bytes());

  mutable_proto()->set_float_ops(node->float_ops());

  mutable_proto()->set_parameters(node->parameters());
  return has_matched_type;
}

MultiGraphNodeProto* ShowMultiNode::mutable_proto() { return &proto_; }

const MultiGraphNodeProto& ShowMultiNode::proto() const { return proto_; }

void ShowMultiNode::AggregateTotalStats(ShowMultiNode* node) {
  MultiGraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      node_pb->total_accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             node_pb->total_cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        node_pb->total_peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            node_pb->total_residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          node_pb->total_output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowMultiNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      proto().accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             proto().cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        proto().peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            proto().residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          proto().output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowMultiNode::ResetTotalStats() {
  formatted_str.clear();
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_accelerator_exec_micros(0);
  mutable_proto()->set_total_cpu_exec_micros(0);

  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_peak_bytes(0);
  mutable_proto()->set_total_residual_bytes(0);
  mutable_proto()->set_total_output_bytes(0);

  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

}  // namespace tfprof
}  // namespace tensorflow
