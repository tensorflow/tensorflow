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

#include "tensorflow/core/profiler/internal/tfprof_stats.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_timeline.h"

namespace tensorflow {
namespace tfprof {
TFStats::TFStats(std::unique_ptr<GraphDef> graph,
                 std::unique_ptr<RunMetadata> run_meta,
                 std::unique_ptr<OpLog> op_log,
                 std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader)
    : has_code_traces_(false),
      graph_(std::move(graph)),
      ckpt_reader_(std::move(ckpt_reader)) {
  CHECK(graph_) << "Must at least have GraphDef";

  printf("Parsing Inputs...\n");
  ParseGraph();
  if (run_meta && run_meta->has_step_stats()) {
    AddRunMeta(0, std::move(run_meta));
  }
  AddOpLog(std::move(op_log));

  if (ckpt_reader_) {
    for (const auto& v : ckpt_reader_->GetVariableToShapeMap()) {
      auto node = nodes_map_.find(v.first);
      if (node != nodes_map_.end()) {
        node->second->AddOpType("_checkpoint_variables");
      }
    }
  }
}

void TFStats::BuildView(const string& cmd) {
  if (cmd == kCmds[0] && !scope_view_) {
    scope_view_.reset(new TFScope(ckpt_reader_.get()));
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      scope_view_->AddNode(it->second.get());
    }
    scope_view_->Build();
  }
  if (cmd == kCmds[1] && !graph_view_) {
    graph_view_.reset(new TFGraph(ckpt_reader_.get()));
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      graph_view_->AddNode(it->second.get());
    }
    graph_view_->Build();
  }
  if (cmd == kCmds[2] && !code_view_) {
    code_view_.reset(new TFCode());
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      code_view_->AddNode(it->second.get());
    }
    code_view_->Build();
  }
  if (cmd == kCmds[3] && !op_view_) {
    op_view_.reset(new TFOp());
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      op_view_->AddNode(it->second.get());
    }
    op_view_->Build();
  }
}

void TFStats::BuildAllViews() {
  std::vector<string> cmds_str(kCmds, kCmds + sizeof(kCmds) / sizeof(*kCmds));
  for (const string& cmd : cmds_str) {
    BuildView(cmd);
  }
}

const TFGraphNodeProto& TFStats::ShowGraphNode(const string& cmd,
                                               const Options& opts) const {
  if (!Validate(opts)) {
    return empty_graph_node_;
  }
  if (cmd == kCmds[0]) {
    return scope_view_->Show(opts);
  } else if (cmd == kCmds[1]) {
    return graph_view_->Show(opts);
  } else {
    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    return empty_graph_node_;
  }
}

const TFMultiGraphNodeProto& TFStats::ShowMultiGraphNode(
    const string& cmd, const Options& opts) const {
  if (!Validate(opts)) {
    return empty_multi_graph_node_;
  }
  if (cmd == kCmds[2]) {
    return code_view_->Show(opts);
  } else if (cmd == kCmds[3]) {
    return op_view_->Show(opts);
  } else {
    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    return empty_multi_graph_node_;
  }
}

void TFStats::ParseGraph() {
  for (const NodeDef& node : graph_->node()) {
    CHECK(nodes_map_.find(node.name()) == nodes_map_.end());
    nodes_map_[node.name()] =
        std::unique_ptr<TFGraphNode>(new TFGraphNode(&node));
  }
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    const NodeDef* node_def = it->second->node_def();
    for (int i = 0; i < node_def->input_size(); ++i) {
      string node_input = node_def->input(i);
      int output_idx = 0;
      // input name format can be: "^node:src_output"
      auto prefix_pos = node_input.find(":");
      if (prefix_pos != node_input.npos) {
        std::vector<string> input_parts = str_util::Split(node_input, ":");
        CHECK(input_parts.size() == 2)
            << "Unknown NodeDef.input format: " << node_input;
        node_input = input_parts[0];
        CHECK(strings::safe_strto32(input_parts[1], &output_idx))
            << "Failed to parse integer: " << output_idx;
      }
      if (node_input.substr(0, 1) == "^") {
        node_input = node_input.substr(1);
      }
      auto input_node = nodes_map_.find(node_input);
      if (input_node == nodes_map_.end()) {
        continue;
      }
      it->second->AddInput(input_node->second.get(), output_idx, i);
    }
  }
}

void TFStats::AddOpLog(std::unique_ptr<OpLog> op_log) {
  if (!op_log) {
    return;
  }
  for (const OpLogEntry& entry : op_log->log_entries()) {
    auto node = nodes_map_.find(entry.name());
    if (node == nodes_map_.end()) continue;
    for (const string& type : entry.types()) {
      node->second->AddOpType(type);
    }
    if (entry.float_ops()) {
      node->second->AddFloatOps(entry.float_ops());
    }
    if (entry.has_code_def()) {
      has_code_traces_ = true;
      node->second->AddCode(entry.code_def());
    }
  }
}

void TFStats::AddRunMeta(int64 step, std::unique_ptr<RunMetadata> run_meta) {
  if (!run_meta || !run_meta->has_step_stats()) {
    fprintf(stderr, "Invalid RunMetadata for step %lld\n", step);
    return;
  }
  if (steps_.find(step) != steps_.end()) {
    fprintf(stderr, "The same step %lld has been added before.\n", step);
    return;
  }
  steps_.insert(step);

  for (const auto& dev_stat : run_meta->step_stats().dev_stats()) {
    for (const NodeExecStats& node_stat : dev_stat.node_stats()) {
      string name = node_stat.node_name();
      // Sometimes the node_name is suffixed with unnecessary information.
      auto split_pos = node_stat.node_name().find(":");
      if (split_pos != node_stat.node_name().npos) {
        name = node_stat.node_name().substr(0, split_pos);
      }
      auto node = nodes_map_.find(name);
      if (node != nodes_map_.end()) {
        node->second->AddStepStat(step, dev_stat.device(), node_stat);
      }
    }
  }
}

bool TFStats::Validate(const Options& opts) const {
  if (opts.step >= 0 && steps_.find(opts.step) == steps_.end()) {
    fprintf(stderr, "Options -step=%lld not found\n", opts.step);
    return false;
  }
  return true;
}

void TFStats::AddNodeForTest(int64 step, std::unique_ptr<TFGraphNode> node) {
  steps_.insert(step);
  nodes_map_[node->name()] = std::move(node);
}
}  // namespace tfprof
}  // namespace tensorflow
