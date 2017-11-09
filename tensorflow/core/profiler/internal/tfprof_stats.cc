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
namespace {
bool CreateRunMetadataNode(const string& name, NodeDef* def) {
  // TODO(xpan): Better solution than blacklisting this 2 nodes. They
  // actually cost some resources, maybe include them. Some nodes, such
  // as _SOURCE appear in multiple devices, which breaks tfprof's assumption.
  if (name == "RecvTensor" || name == "_SOURCE" ||
      name.find("MEMCPY") != name.npos) {
    return false;
  }
  def->set_name(name);
  // TODO(xpan): Better operation type.
  // This is because some times a node doesn't have a op type,
  // so we use node name as the op type.
  def->set_op(name);
  return true;
}
}  // namespace

TFStats::TFStats(std::unique_ptr<GraphDef> graph,
                 std::unique_ptr<RunMetadata> run_meta,
                 std::unique_ptr<OpLogProto> op_log,
                 std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader)
    : has_code_traces_(false),
      ckpt_reader_(std::move(ckpt_reader)) {
  CHECK(graph) << "Must at least have GraphDef";

  printf("Parsing Inputs...\n");
  AddGraph(std::move(graph));
  if (run_meta && run_meta->has_step_stats()) {
    AddRunMeta(0, std::move(run_meta));
  }
  AddOpLogProto(std::move(op_log));

  if (ckpt_reader_) {
    for (const auto& v : ckpt_reader_->GetVariableToShapeMap()) {
      auto node = nodes_map_.find(v.first);
      if (node != nodes_map_.end()) {
        node->second->AddOpType("_checkpoint_variables");
      }
    }
  }
}

TFStats::TFStats(const string& filename,
                 std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader)
    : has_code_traces_(false), ckpt_reader_(std::move(ckpt_reader)) {
  string str;
  Status s = ReadFileToString(Env::Default(), filename, &str);
  if (!s.ok()) {
    fprintf(stderr, "Failed to read profile: %s", s.ToString().c_str());
    return;
  }

  ProfileProto profile;
  if (!profile.ParseFromString(str)) {
    fprintf(stderr, "Failed to parse profile\n");
    return;
  }
  for (const auto& entry : profile.id_to_string()) {
    id_to_string_[entry.first] = entry.second;
  }
  for (const auto& node_pb : profile.nodes()) {
    std::unique_ptr<TFGraphNode> node(
        new TFGraphNode(node_pb.second, profile, &id_to_string_, &nodes_map_));
    nodes_map_.insert(std::pair<string, std::unique_ptr<TFGraphNode>>(
        node_pb.second.name(), std::move(node)));
  }
  has_code_traces_ = profile.has_trace();
  for (int64 s : profile.steps()) {
    steps_.insert(s);
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

const GraphNodeProto& TFStats::ShowGraphNode(const string& cmd,
                                             const Options& opts) const {
  if (!Validate(opts)) {
    return empty_graph_node_;
  }
  if (cmd == kCmds[0]) {
    return scope_view_->Show(opts);
  } else if (cmd == kCmds[1]) {
    if (opts.step < 0 && opts.output_type == kOutput[0]) {
      for (int64 step : steps_) {
        Options nopts = opts;
        nopts.step = step;
        graph_view_->Show(nopts);
      }
      return empty_graph_node_;
    }
    return graph_view_->Show(opts);
  } else {
    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    return empty_graph_node_;
  }
}

const MultiGraphNodeProto& TFStats::ShowMultiGraphNode(
    const string& cmd, const Options& opts) const {
  if (!Validate(opts)) {
    return empty_multi_graph_node_;
  }
  if (cmd == kCmds[2]) {
    if (!has_code_traces()) {
      fprintf(stderr, "No code trace information\n");
      return empty_multi_graph_node_;
    }
    return code_view_->Show(opts);
  } else if (cmd == kCmds[3]) {
    return op_view_->Show(opts);
  } else {
    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    return empty_multi_graph_node_;
  }
}

void TFStats::AddGraph(std::unique_ptr<GraphDef> graph) {
  std::map<string, const NodeDef*> node_defs;
  bool node_added = false;
  for (const NodeDef& node : graph->node()) {
    if (nodes_map_.find(node.name()) != nodes_map_.end()) {
      continue;
    }
    node_added = true;
    nodes_map_[node.name()] = std::unique_ptr<TFGraphNode>(
        new TFGraphNode(&node, nodes_map_.size(), &nodes_map_));
    node_defs[node.name()] = &node;
  }
  for (auto it = node_defs.begin(); it != node_defs.end(); it++) {
    TFGraphNode* node = nodes_map_.at(it->first).get();
    for (int i = 0; i < it->second->input_size(); ++i) {
      string node_input = it->second->input(i);
      int output_idx = 0;
      // input name format can be: "^node:src_output"
      // if not :src_output, then it's the first one (further verify?)
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
      // Delay input TFGraphNode retrieval as late as possible.
      // In long run, when we have TensorFlow runtime graph, the
      // graph connection should be dynamic and per-step.
      node->AddInput(node_input, output_idx, i);
    }
  }
  if (node_added) {
    graph_view_.reset(nullptr);
    scope_view_.reset(nullptr);
    op_view_.reset(nullptr);
    code_view_.reset(nullptr);
  }
}

void TFStats::AddOpLogProto(std::unique_ptr<OpLogProto> op_log) {
  if (!op_log) {
    return;
  }
  for (const auto& entry : op_log->id_to_string()) {
    if (id_to_string_.find(entry.first) == id_to_string_.end()) {
      id_to_string_[entry.first] = entry.second;
    }
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
      node->second->AddCode(entry.code_def(), &id_to_string_);
    }
  }
}

void TFStats::AddRunMeta(int64 step, std::unique_ptr<RunMetadata> run_meta) {
  if (!run_meta || !run_meta->has_step_stats()) {
    fprintf(stderr, "Invalid RunMetadata for step %lld\n", step);
    return;
  }
  if (steps_.find(step) == steps_.end()) {
    steps_.insert(step);
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
      if (node == nodes_map_.end()) {
        NodeDef def;
        if (CreateRunMetadataNode(name, &def)) {
          nodes_map_[name] = std::unique_ptr<TFGraphNode>(
              new TFGraphNode(&def, nodes_map_.size(), &nodes_map_));
          nodes_map_.at(name)->AddStepStat(step, dev_stat.device(), node_stat);
        }
      } else {
        covered_nodes_.insert(node->second->id());
        node->second->AddStepStat(step, dev_stat.device(), node_stat);
      }
    }
  }
}

void TFStats::WriteProfile(const string& filename) {
  ProfileProto profile;
  for (const auto& entry : id_to_string_) {
    (*profile.mutable_id_to_string())[entry.first] = entry.second;
  }
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    if (it->second->id() < 0) {
      continue;
    }
    (*profile.mutable_nodes())[it->second->id()].MergeFrom(
        it->second->ToProto(nodes_map_));
  }

  profile.set_has_trace(has_code_traces_);
  for (int64 s : steps_) {
    profile.add_steps(s);
  }
  Status s =
      WriteStringToFile(Env::Default(), filename, profile.SerializeAsString());
  if (!s.ok()) {
    fprintf(stderr, "%s\n", s.ToString().c_str());
  }
}

bool TFStats::Validate(const Options& opts) const {
  if (opts.step >= 0 && steps_.find(opts.step) == steps_.end()) {
    fprintf(stderr,
            "Options -step=%lld not found.\nAvailable steps: ", opts.step);
    for (int64 s : steps_) {
      fprintf(stderr, "%lld ", s);
    }
    fprintf(stderr, "\n");
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
