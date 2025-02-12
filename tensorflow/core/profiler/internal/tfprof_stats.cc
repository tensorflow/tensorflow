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

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_timeline.h"

namespace tensorflow {
namespace tfprof {
namespace {

const char* const kProfilePrefix = "Profile:\n";

bool CreateRunMetadataNode(const string& name, NodeDef* def) {
  // TODO(xpan): Better solution than denylisting this 2 nodes. They
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
      miss_accelerator_stream_(false),
      ckpt_reader_(std::move(ckpt_reader)) {
  CHECK(graph) << "Must at least have GraphDef";

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
    : has_code_traces_(false),
      miss_accelerator_stream_(false),
      ckpt_reader_(std::move(ckpt_reader)) {
  string str;
  absl::Status s = ReadFileToString(Env::Default(), filename, &str);
  if (!s.ok()) {
    absl::FPrintF(stderr, "Failed to read profile: %s", s.ToString());
    return;
  }

  ProfileProto profile;
  if (!profile.ParseFromString(str)) {
    absl::FPrintF(stderr, "Failed to parse profile\n");
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
  for (int64_t s : profile.steps()) {
    steps_.insert(s);
  }
}

void TFStats::BuildView(const string& cmd) {
  if (cmd == kCmds[0] && !scope_view_) {
    scope_view_ = std::make_unique<TFScope>(ckpt_reader_.get());
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      scope_view_->AddNode(it->second.get());
    }
    scope_view_->Build();
  }
  if (cmd == kCmds[1] && !graph_view_) {
    graph_view_ = std::make_unique<TFGraph>(ckpt_reader_.get());
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      graph_view_->AddNode(it->second.get());
    }
    graph_view_->Build();
  }
  if (cmd == kCmds[2] && !code_view_) {
    code_view_ = std::make_unique<TFCode>();
    for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
      code_view_->AddNode(it->second.get());
    }
    code_view_->Build();
  }
  if (cmd == kCmds[3] && !op_view_) {
    op_view_ = std::make_unique<TFOp>();
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
  string prefix = MaybeReportMissingTrace();
  prefix += QueryDoc(cmd, opts) + kProfilePrefix;

  if (cmd == kCmds[0]) {
    return scope_view_->Show(prefix, opts);
  } else if (cmd == kCmds[1]) {
    if (opts.step < 0 && opts.output_type == kOutput[0]) {
      for (int64_t step : steps_) {
        Options nopts = opts;
        nopts.step = step;
        graph_view_->Show(prefix, nopts);
      }
      return empty_graph_node_;
    }
    return graph_view_->Show(prefix, opts);
  } else {
    absl::FPrintF(stderr, "Unknown command: %s\n", cmd);
    return empty_graph_node_;
  }
}

const MultiGraphNodeProto& TFStats::ShowMultiGraphNode(
    const string& cmd, const Options& opts) const {
  if (!Validate(opts)) {
    return empty_multi_graph_node_;
  }
  string prefix = MaybeReportMissingTrace();
  prefix += QueryDoc(cmd, opts) + kProfilePrefix;

  if (cmd == kCmds[2]) {
    if (!has_code_traces()) {
      absl::FPrintF(stderr, "No code trace information\n");
      return empty_multi_graph_node_;
    }
    return code_view_->Show(prefix, opts);
  } else if (cmd == kCmds[3]) {
    return op_view_->Show(prefix, opts);
  } else {
    absl::FPrintF(stderr, "Unknown command: %s\n", cmd);
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
    size_t num_nodes = nodes_map_.size();
    nodes_map_[node.name()] =
        std::make_unique<TFGraphNode>(&node, num_nodes, &nodes_map_);
    node_defs[node.name()] = &node;
  }
  for (auto it = node_defs.begin(); it != node_defs.end(); it++) {
    TFGraphNode* node = nodes_map_.at(it->first).get();
    for (int i = 0; i < it->second->input_size(); ++i) {
      string node_input = it->second->input(i);
      int output_idx = 0;
      // input name format can be: "^node:src_output"
      // if not :src_output, then it's the first one (further verify?)
      auto prefix_pos = node_input.find(':');
      if (prefix_pos != node_input.npos) {
        std::vector<string> input_parts = absl::StrSplit(node_input, ':');
        DCHECK(input_parts.size() == 2)
            << "Unknown NodeDef.input format: " << node_input;
        node_input = input_parts[0];
        DCHECK(absl::SimpleAtoi(input_parts[1], &output_idx))
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

void TFStats::AddRunMeta(int64_t step, std::unique_ptr<RunMetadata> run_meta) {
  if (!run_meta || !run_meta->has_step_stats()) {
    absl::FPrintF(stderr, "Invalid RunMetadata for step %d\n", step);
    return;
  }
  if (steps_.find(step) == steps_.end()) {
    steps_.insert(step);
  }
  steps_.insert(step);

  bool has_gpu_scheduling = false;
  bool has_gpu_stream = false;

  for (const auto& dev_stat : run_meta->step_stats().dev_stats()) {
    string dev = absl::AsciiStrToLower(dev_stat.device());
    if (IsPlacedOnAccelerator(dev)) {
      has_gpu_scheduling = true;
      if (CountAsAcceleratorTime(dev)) {
        has_gpu_stream = true;
      }
    }
    for (const NodeExecStats& node_stat : dev_stat.node_stats()) {
      string name = node_stat.node_name();
      // Sometimes the node_name is suffixed with unnecessary information.
      auto split_pos = node_stat.node_name().find(':');
      if (split_pos != node_stat.node_name().npos) {
        name = node_stat.node_name().substr(0, split_pos);
      }
      auto node = nodes_map_.find(name);
      if (node == nodes_map_.end()) {
        NodeDef def;
        if (CreateRunMetadataNode(name, &def)) {
          size_t num_nodes = nodes_map_.size();
          nodes_map_[name] =
              std::make_unique<TFGraphNode>(&def, num_nodes, &nodes_map_);
          nodes_map_.at(name)->AddStepStat(step, dev_stat.device(), node_stat);
        }
      } else {
        covered_nodes_.insert(node->second->id());
        node->second->AddStepStat(step, dev_stat.device(), node_stat);
      }
    }
  }

  if (has_gpu_scheduling && !has_gpu_stream) {
    miss_accelerator_stream_ = true;
  }
}

string TFStats::MaybeReportMissingTrace() const {
  string report = "";
  if (miss_accelerator_stream_) {
    report +=
        "\n\nFound accelerator operation but misses accelerator "
        "stream stats!\n\n"
        "It's likely a gpu tracing issue rather than tf-profiler issue.\n"
        "If you found your operation missing accelerator time, "
        "consider to post to discuss@tensorflow.org!\n\n";
  }
  return report;
}

void TFStats::SerializeToString(string* content) {
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
  profile.set_miss_accelerator_stream(miss_accelerator_stream_);
  for (int64_t s : steps_) {
    profile.add_steps(s);
  }
  *content = profile.SerializeAsString();
}

void TFStats::WriteProfile(const string& filename) {
  string content;
  SerializeToString(&content);
  absl::Status s = WriteStringToFile(Env::Default(), filename, content);
  if (!s.ok()) {
    absl::FPrintF(stderr, "%s\n", s.ToString());
  }
}

bool TFStats::Validate(const Options& opts) const {
  if (opts.step >= 0 && steps_.find(opts.step) == steps_.end()) {
    absl::FPrintF(stderr,
                  "Options -step=%d not found.\nAvailable steps: ", opts.step);
    for (int64_t s : steps_) {
      absl::FPrintF(stderr, "%d ", s);
    }
    absl::FPrintF(stderr, "\n");
    return false;
  }
  return true;
}

void TFStats::AddNodeForTest(int64_t step, std::unique_ptr<TFGraphNode> node) {
  steps_.insert(step);
  nodes_map_[node->name()] = std::move(node);
}
}  // namespace tfprof
}  // namespace tensorflow
