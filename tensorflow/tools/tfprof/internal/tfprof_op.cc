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

#include "tensorflow/tools/tfprof/internal/tfprof_op.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/tools/tfprof/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {

void TFOp::AddNode(TFGraphNode* node) {
  const string& op = node->op();
  if (tfcnodes_map_.find(op) == tfcnodes_map_.end()) {
    tfcnodes_map_[op] =
        std::unique_ptr<TFMultiGraphNode>(new TFMultiGraphNode(op));
  }
  TFMultiGraphNode* tfcnode = tfcnodes_map_[op].get();
  tfcnode->AddGraphNode(node);
}

void TFOp::Build() {
  for (auto& tn : tfcnodes_map_) {
    cnodes_map_[tn.first] =
        std::unique_ptr<OpNode>(new OpNode(tn.second.get()));
  }

  tfcnodes_map_[kTFProfRoot] =
      std::unique_ptr<TFMultiGraphNode>(new TFMultiGraphNode(kTFProfRoot));
  root_.reset(new OpNode(tfcnodes_map_[kTFProfRoot].get()));
}

const ShowMultiNode* TFOp::ShowInternal(const Options& opts,
                                         Timeline* timeline) {
  root_->ResetTotalStats();
  if (opts.output_type == kOutput[1] || opts.output_type == kOutput[2]) {
    root_->formatted_str = FormatNode(root_.get(), root_.get(), opts);
  }
  if (timeline) {
    fprintf(stderr, "op view doesn't support timeline yet. "
                    "Consider graph/scope/code view.\n");
    return root_.get();
  }
  if (cnodes_map_.empty()) {
    return root_.get();
  }

  std::vector<OpNode*> nodes;
  for (auto& n : cnodes_map_) {
    n.second->account = ReAccount(n.second.get(), opts);
    n.second->ResetTotalStats();
    n.second->AddSelfToTotalStats();
    nodes.push_back(n.second.get());
  }
  nodes = SortNodes(nodes, opts);

  OpNode* pre = nullptr;
  std::vector<OpNode*> account_nodes;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if ((*it)->account) {
      if (pre) (*it)->AggregateTotalStats(pre);
      account_nodes.push_back(*it);
      pre = *it;
    }
  }
  std::reverse(std::begin(account_nodes), std::end(account_nodes));
  if (pre) {
    root_->AggregateTotalStats(pre);
  }

  // Perform the display and optionally redo accounting.
  int64 depth = 0;
  std::vector<OpNode*> show_nodes;
  int64 start = SearchRoot(account_nodes, opts.start_name_regexes);
  for (int64 i = start; i < account_nodes.size(); ++i, ++depth) {
    OpNode* n = account_nodes[i];
    if (ShouldTrim(n, opts.trim_name_regexes) || depth > opts.max_depth) {
      break;
    }
    n->show = ShouldShow(n, opts, depth);
    if (n->show) show_nodes.push_back(n);
  }

  pre = nullptr;
  for (auto it = show_nodes.rbegin(); it != show_nodes.rend(); ++it) {
    if (opts.account_displayed_op_only) {
      (*it)->ResetTotalStats();
      (*it)->AddSelfToTotalStats();
      if (pre) (*it)->AggregateTotalStats(pre);
    }
    if (pre) {
      (*it)->mutable_proto()->add_children()->MergeFrom(pre->proto());
      pre->mutable_proto()->clear_children();
    }
    pre = *it;
  }
  if (opts.account_displayed_op_only) {
    root_->ResetTotalStats();
    if (pre) {
      root_->AggregateTotalStats(pre);
      root_->mutable_proto()->add_children()->MergeFrom(pre->proto());
      pre->mutable_proto()->clear_children();
    }
  }

  if (opts.output_type == kOutput[1] || opts.output_type == kOutput[2]) {
    string display_str = FormatLegend(opts);
    for (OpNode* node : show_nodes) {
      display_str += FormatNode(node, root_.get(), opts);
    }
    root_->formatted_str = display_str;
  }
  return root_.get();
}

int64 TFOp::SearchRoot(const std::vector<OpNode*> nodes,
                       const std::vector<string>& regexes) {
  if (regexes.empty() || (regexes.size() == 1 && regexes[0] == ".*")) {
    return 0;
  }
  int64 i = 0;
  for (; i < nodes.size(); ++i) {
    for (const string& regex : regexes) {
      if (RE2::FullMatch(nodes[i]->name(), regex)) {
        return i;
      }
    }
  }
  return i;
}

string TFOp::FormatNode(OpNode* node, OpNode* root, const Options& opts) {
  std::vector<string> attrs;

  if (opts.select.find(kShown[0]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().requested_bytes() > 0) {
      accu_pct = 100.0 * node->proto().total_requested_bytes() /
          root->proto().total_requested_bytes();
      pct = 100.0 * node->proto().requested_bytes() /
          root->proto().total_requested_bytes();
    }
    attrs.push_back(strings::Printf(
        "%30s",
        strings::Printf("%s (%.2f%%, %.2f%%)",
                        FormatMemory(node->proto().requested_bytes()).c_str(),
                        accu_pct, pct)
            .c_str()));
  }

  if (opts.select.find(kShown[1]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().total_exec_micros() > 0) {
      accu_pct = 100.0 * node->proto().total_exec_micros() /
          root->proto().total_exec_micros();
      pct = 100.0 * node->proto().exec_micros() /
          root->proto().total_exec_micros();
    }

    attrs.push_back(strings::Printf(
        "%30s", strings::Printf("%s (%.2f%%, %.2f%%)",
                                FormatTime(node->proto().exec_micros()).c_str(),
                                accu_pct, pct)
                    .c_str()));
  }

  if (opts.select.find(kShown[2]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().total_parameters() > 0) {
      accu_pct = 100.0 * node->proto().total_parameters() /
          root->proto().total_parameters();
      pct = 100.0 * node->proto().parameters() /
          root->proto().total_parameters();
    }
    attrs.push_back(strings::Printf(
        "%30s",
        strings::Printf("%s params (%.2f%%, %.2f%%)",
                        FormatNumber(node->proto().parameters()).c_str(),
                        accu_pct, pct)
            .c_str()));
  }

  if (opts.select.find(kShown[3]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().total_float_ops() > 0) {
      accu_pct = 100.0 * node->proto().total_float_ops() /
          root->proto().total_float_ops();
      pct = 100.0 * node->proto().float_ops() /
          root->proto().total_float_ops();
    }

    attrs.push_back(strings::Printf(
        "%30s", strings::Printf("%s float_ops (%.2f%%, %.2f%%)",
                                FormatNumber(node->proto().float_ops()).c_str(),
                                accu_pct, pct)
                    .c_str()));
  }

  if (opts.select.find(kShown[5]) != opts.select.end()) {
    attrs.push_back(str_util::Join(node->node->devices(), "|"));
  }

  if (opts.select.find(kShown[6]) != opts.select.end()) {
    std::set<string> op_types = node->node->op_types();
    attrs.push_back(str_util::Join(op_types, "|"));
  }

  if (opts.select.find(kShown[7]) != opts.select.end()) {
    attrs.push_back(strings::Printf(
        "%10s",
        strings::Printf("%d", node->proto().graph_nodes_size()).c_str()));
  }

  string node_str = strings::Printf("%-25s%s\n", node->name().c_str(),
                                    str_util::Join(attrs, ", ").c_str());

  if (opts.select.find(kShown[8]) != opts.select.end()) {
    string input_shape_str = FormatInputShapes(node->proto());
    if (!input_shape_str.empty()) {
      node_str = strings::Printf("%s\n%s\n\n", node_str.c_str(),
                                 input_shape_str.c_str());
    }
  }
  return node_str;
}
}  // namespace tfprof
}  // namespace tensorflow
