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

// Nodes used for different views.
// ScopeNode is for scope view. GraphNode is for graph view and CodeNode
// is for code view.

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_SHOW_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_SHOW_H_

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/tools/tfprof/internal/tfprof_node.h"
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/tools/tfprof/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class ShowNode {
 public:
  explicit ShowNode(const TFGraphNode* node);
  virtual ~ShowNode() {}

  const string& name() const { return node->name(); }
  TFGraphNodeProto* mutable_proto();
  const TFGraphNodeProto& proto() const;

  void ReInit();

  string Format(const Options& opts);

  string FormatMeta(const Options& opts);

  const TFGraphNode* node;
  bool account;
  string formatted_str;

 protected:
  void AggregateTotalStats(ShowNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  TFGraphNodeProto proto_;
};

class GraphNode : public ShowNode {
 public:
  explicit GraphNode(TFGraphNode* node) : ShowNode(node) {
    mutable_proto()->set_inputs(node->inputs().size());
    mutable_proto()->set_total_inputs(0);
    trackable = Trackable();
  }

  void ReInit() {
    ShowNode::ReInit();
    mutable_proto()->set_inputs(node->inputs().size());
    mutable_proto()->set_total_inputs(0);
  }

  void AggregateTotalStats(GraphNode* node) {
    ShowNode::AggregateTotalStats(node);
    mutable_proto()->set_total_inputs(proto().total_inputs() +
                                      node->proto().total_inputs() + 1);
  }

  void AddSelfToTotalStats() {
    ShowNode::AddSelfToTotalStats();
    mutable_proto()->set_total_inputs(proto().total_inputs() +
                                      proto().inputs());
  }

  void ResetTotalStats() {
    ShowNode::ResetTotalStats();
    mutable_proto()->set_total_inputs(0);
    show_children.clear();
  }

  bool Trackable() {
    if (!node->step_stats()) return false;
    if (node->all_start_micros() == 0) return false;
    if (node->canonical_device().empty() || node->host_device().empty()) {
      return false;
    }
    return true;
  }

  bool trackable;
  std::vector<GraphNode*> children;
  std::vector<GraphNode*> show_children;
};

class ScopeNode : public ShowNode {
 public:
  explicit ScopeNode(const TFGraphNode* node) : ShowNode(node) {}
  ~ScopeNode() override {}

  void ReInit() { ShowNode::ReInit(); }

  void AggregateTotalStats(ScopeNode* node) {
    ShowNode::AggregateTotalStats(node);
  }

  void AddSelfToTotalStats() { ShowNode::AddSelfToTotalStats(); }

  void ResetTotalStats() {
    ShowNode::ResetTotalStats();
    show_children.clear();
  }

  std::vector<ScopeNode*> children;
  std::vector<ScopeNode*> show_children;
};

class ShowCodeNode {
 public:
  explicit ShowCodeNode(const TFCodeNode* node);
  virtual ~ShowCodeNode() {}

  const string& name() const { return node->name(); }
  TFCodeNodeProto* mutable_proto();
  const TFCodeNodeProto& proto() const;

  string Format(const Options& opts);

  string FormatMeta(const Options& opts);

  const TFCodeNode* node;
  bool account;
  string formatted_str;

 protected:
  void AggregateTotalStats(ShowCodeNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  TFCodeNodeProto proto_;
};

class CodeNode : public ShowCodeNode {
 public:
  explicit CodeNode(const TFCodeNode* node) : ShowCodeNode(node) {}
  ~CodeNode() override {}

  void AggregateTotalStats(CodeNode* node) {
    ShowCodeNode::AggregateTotalStats(node);
  }

  void AddSelfToTotalStats() { ShowCodeNode::AddSelfToTotalStats(); }

  void ResetTotalStats() {
    ShowCodeNode::ResetTotalStats();
    show_children.clear();
  }

  std::vector<CodeNode*> children;
  std::vector<CodeNode*> show_children;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_SHOW_H_
