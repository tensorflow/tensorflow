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

// Node classes used for different views. They are wrappers with "show"
// methods.
//
// ScopeNode is for scope view. GraphNode is for graph view, CodeNode
// is for code view and OpNode for op view.
// ScopeNode and GraphNode each maps to one TFGraphNode.
// CodeNode and OpNode each maps to one TFMultiGraphNode.

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

  void ReInit(int64 step);

  void AggregateTotalStats(ShowNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  const TFGraphNode* node;
  bool account;
  string formatted_str;

 protected:
  TFGraphNodeProto proto_;
};

class GraphNode : public ShowNode {
 public:
  explicit GraphNode(TFGraphNode* node) : ShowNode(node) {}

  bool Trackable(int64 step) const { return node->trackable(step); }

  std::vector<GraphNode*> children;
  std::vector<GraphNode*> show_children;
};

class ScopeNode : public ShowNode {
 public:
  explicit ScopeNode(const TFGraphNode* node) : ShowNode(node) {}
  ~ScopeNode() override {}

  std::vector<ScopeNode*> children;
  std::vector<ScopeNode*> show_children;
};

class ShowMultiNode {
 public:
  explicit ShowMultiNode(TFMultiGraphNode* node);
  virtual ~ShowMultiNode() {}

  bool ReInit(int64 step, const std::vector<string>& type_regexes);

  const string& name() const { return node->name(); }
  TFMultiGraphNodeProto* mutable_proto();
  const TFMultiGraphNodeProto& proto() const;

  void AggregateTotalStats(ShowMultiNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  TFMultiGraphNode* node;
  bool account;
  bool show;
  string formatted_str;

 protected:
  TFMultiGraphNodeProto proto_;
};

class CodeNode : public ShowMultiNode {
 public:
  explicit CodeNode(TFMultiGraphNode* node) : ShowMultiNode(node) {}
  ~CodeNode() override {}

  std::vector<CodeNode*> children;
  std::vector<CodeNode*> show_children;
};

class OpNode : public ShowMultiNode {
 public:
  explicit OpNode(TFMultiGraphNode* node) : ShowMultiNode(node) {}
  ~OpNode() override {}
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_SHOW_H_
