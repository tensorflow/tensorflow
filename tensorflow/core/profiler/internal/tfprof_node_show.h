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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_SHOW_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_SHOW_H_

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class ShowNode {
 public:
  explicit ShowNode(const TFGraphNode* node);
  virtual ~ShowNode() {}

  const string& name() const { return node->name(); }
  GraphNodeProto* mutable_proto();
  const GraphNodeProto& proto() const;

  void ReInit(int64_t step);

  void AggregateTotalStats(ShowNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  const TFGraphNode* node;
  bool account;
  string formatted_str;

 protected:
  GraphNodeProto proto_;
};

class GraphNode : public ShowNode {
 public:
  explicit GraphNode(TFGraphNode* node) : ShowNode(node) {}

  bool Trackable(int64_t step) const { return node->trackable(step); }

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

  bool ReInit(int64_t step, const std::vector<string>& type_regexes);

  const string& name() const { return node->name(); }
  MultiGraphNodeProto* mutable_proto();
  const MultiGraphNodeProto& proto() const;

  void AggregateTotalStats(ShowMultiNode* node);

  void AddSelfToTotalStats();

  void ResetTotalStats();

  TFMultiGraphNode* node;
  bool account;
  bool show;
  string formatted_str;

 protected:
  MultiGraphNodeProto proto_;
};

class CodeNode : public ShowMultiNode {
 public:
  CodeNode(TFMultiGraphNode* node, const CallStack::Trace* trace,
           const string& suffix)
      : ShowMultiNode(node), trace_(trace), suffix_(suffix) {}
  ~CodeNode() override {}

  CodeNode* AddChildren(const string& name, const CallStack::Trace* trace,
                        const string suffix) {
    auto it = children_.find(name);
    if (it != children_.end()) {
      return it->second.get();
    }

    graph_children_.push_back(
        std::unique_ptr<TFMultiGraphNode>(new TFMultiGraphNode(name)));
    auto child = &children_[name];
    child->reset(new CodeNode(graph_children_.back().get(), trace, suffix));
    children.push_back(child->get());
    return child->get();
  }

  bool has_trace() const { return trace_ != nullptr; }
  const int32 lineno() const { return trace_->lineno(); }
  string file() const { return trace_->file(); }
  string function() const { return trace_->function() + suffix_; }
  int32 func_start_line() const { return trace_->func_start_line(); }

  std::vector<CodeNode*> children;
  std::vector<CodeNode*> show_children;

 private:
  const CallStack::Trace* trace_;
  string suffix_;
  std::vector<std::unique_ptr<TFMultiGraphNode>> graph_children_;
  std::map<string, std::unique_ptr<CodeNode>> children_;
};

class OpNode : public ShowMultiNode {
 public:
  explicit OpNode(TFMultiGraphNode* node) : ShowMultiNode(node) {}
  ~OpNode() override {}
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_SHOW_H_
