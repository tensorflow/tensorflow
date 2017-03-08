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

// Build a tree structure based on the TensorFlow op names.
// For example, 'name1/name2' is a child of 'name1'.
// Stats are aggregated from descendants from ancestors.

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_SCOPE_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_SCOPE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tools/tfprof/internal/tfprof_node.h"
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/internal/tfprof_show.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/tools/tfprof/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class ScopeNode : public ShowNode {
 public:
  explicit ScopeNode(TFNode* node) : ShowNode(node) {}
  ~ScopeNode() override {}

  void AggregateTotalStats(ScopeNode* node) {
    ShowNode::AggregateTotalStats(node);
  }

  void AddSelfToTotalStats() { ShowNode::AddSelfToTotalStats(); }

  void ResetTotalStats() { ShowNode::ResetTotalStats(); }

  std::vector<ScopeNode*> children;
};

class TFScope : public TFShow {
 public:
  explicit TFScope(checkpoint::CheckpointReader* ckpt_reader)
      : TFShow(ckpt_reader) {}
  ~TFScope() override {}

  void AddNode(TFNode* node) override;

  void Build() override;

 private:
  const ShowNode* ShowInternal(const Options& opts) override;

  ScopeNode* CreateParentNode(const string& name);

  std::vector<ScopeNode*> SearchRoot(std::vector<ScopeNode*> roots,
                                     const std::vector<string>& regexes);

  std::vector<ScopeNode*> PrintScope(const std::vector<ScopeNode*> roots,
                                     const Options& opts, int depth,
                                     int last_ident);

  void Account(const std::vector<ScopeNode*>& roots, const Options& opts);

  std::vector<ScopeNode*> roots_;
  std::vector<std::unique_ptr<NodeDef>> node_defs_;
  std::map<string, std::unique_ptr<TFNode>> parent_nodes_;
  std::map<string, std::unique_ptr<ScopeNode>> nodes_map_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_SCOPE_H_
