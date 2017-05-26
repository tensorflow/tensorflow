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

// Build a tree structure based on the TensorFlow model's python code stacks.
// Stats are aggregated from descendants from ancestors.

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_CODE_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_CODE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tools/tfprof/internal/tfprof_node.h"
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/internal/tfprof_show_multi.h"
#include "tensorflow/tools/tfprof/internal/tfprof_timeline.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/tools/tfprof/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class TFCode : public TFMultiShow {
 public:
  explicit TFCode() : code_root_(nullptr), trace_root_(nullptr) {}
  ~TFCode() override {}

  void AddNode(TFGraphNode* node) override;

  void Build() override;

 private:
  CodeNode* BuildCodeNodes(TFMultiGraphNode* root);

  const ShowMultiNode* ShowInternal(const Options& opts,
                                    Timeline* timeline) override;

  std::vector<CodeNode*> SearchRoot(std::vector<CodeNode*> roots,
                                    const std::vector<string>& regexes);

  std::vector<CodeNode*> PrintScope(const std::vector<CodeNode*> roots,
                                    const Options& opts, int depth,
                                    int last_ident);

  std::vector<CodeNode*> Account(const std::vector<CodeNode*>& roots,
                                 const Options& opts);

  void Format(const std::vector<CodeNode*> roots, string* display_str,
              TFMultiGraphNodeProto* proto);

  string FormatNode(CodeNode* node, const Options& opts, int64 indent);

  std::unique_ptr<CodeNode> root_;
  CodeNode* code_root_;
  std::unique_ptr<TFMultiGraphNode> trace_root_;
  std::unique_ptr<TFMultiGraphNode> tfprof_trace_root_;
  std::set<std::unique_ptr<CodeNode>> code_nodes_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_CODE_H_
