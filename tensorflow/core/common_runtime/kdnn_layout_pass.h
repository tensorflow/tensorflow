// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_KDNN_LAYOUT_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_KDNN_LAYOUT_PASS_H_

#ifdef KDNN_ENABLED

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Forward declarations
class GraphOptimizationPassOptions;
class Node;
class NodeBuilder;

// KDNN canonicalized strings struct
struct KdnnConstStringsInfo {
  std::string sigmoid;
};

extern KdnnConstStringsInfo kdnn_csinfo_;

struct KdnnRewriteInfo {
  std::string op_name;
  std::string new_op_name;
  std::function<void(const Node*, NodeBuilder*, bool)> copy_attrs;
  std::function<bool(const Node*)> should_rewrite;
  std::string rewrite_cause;
};

struct KdnnWorkSpaceInfo {
  std::string op_name;
  int input_slot;
  int output_slot;
};

struct KdnnMergeInfo {
  std::string op1;
  std::string op2;
  std::string merged_op;
};

struct KdnnFusionInfo {
  std::string op1;
  std::string op2;
  std::string fused_op;
};

// KDNN Layout Rewrite Pass class declaration
class KdnnLayoutRewritePass : public GraphOptimizationPass {
 public:
  KdnnLayoutRewritePass();
  Status Run(const GraphOptimizationPassOptions& options) override;
  bool RunPass(std::unique_ptr<Graph>* g);

 private:
  static void CopyAttrsAll(const Node* old_node, NodeBuilder* nb, bool change_format);
  static bool SigmoidRewrite(const Node* n);

  std::vector<KdnnRewriteInfo> rinfo_;
  std::vector<KdnnWorkSpaceInfo> wsinfo_;
  std::vector<KdnnMergeInfo> minfo_;
  std::vector<KdnnFusionInfo> finfo_;
  int num_intra_threads_;
};

}  // namespace tensorflow

#endif  // KDNN_ENABLED

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_KDNN_LAYOUT_PASS_H_