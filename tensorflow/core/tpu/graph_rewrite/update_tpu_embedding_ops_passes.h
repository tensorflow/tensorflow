/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Rewrites ConfigureTPUEmbedding Op into nodes which set up TPUEmbedding.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class UpdateTPUEmbeddingEnqueueOrdinalPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

class UpdateTPUEmbeddingModePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  static Status GetEnqueueOpsFromGraph(
      Graph* graph, absl::flat_hash_map<Node*, bool>* enqueue);
  static Status UpdateGraphEnqueueOp(bool training, Graph* graph,
                                     Node* enqueue);
  static Status GetEnqueueOpsFromFunctionDef(FunctionDef* function,
                                             std::map<int, bool>* enqueue);
  static Status UpdateFunctionDefEnqueueOp(int enqueue, bool training,
                                           FunctionDef* function,
                                           bool* updated);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_UPDATE_TPU_EMBEDDING_OPS_PASSES_H_
