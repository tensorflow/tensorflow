/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Rewrites the graph and function defs in the specified
// GraphOptimizationPassOptions object for software deduplication.
//
// For the graph, groups the RecvTPUEmbeddingActivations and
// SendTPUEmbeddingGradients nodes by their _tpu_replicate attribute. For each
// such group:
// 1. Inserts a XlaRecvTPUEmbeddingDeduplicationData node into the graph.
// 2. Replaces the public RecvTPUEmbeddingActivations node (if present) with the
//    internal XlaRecvTPUEmbeddingActivations node.
// 3. Replaces the public SendTPUEmbeddingGradients node (if present) with the
//    internal XlaSendTPUEmbeddingGradients node.
// 4. Connects the outputs of the XlaRecvTPUEmbeddingDeduplicationData node with
//    the inputs of the XlaRecvTPUEmbeddingActivations and
//    XlaSendTPUEmbeddingGradients nodes.
//
// Iterates through the list of functions in the specified
// GraphOptimizationPassOptions object. Performs the same steps 1-4 specified
// above for each function.
//
// If multiple RecvTPUEmbeddingActivations nodes or SendTPUEmbeddingGradients
// nodes are present in the same function or in the same _tpu_replicate group,
// an InvalidArgument error is returned to the caller.
class TPUEmbeddingSoftwareDeduplicationRewritePass :
    public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_SOFTWARE_DEDUPLICATION_REWRITE_PASS_H_
