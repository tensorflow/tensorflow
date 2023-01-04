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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COMBINE_TPU_EMBEDDING_LOAD_RETRIEVE_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COMBINE_TPU_EMBEDDING_LOAD_RETRIEVE_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// Merges per-table TPUEmbedding load and retrieve operators into global
// operators.
class CombineTPUEmbeddingLoadRetrievePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_COMBINE_TPU_EMBEDDING_LOAD_RETRIEVE_PASS_H_
