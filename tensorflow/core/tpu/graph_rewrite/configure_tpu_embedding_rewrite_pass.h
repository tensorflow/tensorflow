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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_

#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// TODO(shizhiw): Clean up embedding related code from
//  distributed_tpu_configuration_rewrite_pass.cc.
// Replaces dummy ConfigureTPUEmbedding Ops assigned to TPU_SYSTEM
// devices with nodes which will set up TPU Embedding.
class ConfigureTPUEmbeddingRewritePass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_CONFIGURE_TPU_EMBEDDING_REWRITE_PASS_H_
