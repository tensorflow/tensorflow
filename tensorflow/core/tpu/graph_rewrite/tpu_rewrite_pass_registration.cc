/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/tpu/graph_rewrite/combine_tpu_embedding_load_retrieve_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/encapsulate_tpu_computations_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/tpu_embedding_software_deduplication_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/update_tpu_embedding_ops_passes.h"
#include "tensorflow/core/tpu/graph_rewrite/variable_merger_pass.h"

namespace tensorflow {
namespace {

// PRE_PLACEMENT passes:
// The earlier this occurs, the better. Otherwise we do updates to the same ops
// in FunctionDefs and the graph.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      UpdateTPUEmbeddingModePass);
// This pass uses the TPUEmbeddingConfiguration from the
// RecvTPUEmbeddingActivations or the SendTPUEmbeddingGradients ops.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 2,
                      TPUEmbeddingSoftwareDeduplicationRewritePass);
// CombineTPUEmbeddingLoadRetrievePass pass needs the TPUEmbeddingConfiguration
// in ConfigureDistributedTPU, which is removed by
// DistributedTPUConfigurationRewritePass.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 11,
                      CombineTPUEmbeddingLoadRetrievePass);
// This pass removes the TPUEmbeddingConfiguration in ConfigureDistributedTPU.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 20,
                      DistributedTPUConfigurationRewritePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 20,
                      DistributedTPUShutdownRewritePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 34,
                      EncapsulateTPUComputationsPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 39,
                      ExtractOutsideCompilationPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 40,
                      DistributedTPURewritePass);
// Needs to occur after ExtractOutsideCompilationPass (currently phase 39).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 41,
                      UpdateTPUEmbeddingEnqueueOrdinalPass);

// POST_REWRITE_FOR_EXEC pass
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      VariableMergerPass);
}  // namespace
}  // namespace tensorflow
