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
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/encapsulate_tpu_computations_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/variable_merger_pass.h"

namespace tensorflow {
namespace {

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
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      VariableMergerPass);
}  // namespace
}  // namespace tensorflow
