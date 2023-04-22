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

#include <memory>

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

static function_optimization_registration::FunctionOptimizationPassRegistration
    register_mlir_passes(std::make_unique<MlirFunctionOptimizationPass>());

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      MlirV1CompatGraphOptimizationPass);

}  // namespace tensorflow
