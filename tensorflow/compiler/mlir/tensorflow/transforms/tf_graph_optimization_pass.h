/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Create a module pass that will execute the given TF GraphOptimization passes
// in sequence.
// Pass requires that the module ran on is convertible to TF Graph.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTensorFlowGraphOptimizationPass(
    std::vector<tensorflow::GraphOptimizationPass*> tf_passes);

// Same as above but pass names instead of the passes provided. The registered
// passes are queried, if a TF graph optimization pass is not found in registry
// then the pass fails.
// Pass requires that the module ran on is convertible to TF Graph.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTensorFlowGraphOptimizationPass(
    const std::vector<std::string>& pass_names);

// Register the pass for command line testing.
void RegisterGraphOptimizationPasses();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_GRAPH_OPTIMIZATION_PASS_H_
