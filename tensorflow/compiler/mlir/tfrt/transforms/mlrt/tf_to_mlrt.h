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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TF_TO_MLRT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TF_TO_MLRT_H_
#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

namespace tensorflow {
namespace mlrt_compiler {

// The conversion pass that is run before 'tf-mlrt-parallelization' passes. The
// parallelization pass changes the graph content, so any rewrite/conversion
// that depends on the graph instead of individual ops should be done before
// parallelization.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtPreParallelizationConversionPass(
    const TfrtPipelineOptions& options);

// The conversion pass that is run after 'tf-mlrt-parallelization' passes. The
// parallelization pass changes the graph content, so this pass should only
// contain conversion that depends on individual ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtConversionPass(const TfrtPipelineOptions& options);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtConversionPass(const TfrtPipelineOptions& options,
                             const tfrt_stub::FallbackState* fallback_state);

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TF_TO_MLRT_H_
