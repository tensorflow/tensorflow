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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_PARALLELIZATION_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_PARALLELIZATION_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"

namespace tensorflow {
namespace mlrt_compiler {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateParallelizationPass(
    uint64_t cost_threshold, bool merge_inter_dependent_streams,
    const tfrt_stub::CostRecorder* cost_recorder = nullptr);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateParallelizationPass();

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_PARALLELIZATION_H_
