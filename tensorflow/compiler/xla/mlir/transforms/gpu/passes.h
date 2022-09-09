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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_PASSES_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_PASSES_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// Auxiliary passes for lowering to XLA Gpu runtime.
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertMemrefGetGlobalToArgPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertMemrefGetGlobalToArgPass(int64_t min_num_elements);

//===-----------------------------------------------------------------------===/
// Passes for lowering from the `gpu` dialect.
//===-----------------------------------------------------------------------===/

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertGpuToGpuRuntimePass();

//===----------------------------------------------------------------------===//
// Passes for lowering from the `lmhlo` dialect.
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToGpuRuntimePass();

//===----------------------------------------------------------------------===//
// Passes for lowering from the `lmhlo_gpu` dialect.
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloGpuToGpuRuntimePass();

//===-----------------------------------------------------------------------===/

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h.inc"

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_PASSES_H_
