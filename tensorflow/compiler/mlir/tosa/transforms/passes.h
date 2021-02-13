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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace tosa {

std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFPass();
std::unique_ptr<OperationPass<FuncOp>> createFuseBiasTFPass();
std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFLPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertTFLUint8Pass();

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H
