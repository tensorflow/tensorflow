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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {

// Returns true if 'op' is supported to run on 'hardware'.
bool IsSupported(Operation* op, const std::string& hardware);

// Return proper rewriter patterns for different hardwares.
RewritePatternSet GetHardwareRewritePatterns(MLIRContext* context,
                                             const std::string& hardware);

// Convert quantized ops to float, this will essentially insert dequantize &
// quantize pair around the op.
void ConvertQuantizedOpToFloat(func::FuncOp func, OpBuilder* builder);

// This will optimize the quantized ops -> float graph.
void OptimizeQuantizedOpToFloat(func::FuncOp func, MLIRContext* context);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_H_
