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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_

#include <functional>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir {
namespace quant {

using OperationToName = std::function<llvm::StringRef(Operation* op)>;

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<func::FuncOp>> CreateImportQuantStatsPass(
    OperationToName op_to_name, const std::string& stats_str);

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateImportQuantStatsPassForTFControlDialect(const std::string& stats_str);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_
