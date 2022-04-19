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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TFL_TO_STD_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TFL_TO_STD_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// Converts all the tfl.quantize/tfl.dequantize ops to the ops in the mlir.quant
// dialect ones in the function.
void ConvertTFLQuantOpsToMlirQuantOps(func::FuncOp func);

// Converts all the mlir.quant dialect ops to the tfl.quantize/tfl.dequantize
// ops in the function.
void ConvertMlirQuantOpsToTFLQuantOps(func::FuncOp func);

// A helper class to convert target function to another representation using
// `ConvertForward` function during construction and convert target function
// back to the original representation using `ConvertBackward` function during
// deconstruction.
template <void (*ConvertForward)(func::FuncOp),
          void (*ConvertBackward)(func::FuncOp)>
class ScopedOpsConverter {
 public:
  explicit ScopedOpsConverter(func::FuncOp func) : func_(func) {
    ConvertForward(func_);
  }

  ScopedOpsConverter(const ScopedOpsConverter&) = delete;
  ScopedOpsConverter operator=(const ScopedOpsConverter&) = delete;
  ScopedOpsConverter(const ScopedOpsConverter&&) = delete;
  ScopedOpsConverter operator=(const ScopedOpsConverter&&) = delete;

  ~ScopedOpsConverter() { ConvertBackward(func_); }

 private:
  func::FuncOp func_;
};

using ScopedTFLQuantOpsToMlirQuantOpsConverter =
    ScopedOpsConverter<ConvertTFLQuantOpsToMlirQuantOps,
                       ConvertMlirQuantOpsToTFLQuantOps>;
using ScopedMlirQuantOpsToTFLQuantOpsConverter =
    ScopedOpsConverter<ConvertMlirQuantOpsToTFLQuantOps,
                       ConvertTFLQuantOpsToMlirQuantOps>;
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TFL_TO_STD_H_
