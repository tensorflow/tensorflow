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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ARITHMETIC_COUNT_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ARITHMETIC_COUNT_UTIL_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// For add/mul/div/sub and other broadcastable ops.
class ArithmeticCountUtilHelper {
 public:
  static bool GetFirstOutputCount(mlir::Operation* op, int64_t* count) {
    auto output = op->getResult(0);
    auto output_type =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(output.getType());
    if (!output_type || !output_type.hasStaticShape()) return false;

    *count = output_type.getNumElements();
    return true;
  }

  static bool GetInputTensorTotalSize(mlir::Operation* op, int64_t* count) {
    int64_t total_count = 0;
    for (auto input : op->getOperands()) {
      auto input_type =
          mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
      if (!input_type || !input_type.hasStaticShape()) {
        return false;
      }
      total_count += input_type.getNumElements();
    }
    *count = total_count;
    return true;
  }

  // For conv2d/depthwise_conv/fully_connected ops.
  // This algorithm actually comes from TOCO tooling_util.cc
  static bool GetArithmeticCountForConvAndFullyconnectedOp(mlir::Operation* op,
                                                           int64_t* count) {
    auto weight = op->getOperand(1);
    auto weight_type =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(weight.getType());
    if (weight_type == nullptr || !weight_type.hasStaticShape()) return false;

    auto output = op->getResult(0);
    auto output_type =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(output.getType());
    if (output_type == nullptr || !output_type.hasStaticShape()) return false;

    int64_t cols = 1;
    for (int i = 0; i < output_type.getRank() - 1; ++i) {
      cols *= output_type.getDimSize(i);
    }
    const int64_t cost_per_col = 2 * weight_type.getNumElements();

    *count = cost_per_col * cols;

    auto bias = op->getOperand(2);
    if (bias) {
      auto bias_type =
          mlir::dyn_cast_or_null<mlir::RankedTensorType>(bias.getType());
      if (bias_type && bias_type.hasStaticShape()) {
        *count += output_type.getNumElements();
      }
    }

    return true;
  }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ARITHMETIC_COUNT_UTIL_H_
