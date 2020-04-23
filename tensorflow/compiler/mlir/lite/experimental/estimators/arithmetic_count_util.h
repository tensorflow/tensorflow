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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ARITHMETIC_COUNT_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ARITHMETIC_COUNT_UTIL_H_

// For add/mul/div/sub and other broadcastable ops.
class ArithmeticCountUtilHelper {
 public:
  static bool GetArithmeticCountForBroadcastableOp(mlir::Operation* op,
                                                   int64_t* count) {
    auto output = op->getResult(0);
    auto output_type = output.getType().dyn_cast_or_null<RankedTensorType>();
    if (!output_type || !output_type.hasStaticShape()) return false;

    *count = output_type.getNumElements();
    return true;
  }

  static bool GetInputTensorTotalSize(mlir::Operation* op, int64_t* count) {
    int64_t total_count = 0;
    for (auto input : op->getOperands()) {
      auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
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
  static bool GetArithmeticCountForConvAndFullyconnectedOp(Operation* op,
                                                           int64_t* count) {
    auto weight = op->getOperand(1);
    auto weight_type = weight.getType().dyn_cast_or_null<RankedTensorType>();
    if (weight_type == nullptr || !weight_type.hasStaticShape()) return false;

    auto output = op->getResult(0);
    auto output_type = output.getType().dyn_cast_or_null<RankedTensorType>();
    if (output_type == nullptr || !output_type.hasStaticShape()) return false;

    int64_t cols = 1;
    for (int i = 0; i < output_type.getRank() - 1; ++i) {
      cols *= output_type.getDimSize(i);
    }
    const int64_t cost_per_col = 2 * weight_type.getNumElements();

    *count = 2 * cost_per_col * cols;

    auto bias = op->getOperand(2);
    if (bias) {
      auto bias_type = bias.getType().dyn_cast_or_null<RankedTensorType>();
      if (bias_type && bias_type.hasStaticShape()) {
        *count += bias_type.getNumElements();
      }
    }

    return true;
  }
};

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ARITHMETIC_COUNT_UTIL_H_
