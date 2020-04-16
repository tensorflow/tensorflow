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
};

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ARITHMETIC_COUNT_UTIL_H_
