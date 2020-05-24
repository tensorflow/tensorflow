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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ESTIMATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ESTIMATOR_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/estimators/hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h.inc"

template <typename Op, typename TargetHardware>
class TFLiteCostEstimator {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) {
    llvm::errs() << "No defined support for op: "
                 << op->getName().getStringRef().str();
    return false;
  }
};

// All ops on CPU are supported.
// TODO(karimnosseir): Only allow TFL ops in the "TFL_OP" param.
template <typename TFL_OP>
class TFLiteCostEstimator<TFL_OP, hardware::CPU> {
 public:
  // TODO(karimnosseir): Update and use table based method and lookup
  // cost from a loadable table ?
  static double GetCost(mlir::Operation* op) { return 0.0; }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_ESTIMATOR_H_
