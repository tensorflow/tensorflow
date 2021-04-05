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

// This header file defines common utils used by TFLite transformation
// passes to work with tf.FakeQuant* ops.
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace TFL {

// Removes the wrapper of the tf.FakeQuant* ops.
LogicalResult ConvertFakeQuantOps(FuncOp func, MLIRContext *ctx);

// Returns the names of all the considered tf.FakeQuant* ops.
std::vector<std::string> AllTfFakeQuantOps();

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_
