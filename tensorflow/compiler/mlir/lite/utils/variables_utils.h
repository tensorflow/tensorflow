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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VARIABLES_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VARIABLES_UTILS_H_

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace utils {

// Returns true if 'op' has type that is supported by native TFLite
// variables.
bool IsSupportedVariableType(Operation* op);

// Returns true if 'type' is supported by native tflite variables.
bool IsSupportedVariableType(ShapedType type);

}  // namespace utils
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VARIABLES_UTILS_H_
