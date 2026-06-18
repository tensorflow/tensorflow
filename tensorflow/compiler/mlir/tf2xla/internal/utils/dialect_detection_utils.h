/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_DIALECT_DETECTION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_DIALECT_DETECTION_UTILS_H_

#include "mlir/IR/Operation.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Returns true if the op has a valid namespace during clustering & tf dialect
// to executor components of the Bridge.
bool IsInBridgeAcceptableDialects(mlir::Operation* op);

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_DIALECT_DETECTION_UTILS_H_
