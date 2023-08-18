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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_LEGALIZE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_LEGALIZE_UTILS_H_

#include "mlir/IR/Builders.h"  // from @llvm-project


namespace mlir {
namespace tensor {

// If the tensor given as an input argument is unranked, emit a 'tensor.cast'
// op to cast it to a ranked tensor with the given number of dimensions. If the
// given tensor is already ranked, the same tensor is returned. The given value
// is expected to be of type tensor.
Value castUnrankedTensor(OpBuilder& builder, Location loc, Value tensor,
                         int rank);

// Process argument 'shape' to eliminate a possible occurrence of -1. If
// found, it is replaced with the total size of the 'input' tensor divided by
// all remaining components of 'shape'.
Value substituteShapeWildcard(OpBuilder& builder, Location loc, Value input,
                              Value shape);

}  // namespace tensor
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_LEGALIZE_UTILS_H_

