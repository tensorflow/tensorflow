/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONST_OP_SIZE_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONST_OP_SIZE_H_

#include <cstdint>

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

// Returns the size in bytes of the underlying data of `const_op`. If the
// underlying type's size cannot be determined, it assumes 4 bytes per element.
int64_t GetSizeInBytes(TF::ConstOp const_op);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONST_OP_SIZE_H_
