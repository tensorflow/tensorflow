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
// Functions for quantization specifications of Uniform Quantized ops.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_UNIFORM_OP_QUANT_SPEC_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_UNIFORM_OP_QUANT_SPEC_H_

#include <memory>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"

namespace mlir {
namespace quant {

// Returns the spec for the given operation that can be used for both of
// dynamic and static range quantization.
std::unique_ptr<tf_quant::OpQuantSpec> GetUniformOpQuantSpec(Operation* op);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_UNIFORM_OP_QUANT_SPEC_H_
