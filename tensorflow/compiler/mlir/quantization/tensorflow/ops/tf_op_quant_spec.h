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
// Functions for quantization specifications of TensorFlow ops.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_OP_QUANT_SPEC_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_OP_QUANT_SPEC_H_

#include <memory>
#include <optional>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir {
namespace quant {

// Check if the op has data movement trait. Ops with this trait do not perform
// any computations but just move data and has one result operand.
bool IsOpWithDataMovementTrait(Operation* op);

// Check if the op is quantizable. Currently, the scope of quantizable op is
// limited to compute intense operations and the ops that supports integer
// operands.
bool IsOpWithQuantizableTrait(Operation* op);

// Check if the op's operand accepts int8 type.
bool IsOpWithInt8TypeOperand(Operation* op);

// Check if the data is in quantizable precision. Currently, a value in f32 or
// bf16 is quantizable.
bool IsValueWithQuantizablePrecision(Value val);

std::optional<tensorflow::quantization::QuantizationComponentSpec>
GetWeightComponentSpec(
    const tensorflow::quantization::QuantizationOptions& quantization_options);

// Returns the spec for the given operation that can be used for both of
// dynamic and static range quantization.
std::unique_ptr<OpQuantSpec> GetTFOpQuantSpec(Operation* op);

// Returns quantization scale specs (fixed output, same scale) for a TF op.
std::unique_ptr<OpQuantScaleSpec> GetTfQuantScaleSpec(Operation* op);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_OP_QUANT_SPEC_H_
