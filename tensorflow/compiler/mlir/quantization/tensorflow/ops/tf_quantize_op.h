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

// This file provides a list of supported quantization algorithms in the format
// of "apply<Name of the Quantization Algorithm>Quantization".
// After applying the function, a quantize/dequantize functions are created
// where the body of each function contains a specific quantization algorithm.
// The input of the quantize function has one operand of
// IsValueWithQuantizablePrecision and the output is a tensor with supported
// quantized precision (like int8). For dequantize function, it is the other way
// around.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_QUANTIZE_OP_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_QUANTIZE_OP_H_

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

std::optional<TF::PartitionedCallOp> ApplyUniformQuantization(
    PatternRewriter& rewriter, TF::ConstOp op,
    tensorflow::quantization::QuantizationComponentSpec& weight_spec);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_OPS_TF_QUANTIZE_OP_H_
