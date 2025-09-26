/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::TFL {

inline constexpr char kPropagatedQuantizeOpAttr[] = "propagated";

std::optional<quant::QuantizedType> GetPropagatedType(
    SameScalesOpInterface same_scales_op);

// If `value` is the result of a DequantizeOp, returns the quantized type of the
// DequantizeOp's input. Otherwise, returns std::nullopt.
// The IR pattern looks like:
// ... -> [quantized type] -> DequantizeOp -> [value]
// Otherwise, returns std::nullopt.
std::optional<quant::QuantizedType> GetQTypeFromDefiningDequantize(
    mlir::Value value);

// If `value` has only one use and that use is a QuantizeOp, returns the
// quantized type of the QuantizeOp's result. Otherwise, returns std::nullopt.
// The single-use check is to avoid ambiguity in cases of fan-out.
// The IR pattern looks like:
// [value] -> QuantizeOp -> ...
std::optional<quant::QuantizedType> GetQTypeFromConsumingQuantize(
    mlir::Value value);

// Inserts a Quantize-Dequantize (QDQ) pair for a value.
// If `target_op` is provided, it only replaces the uses of `value` within
// `target_op`. Otherwise, it replaces all uses of `value` (except for the
// newly created Quantize op).
LogicalResult InsertQDQ(mlir::Value value, quant::QuantizedType qtype,
                        PatternRewriter& rewriter,
                        mlir::Operation* target_op = nullptr);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_QUANTIZATION_QUANT_UTILS_H_
