/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// passes to work with op attributes.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_QUANTIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_QUANTIZATION_UTILS_H_

#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir

namespace mlir {
namespace TFL {

// A generic rewrite pattern which matches any N-in-1-out operations with
// quantization parameters propagated to all the operands and results values.
// The quantization parameters are annotated by the Q/DQ op pairs. Each matched
// pattern are rewritten by its quantized alternatives.
//
// This pattern assumes all the matched ops are quantizable. This assumption is
// always right, except when a "Q" op is used as a requantize op. For non-"Q"
// ops, quantization parameters should be propagated to their result.
//
// This pattern only matches ops which only have one result.
template <typename Q, typename DQ>
struct GenericFullQuantizationPattern : public RewritePattern {
  explicit GenericFullQuantizationPattern(MLIRContext* context)
      : RewritePattern(Q::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation* op,
                                     PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return matchFailure();
    }
    auto quantize_op = cast<Q>(op);
    auto quantized_op = quantize_op.input()->getDefiningOp();
    // If it is a block argument, requantize op, or has more than one result, we
    // shouldn't rewrite this op.
    if (!quantized_op || llvm::isa<Q>(quantized_op) ||
        llvm::isa<DQ>(quantized_op) || quantized_op->getNumResults() != 1) {
      return matchFailure();
    }

    // Collect all the quantized inputs and "clone" the matched op by these
    // inputs.
    SmallVector<Value*, 4> inputs;
    inputs.reserve(quantized_op->getNumOperands());
    for (int i = 0, e = quantized_op->getNumOperands(); i != e; ++i) {
      auto* operand = quantized_op->getOperand(i);
      if (auto op_inst = dyn_cast_or_null<DQ>(operand->getDefiningOp())) {
        inputs.push_back(op_inst.input());
      } else {
        return matchFailure();
      }
    }
    // Use OpBuilder so we can use op name to create the new op.
    OpBuilder builder(quantized_op);
    OperationState new_state(
        quantized_op->getLoc(), quantized_op->getName().getStringRef(), inputs,
        op->getResult(0)->getType(), quantized_op->getAttrs());
    Operation* new_op = builder.createOperation(new_state);
    rewriter.replaceOp(op, {new_op->getResult(0)});
    return matchSuccess();
  }
};

// Converts the min/max/storage_type/narrow_range information to a
// QuantizedType, and then returns the attribute containing the QuantizedType.
TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, FloatAttr min,
                              FloatAttr max, Type storage_type,
                              bool narrow_range = false);

// Converts the min/max/num_bits/narrow_range information to a
// QuantizedType, and then returns the attribute containing the QuantizedType.
TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, IntegerAttr num_bits,
                              BoolAttr narrow_range);

// Quantizes the elements in the attribute `real_value` by the quantization
// parameters in `tensor_type`. Returns empty Attribute if the
// `tensor_type` is not a QuantizedType or the quantization fails.
ElementsAttr Quantize(Attribute real_value, Type tensor_type);

// Returns the quantized type for an element attribute. The quantization
// parameters in this type is based on the min and max element of the attribute.
// When the elements in the `attr` are not in floating-point, or the value range
// isn't straddling zero, an empty type is returned.
Type GetUniformQuantizedTypeForElementsAttr(ElementsAttr attr,
                                            unsigned storage_type_width,
                                            bool narrow_range = false);

// Returns the quantized type of a bias input, given the quantized types of
// other operands which are multiply-accumulated (the bias is added to the
// accumulated value).
quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types);

// Propagates quantization parameters across ops in this function and satisfy
// the quantization specification of the ops. This methods assumes the initial
// quantization parameters are stored as adjacent quantize and dequantize ops
// and the propagation results are materialized by inserting pairs of quantize
// and dequantize ops to this function.
void ApplyQuantizationParamsPropagation(mlir::FuncOp func);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_QUANTIZATION_UTILS_H_
