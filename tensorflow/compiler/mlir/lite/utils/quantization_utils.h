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

#include <unordered_map>

#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir

namespace mlir {
namespace TFL {

using QuantParams = quant::QuantizedType;
using SignedInteger = std::pair<unsigned, unsigned>;  // bitwidth and sign
using QuantParamsForResults = llvm::SmallVector<QuantParams, 4>;
using AccumulatorScaleFunc =
    std::function<QuantParams(const std::vector<QuantParams>&)>;

// Quantization spec of an op, driving the quantization algorithm.
struct OpQuantSpec {
  // Whether the op has quantizable result. This flag is set to false if the op
  // has "TFL::NoQuantizableResult" trait.
  bool is_quantizable = true;

  // Whether it requires same inputs and result scale. This flag is set to true
  // if the op has "TFL::SameOperandsAndResultScale" trait.
  bool requires_same_scale = false;

  // Maps the operand index of a bias input to its quantization specifications,
  // including the non-bias operand indexes and the method retrieving
  // quantization parameters from list of parameters of the non-bias operands.
  // This map is empty if the op doesn't havea bias operand.
  std::unordered_map<int, std::pair<std::vector<int>, AccumulatorScaleFunc>>
      biases_params;

  // Quantization parameters for value restricted outputs. This is the
  // "hard-coded" parameters and should be used unconditionally for the
  // quantized op. This vector is empty if the op doesn't have value resctricted
  // outputs.
  llvm::DenseMap<SignedInteger, QuantParamsForResults> restricted_output_params;
};

// A function signature for getting the particular OpQuantSpec for the provided
// op.
typedef std::unique_ptr<OpQuantSpec> (*OpQuantSpecGetter)(Operation* op);

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
      auto operand_ele_type =
          operand->getType().template cast<TensorType>().getElementType();
      if (auto op_inst = dyn_cast_or_null<DQ>(operand->getDefiningOp())) {
        inputs.push_back(op_inst.input());
      } else if (operand_ele_type.template isa<IntegerType>()) {
        // If the operand is an integer tensor, then it doesn't require the
        // DQ op in the pattern.
        inputs.push_back(operand);
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
                              bool narrow_range = false,
                              bool is_signed = false);

// Converts the min/max/num_bits/narrow_range information to a
// QuantizedType, and then returns the attribute containing the QuantizedType.
// Note that this method assumes an unsigned quantization type, which is
// implicitly defined by FakeQuant* ops in TensorFlow.
TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, IntegerAttr num_bits,
                              BoolAttr narrow_range);

// Casts the `target` type to a quantized type by using the quantization
// parameters from the type in the `source` type attribute.
// Examples:
//   f32 -> !quant.uniform<i8:f32, 1.0>
//   tensor<4xf32> -> tensor<4x!quant.uniform<i8:f32, 1.0>>
// The result is wrapped by a type attribute. Returns nullptr if the cast isn't
// valid.
TypeAttr CastQuantizedTypeAttrFromExpressedType(Builder builder,
                                                TypeAttr source, Type target);

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
                                            bool is_sign, bool narrow_range);

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
void ApplyQuantizationParamsPropagation(mlir::FuncOp func, bool is_signed,
                                        OpQuantSpecGetter op_quant_spec_getter);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_QUANTIZATION_UTILS_H_
