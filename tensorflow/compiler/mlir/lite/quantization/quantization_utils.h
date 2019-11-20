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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_

#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"

namespace mlir {
namespace TFL {

using QuantParams = quant::QuantizedType;
using SignedInteger = std::pair<unsigned, unsigned>;  // bitwidth and sign
using QuantParamsForResults = llvm::SmallVector<QuantParams, 4>;
using AccumulatorScaleFunc =
    std::function<QuantParams(const std::vector<QuantParams>&)>;

// Quantization spec of an op, driving the quantization algorithm.
struct OpQuantSpec {
  // Maps the operand index of a bias input to its quantization specifications,
  // including the non-bias operand indexes and the method retrieving
  // quantization parameters from list of parameters of the non-bias operands.
  // This map is empty if the op doesn't have a bias operand.
  std::unordered_map<int, std::pair<std::vector<int>, AccumulatorScaleFunc>>
      biases_params;

  // Quantization parameters for value restricted outputs. This is the
  // "hard-coded" parameters and should be used unconditionally for the
  // quantized op. This vector is empty if the op doesn't have value restricted
  // outputs.
  llvm::DenseMap<SignedInteger, QuantParamsForResults> restricted_output_params;

  // Coefficient operand index and whether supporting per-channel quantization.
  // For QAT, this information is carried by the FakeQuant*/QDQ ops, but
  // post-training quantization, the quantization parameters need to be inferred
  // from the tensor content and op property. A "-1" value indicates the
  // operand doesn't support per-channel quantization.
  llvm::DenseMap<int, int> coeff_op_quant_dim;
};

// A function signature for getting the particular OpQuantSpec for the provided
// op.
typedef std::unique_ptr<OpQuantSpec> (*OpQuantSpecGetter)(Operation* op);

template <typename Q, typename DQ>
struct ConvertStatsToQDQs : public OpRewritePattern<quant::StatisticsOp> {
  ConvertStatsToQDQs(int num_bits, bool narrow_range, bool is_signed,
                     MLIRContext* context)
      : OpRewritePattern<quant::StatisticsOp>(context),
        num_bits(num_bits),
        narrow_range(narrow_range),
        is_signed(is_signed) {}

  PatternMatchResult matchAndRewrite(quant::StatisticsOp op,
                                     PatternRewriter& rewriter) const override {
    Type expressed = op.getType().cast<ShapedType>().getElementType();
    quant::QuantizedType quant_type;
    SmallVector<double, 4> mins, maxs;

    if (op.axisStats().hasValue()) {
      int stats_num = op.axisStats()->getNumElements();
      if (stats_num == 0 || stats_num % 2 != 0) return this->matchFailure();
      auto stats = op.axisStats()->dyn_cast<DenseFPElementsAttr>();
      if (!stats) return this->matchFailure();

      for (auto it = stats.begin(), e = stats.end(); it != e; ++it) {
        mins.push_back(FloatAttr::getValueAsDouble(*it++));
        maxs.push_back(FloatAttr::getValueAsDouble(*it));
      }
      quant_type = quant::fakeQuantAttrsToType(
          op.getLoc(), num_bits, op.axis()->getSExtValue(), mins, maxs,
          narrow_range, expressed, is_signed);
    } else if (auto stats = op.layerStats().dyn_cast<DenseFPElementsAttr>()) {
      double rmin = FloatAttr::getValueAsDouble(stats.getValue<APFloat>({0}));
      double rmax = FloatAttr::getValueAsDouble(stats.getValue<APFloat>({1}));
      quant_type =
          quant::fakeQuantAttrsToType(op.getLoc(), num_bits, rmin, rmax,
                                      narrow_range, expressed, is_signed);
    } else {
      return this->matchFailure();
    }

    rewriter.setInsertionPointAfter(op);
    Type result_type = quant_type.castFromExpressedType(op.getType());
    auto q = rewriter.create<Q>(op.getLoc(), result_type, op.arg(),
                                TypeAttr::get(result_type));
    auto dq = rewriter.create<DQ>(op.getLoc(), op.getType(), q);
    op.getResult()->replaceAllUsesWith(dq);
    q.getOperation()->replaceUsesOfWith(dq, op.arg());
    op.erase();

    return this->matchSuccess();
  }

 private:
  int num_bits;
  bool narrow_range;
  bool is_signed;
};

// A base rewrite pattern which matches any N-in-M-out operations with
// quantization parameters propagated to at least one of its operands. The
// quantization parameters are annotated by the Q/DQ op pairs. Each
// matched pattern are rewritten by its quantized alternatives.
//
// The concret pattern, extends from this base pattern, can specify whether it
// allows "hybrid" operands or results. These "hybrid" operands and results
// don't have quantization parameters propagated to, so will be in float in the
// quantized results. The concret pattern should define the following two
// functions:
//
//   bool AllowHybridOperand() const
//   bool AllowHybridResult() const
//
// Full integer quantization disallows "hybrid" operands or results.
// Weight quantization allows "hybrid" operands and results.
template <typename ConcretTy, typename Q, typename DQ>
struct QuantizationPattern : public RewritePattern {
  using BaseType = QuantizationPattern<ConcretTy, Q, DQ>;

  explicit QuantizationPattern(MLIRContext* context)
      : RewritePattern(DQ::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation* op,
                                     PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return matchFailure();
    }
    Value* quantized_value = op->getResult(0);
    for (Operation* quantized_op : quantized_value->getUsers()) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<Q>(quantized_op) || llvm::isa<DQ>(quantized_op)) {
        return matchFailure();
      }

      // If it is terminator or not quantizable, we shouldn't rewrite.
      if (quantized_op->isKnownTerminator() ||
          quantized_op->hasTrait<OpTrait::quant::NoQuantizableResult>()) {
        return matchFailure();
      }

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value*, 4> inputs;
      inputs.reserve(quantized_op->getNumOperands());
      for (auto operand : quantized_op->getOperands()) {
        Type operand_type = operand->getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }

        auto ele_type = operand->getType().cast<TensorType>().getElementType();
        if (auto op_inst = dyn_cast_or_null<DQ>(operand->getDefiningOp())) {
          inputs.push_back(op_inst.input());
        } else if (ele_type.isa<IntegerType>()) {
          // If the operand is an integer tensor, then it doesn't require the
          // DQ op in the pattern.
          inputs.push_back(operand);
        } else if (static_cast<const ConcretTy*>(this)->AllowHybridOperand()) {
          inputs.push_back(operand);
        } else {
          return matchFailure();
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value*, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantized_op->getNumResults());
      for (auto enumerated_result :
           llvm::enumerate(quantized_op->getResults())) {
        Value* result = enumerated_result.value();
        Type result_type = result->getType();
        // Add this to the test coverage once we create test ops with none type
        // results.
        if (result_type.isa<NoneType>()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        Type result_ele_type =
            result->getType().cast<TensorType>().getElementType();
        // If the user is the Quantize op, it must be the only user.
        if (result->hasOneUse() && llvm::isa<Q>(*result->user_begin())) {
          auto user = llvm::cast<Q>(*result->user_begin());
          outputs_replaced.insert({user.output(), enumerated_result.index()});
          output_types.push_back(user.getType());
        } else if (result_ele_type.template isa<IntegerType>()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result->getType());
        } else if (static_cast<const ConcretTy*>(this)->AllowHybridResult()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result->getType());
        } else {
          return matchFailure();
        }
      }

      rewriter.setInsertionPoint(quantized_op);
      OperationState new_state(quantized_op->getLoc(),
                               quantized_op->getName().getStringRef(), inputs,
                               output_types, quantized_op->getAttrs());
      Operation* new_op = rewriter.createOperation(new_state);
      for (auto output : outputs_replaced) {
        output.getFirst()->replaceAllUsesWith(
            new_op->getResult(output.getSecond()));
      }
    }
    return matchSuccess();
  }
};

// Converts quantize ops with unsigned quantized types to these with signed
// quantized types and preserves the scales.
template <typename Q>
struct ConvertUnsignedToSigned : public OpRewritePattern<Q> {
  using BaseType = ConvertUnsignedToSigned<Q>;
  using QType = quant::QuantizedType;

  explicit ConvertUnsignedToSigned(MLIRContext* context)
      : OpRewritePattern<Q>(context, 1) {}

  PatternMatchResult matchAndRewrite(Q op,
                                     PatternRewriter& rewriter) const override {
    Type output_type = op.output()->getType();
    auto qtype = QType::getQuantizedElementType(output_type);
    if (!qtype || qtype.isSigned()) return this->matchFailure();

    int num_bits = qtype.getStorageTypeIntegralWidth();
    // This is a positive value, and will be applied on zero points and fixed
    // point ranges.
    int64_t offset =
        QType::getDefaultMinimumForInteger(/*isSigned=*/false, num_bits) -
        QType::getDefaultMinimumForInteger(/*isSigned=*/true, num_bits);

    auto flags = quant::QuantizationFlags::Signed;
    QType new_qtype;
    if (auto uqtype = qtype.template dyn_cast<quant::UniformQuantizedType>()) {
      new_qtype = quant::UniformQuantizedType::getChecked(
          flags, qtype.getStorageType(), qtype.getExpressedType(),
          uqtype.getScale(), uqtype.getZeroPoint() - offset,
          uqtype.getStorageTypeMin() - offset,
          uqtype.getStorageTypeMax() - offset, op.getLoc());
    } else if (auto aqtype = qtype.template dyn_cast<
                             quant::UniformQuantizedPerAxisType>()) {
      auto zero_points = aqtype.getZeroPoints();
      llvm::SmallVector<int64_t, 4> new_zero_points(zero_points.begin(),
                                                    zero_points.end());
      for (int i = 0, e = new_zero_points.size(); i != e; ++i) {
        new_zero_points[i] -= offset;
      }
      new_qtype = quant::UniformQuantizedPerAxisType::getChecked(
          flags, qtype.getStorageType(), qtype.getExpressedType(),
          aqtype.getScales(), new_zero_points, aqtype.getQuantizedDimension(),
          aqtype.getStorageTypeMin() - offset,
          aqtype.getStorageTypeMax() - offset, op.getLoc());
    } else {
      return this->matchFailure();
    }

    Type new_output_type = new_qtype.castFromExpressedType(
        QType::castToExpressedType(output_type));
    rewriter.replaceOpWithNewOp<Q>(op, new_output_type, op.input(),
                                   TypeAttr::get(new_output_type));
    return this->matchSuccess();
  }
};

// Converts the min/max/num_bits/narrow_range information to a
// QuantizedType, and then returns the attribute containing the QuantizedType.
// The `min` and `max` arguments can be FloatAttr or DenseFPElementsAttr and
// returns UniformQuantizedType or UniformQuantizedPerAxisType respectively.
// `narrow_range` is set to true for weights and `is_signed` is set to true
// if it is using signed int symmetric quantization.
//
// Note that this method may broadcast min and max to match the dimension length
// of `input_type`, if the the `quant_dim` is valid. On the other hand, the
// symmetry of min and max is not adjusted by this method. The QAT workflow
// should set min/max correctly (and use `narrow_range`=true, `is_signed`=true)
// if symmetric quantization is required.
TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, int quant_dim,
                              IntegerAttr num_bits, BoolAttr narrow_range,
                              bool is_signed);

// Casts the `target` type to a quantized type by using the quantization
// parameters from the type in the `source` type attribute.
// Examples:
//   f32 -> !quant.uniform<i8:f32, 1.0>
//   tensor<4xf32> -> tensor<4x!quant.uniform<i8:f32, 1.0>>
// The result is wrapped by a type attribute. Returns nullptr if the cast
// isn't valid.
//
// `axis` is to specify the quantization dimension in the `target` and only
// used if the element type of `source` is a per-channel quantized type. During
// the casting, the quantization dimension of the result type needs to be set
// this new `axis` value.
TypeAttr CastQuantizedTypeAttrFromExpressedType(Builder builder,
                                                TypeAttr source, Type target,
                                                int axis);

// Quantizes the elements in the attribute `real_value` by the quantization
// parameters in `tensor_type`. Returns empty Attribute if the
// `tensor_type` is not a QuantizedType or the quantization fails.
ElementsAttr Quantize(Attribute real_value, Type tensor_type);

// Returns the quantized type for an element attribute. The quantization
// parameters in this type is based on the min and max element of the
// attribute. When the elements in the `attr` are not in floating-point, or
// the value range isn't straddling zero, an empty type is returned.
Type GetUniformQuantizedTypeForWeight(ElementsAttr attr, unsigned num_bits,
                                      bool is_sign, bool narrow_range);

// Returns the per channel quantized type for an element attribute.
// `quant_dim` defines the quantization axis. The channel min/max are ajusted
// to by symmetric if `symmetric` flag is set to True.
Type GetUniformQuantizedPerAxisTypeForWeight(ElementsAttr attr, int quant_dim,
                                             bool symmetric, unsigned num_bits,
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
// and dequantize ops to this function. Set `disable_per_channel` to true to not
// use per channel quantization even the op supports it.
void ApplyQuantizationParamsPropagation(mlir::FuncOp func, bool is_signed,
                                        bool disable_per_channel,
                                        OpQuantSpecGetter op_quant_spec_getter);

// The function might contain more stats ops than required, and it will
// introduce requantize if the calibration stats have conflicts. This method
// tries to remove all the redundant stats ops.
bool RemoveRedundantStatsOps(mlir::FuncOp func,
                             OpQuantSpecGetter op_quant_spec_getter);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_
