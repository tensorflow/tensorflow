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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"

namespace mlir {
namespace quant {

// A unit attribute can be attached to the quantize/dequantize ops which are
// added by the quantization passes. These ops can be removed erased without
// losing accuracy.
constexpr char kVolatileOpAttrName[] = "volatile";

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

  LogicalResult matchAndRewrite(quant::StatisticsOp op,
                                PatternRewriter& rewriter) const override {
    Type expressed = op.getType().cast<ShapedType>().getElementType();
    quant::QuantizedType quant_type;
    SmallVector<double, 4> mins, maxs;

    if (op.axisStats().hasValue()) {
      int stats_num = op.axisStats()->getNumElements();
      if (stats_num == 0 || stats_num % 2 != 0) return failure();
      auto stats = op.axisStats()->dyn_cast<DenseFPElementsAttr>();
      if (!stats) return failure();

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
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    Type result_type = quant_type.castFromExpressedType(op.getType());
    auto q = rewriter.create<Q>(op.getLoc(), result_type, op.arg());
    auto dq = rewriter.create<DQ>(op.getLoc(), op.getType(), q);
    op.getResult().replaceAllUsesWith(dq);
    q.getOperation()->replaceUsesOfWith(dq, op.arg());
    op.erase();

    return success();
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
// The concrete pattern, extends from this base pattern, can specify whether it
// allows "hybrid" operands or results. These "hybrid" operands and results
// don't have quantization parameters propagated to, so will be in float in the
// quantized results. The concrete pattern should define the following two
// functions:
//
//   bool AllowHybridOperand() const
//   bool AllowHybridResult() const
//
// Full integer quantization disallows "hybrid" operands or results.
// Weight quantization allows "hybrid" operands and results.
template <typename ConcretTy, typename Q, typename DQ, typename VERIFIER>
struct QuantizationPattern : public RewritePattern {
  using BaseType = QuantizationPattern<ConcretTy, Q, DQ, VERIFIER>;

  explicit QuantizationPattern(MLIRContext* context, bool enable_verify,
                               float error_tolerance, bool single_layer_verify)
      // Set the score to a large number so it is always preferred.
      : RewritePattern(DQ::getOperationName(), 300, context),
        enable_verify(enable_verify),
        error_tolerance(error_tolerance),
        single_layer_verify(single_layer_verify) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return failure();
    }
    Value quantized_value = op->getResult(0);
    for (Operation* quantized_op : quantized_value.getUsers()) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<Q>(quantized_op) || llvm::isa<DQ>(quantized_op)) {
        return failure();
      }

      // If it is terminator or not quantizable or any ops form the mlir quant
      // ops dialect, we shouldn't rewrite.
      if (quantized_op->isKnownTerminator() ||
          quantized_op->hasTrait<OpTrait::quant::NoQuantizableResult>() ||
          llvm::isa<quant::QuantizeCastOp>(quantized_op) ||
          llvm::isa<quant::DequantizeCastOp>(quantized_op)) {
        return failure();
      }

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantized_op->getNumOperands());
      for (auto operand : quantized_op->getOperands()) {
        Type operand_type = operand.getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }

        auto ele_type = operand.getType().cast<TensorType>().getElementType();
        if (auto op_inst = dyn_cast_or_null<DQ>(operand.getDefiningOp())) {
          inputs.push_back(op_inst.input());
        } else if (ele_type.isSignlessInteger()) {
          // If the operand is an integer tensor, then it doesn't require the
          // DQ op in the pattern.
          inputs.push_back(operand);
        } else if (static_cast<const ConcretTy*>(this)->AllowHybridOperand()) {
          inputs.push_back(operand);
        } else {
          return failure();
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantized_op->getNumResults());
      for (auto enumerated_result :
           llvm::enumerate(quantized_op->getResults())) {
        Value result = enumerated_result.value();
        Type result_type = result.getType();
        // Add this to the test coverage once we create test ops with none type
        // results.
        if (result_type.isa<NoneType>()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        Type result_ele_type =
            result.getType().cast<TensorType>().getElementType();
        // If the user is the Quantize op, it must be the only user.
        if (result.hasOneUse() && llvm::isa<Q>(*result.user_begin())) {
          auto user = llvm::cast<Q>(*result.user_begin());
          outputs_replaced.insert({user.output(), enumerated_result.index()});
          output_types.push_back(user.getType());
        } else if (result_ele_type.isSignlessInteger()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else if (static_cast<const ConcretTy*>(this)->AllowHybridResult()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          return failure();
        }
      }

      rewriter.setInsertionPointAfter(quantized_op);
      OperationState new_state(quantized_op->getLoc(),
                               quantized_op->getName().getStringRef(), inputs,
                               output_types, quantized_op->getAttrs());
      Operation* new_op = rewriter.createOperation(new_state);
      for (auto output : outputs_replaced) {
        output.getFirst().replaceAllUsesWith(
            new_op->getResult(output.getSecond()));
      }

      // To verify the numericals, the original floating-point ops are
      // preserved in the graph. The result of these floating-point ops are sent
      // to a numeric verifier op as the reference.
      if (enable_verify) {
        // For constant operands, the floating-point constant is duplicated in
        // case it is quantized.
        for (int i = 0, e = new_op->getNumOperands(); i != e; ++i) {
          auto def = new_op->getOperand(i).getDefiningOp();
          if (auto q = llvm::dyn_cast_or_null<Q>(def)) {
            DenseFPElementsAttr attr;
            if (!matchPattern(q.input(), m_Constant(&attr))) {
              continue;
            }
            auto cst = rewriter.create<ConstantOp>(new_op->getLoc(), attr);
            quantized_op->setOperand(i, cst.getResult());
          }
        }

        for (int i = 0, e = new_op->getNumResults(); i != e; ++i) {
          if (!quantized_op->getResult(i)
                   .getType()
                   .cast<ShapedType>()
                   .getElementType()
                   .isa<FloatType>()) {
            continue;
          }
          rewriter.setInsertionPointAfter(new_op);
          FloatAttr tolerance = rewriter.getF32FloatAttr(error_tolerance);
          // Verify the quantized value by sending the result to the verifier.
          rewriter.create<VERIFIER>(quantized_op->getLoc(),
                                    new_op->getResult(i),
                                    quantized_op->getResult(i), tolerance);

          if (single_layer_verify) continue;

          // Find the Dequantize/Dequantize users of the new op results, and
          // replace the usage. Then all the floating-point ops are connected.
          // N.B. the return op will use this floating-point result.
          for (auto user : new_op->getResult(i).getUsers()) {
            // Skip the Requantize op, and we know it has a single user.
            if (llvm::isa<Q>(user)) {
              user = *user->getResult(0).getUsers().begin();
            }
            if (auto dequantize = llvm::dyn_cast<DQ>(user)) {
              dequantize.getResult().replaceAllUsesWith(
                  quantized_op->getResult(i));
            }
          }
        }
      }
    }
    return success();
  }

  bool enable_verify;
  float error_tolerance;
  bool single_layer_verify;
};

// Converts quantize ops with unsigned quantized types to these with signed
// quantized types and preserves the scales.
template <typename Q>
struct ConvertUnsignedToSigned : public OpRewritePattern<Q> {
  using BaseType = ConvertUnsignedToSigned<Q>;
  using QType = quant::QuantizedType;

  explicit ConvertUnsignedToSigned(MLIRContext* context)
      : OpRewritePattern<Q>(context, 1) {}

  LogicalResult matchAndRewrite(Q op,
                                PatternRewriter& rewriter) const override {
    Type output_type = op.getResult().getType();
    auto qtype = QType::getQuantizedElementType(output_type);
    if (!qtype || qtype.isSigned()) return failure();

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
      return failure();
    }

    if (!new_qtype) return failure();
    Type new_output_type = new_qtype.castFromExpressedType(
        QType::castToExpressedType(output_type));
    rewriter.replaceOpWithNewOp<Q>(op, new_output_type, op.arg());
    return success();
  }
};

// Fold Extra Requantize ops if the preceding ops has free scale requirement.
template <typename RQ>
struct FoldTrivalRequantizeOp : public OpRewritePattern<RQ> {
  explicit FoldTrivalRequantizeOp(MLIRContext* context)
      : OpRewritePattern<RQ>(context, 1) {}

  LogicalResult matchAndRewrite(RQ op,
                                PatternRewriter& rewriter) const override {
    Value pre_quantized = op.input();
    auto pre_quantized_type =
        quant::QuantizedType::getQuantizedElementType(pre_quantized.getType());
    if (!pre_quantized_type) return failure();

    Operation* def = pre_quantized.getDefiningOp();
    if (!def) return failure();
    if (def->hasTrait<OpTrait::quant::SameOperandsAndResultsScale>() ||
        def->hasTrait<OpTrait::quant::NoQuantizableResult>()) {
      return failure();
    }

    op.emitWarning("Remove trivial `rescale` op. Please fix the source graph.");

    llvm::SmallVector<Type, 4> new_output_types;
    for (auto result : def->getResults()) {
      result.getUsers().begin()->dump();
      op.dump();
      if (result.hasOneUse() && *result.getUsers().begin() == op) {
        new_output_types.push_back(op.qtype());
      } else {
        new_output_types.push_back(result.getType());
      }
    }

    // Remove this rescale op.
    rewriter.replaceOp(op, {pre_quantized});

    // Replace the output scale of the preceding op.
    rewriter.setInsertionPointAfter(def);
    OperationState new_state(def->getLoc(), def->getName().getStringRef(),
                             def->getOperands(), new_output_types,
                             def->getAttrs());
    Operation* new_op = rewriter.createOperation(new_state);

    rewriter.replaceOp(def, new_op->getResults());
    return success();
  }
};

// Given a quantized type `input`, magnifying its scales by the factor stored in
// `factor`. If `input` isn't a quantized type or the `factor` doesn't match the
// dimension size of `input` or isn't floating-point, nullptr will be returned.
TypeAttr RescaleQuantizedType(Type input, Attribute factor);

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
// the value range isn't straddling zero, an empty type is returned. The min/max
// are adjusted to be symmetric if `symmetric` flag is set to True. And
// `symmetric` can only be set to true when it is signed and narrow_range.
Type GetUniformQuantizedTypeForWeight(ElementsAttr attr, bool symmetric,
                                      unsigned num_bits, bool is_sign,
                                      bool narrow_range);

// Returns the per channel quantized type for an element attribute.
// `quant_dim` defines the quantization axis. The channel min/max are adjusted
// to be symmetric if `symmetric` flag is set to True. And `symmetric` can only
// be set to true when it is signed and narrow_range.
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

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_
