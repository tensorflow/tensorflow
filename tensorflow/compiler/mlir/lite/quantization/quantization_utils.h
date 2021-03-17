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

#include <string>
#include <unordered_map>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"

namespace mlir {
namespace quant {

// A unit attribute can be attached to the quantize/dequantize ops which are
// added by the quantization passes. These ops can be removed erased without
// losing accuracy.
constexpr char kVolatileOpAttrName[] = "volatile";

enum QuantizationTrait { FullyQuantizable, NotQuantizable };
extern const char kQuantTraitAttr[];
extern const absl::string_view QuantTraitValues[];

using QuantParams = quant::QuantizedType;
using SignedInteger = std::pair<unsigned, unsigned>;  // bitwidth and sign
using QuantParamsForResults = llvm::SmallVector<QuantParams, 4>;
using AccumulatorScaleFunc =
    std::function<QuantParams(const std::vector<QuantParams>&, bool)>;

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

// Re-calculates scales again in float instead of simply downcasting existing
// scales.
QuantizedType DownCastScale(QuantizedType type,
                            const SmallVectorImpl<double>& mins,
                            const SmallVectorImpl<double>& maxs, Location loc);

QuantizedType DownCastScale(QuantizedType type, double min, double max,
                            Location loc);

bool IsOpNotQuantizable(Operation* op);

template <typename Q, typename DQ>
struct ConvertStatsToQDQs : public OpRewritePattern<quant::StatisticsOp> {
  ConvertStatsToQDQs(int num_bits, bool narrow_range, bool is_signed,
                     bool legacy_float_scale, MLIRContext* context)
      : OpRewritePattern<quant::StatisticsOp>(context),
        num_bits(num_bits),
        narrow_range(narrow_range),
        is_signed(is_signed),
        legacy_float_scale(legacy_float_scale) {}

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
        double rmin = FloatAttr::getValueAsDouble(*it++);
        double rmax = FloatAttr::getValueAsDouble(*it);
        // The default nudging implementation of mlir quant library might cause
        // clamping during inference if the calibration range isn't wide enough.
        // So here we adjust the range to include 0.0.
        rmin = std::min(rmin, 0.0);
        rmax = std::max(rmax, 0.0);
        TensorRangeSanityCheck(op, rmin, rmax);
        mins.push_back(rmin);
        maxs.push_back(rmax);
      }
      quant_type =
          quant::fakeQuantAttrsToType(op.getLoc(), num_bits, *op.axis(), mins,
                                      maxs, narrow_range, expressed, is_signed);
      if (legacy_float_scale) {
        quant_type = DownCastScale(quant_type, mins, maxs, op->getLoc());
      }
    } else if (auto stats = op.layerStats().dyn_cast<DenseFPElementsAttr>()) {
      double rmin = FloatAttr::getValueAsDouble(stats.getValue<APFloat>({0}));
      double rmax = FloatAttr::getValueAsDouble(stats.getValue<APFloat>({1}));
      // The default nudging implementation of mlir quant library might cause
      // clamping during inference if the calibration range isn't wide enough.
      // So here we adjust the range to include 0.0.
      rmin = std::min(rmin, 0.0);
      rmax = std::max(rmax, 0.0);
      TensorRangeSanityCheck(op, rmin, rmax);
      quant_type =
          quant::fakeQuantAttrsToType(op.getLoc(), num_bits, rmin, rmax,
                                      narrow_range, expressed, is_signed);
      if (legacy_float_scale) {
        quant_type = DownCastScale(quant_type, rmin, rmax, op->getLoc());
      }
    } else {
      return failure();
    }

    rewriter.setInsertionPointAfter(op.getOperation());
    Type result_type = quant_type.castFromExpressedType(op.getType());
    auto q = rewriter.create<Q>(op.getLoc(), result_type, op.arg());
    q->setAttr(kVolatileOpAttrName, rewriter.getUnitAttr());

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
  bool legacy_float_scale;

  // Emits an op warning message if the calibrated range is larger than 10.0 and
  // the storage type is less than or equal to 8 bits.
  void TensorRangeSanityCheck(quant::StatisticsOp op, double min,
                              double max) const {
    double range = std::fabs(max - min);
    if (num_bits <= 8 && range >= 10.0) {
      op.emitWarning(
          "Tensor range is too wide to be quantized. Use tf.clip_by_value or "
          "tf.relu6 to narrow the tensor range. Range: " +
          std::to_string(range) + ", bit width: " + std::to_string(num_bits));
    }
  }
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
                               float error_tolerance, bool single_layer_verify,
                               bool log_if_failed = false)
      // Set the score to a large number so it is always preferred.
      : RewritePattern(DQ::getOperationName(), 300, context),
        enable_verify(enable_verify),
        error_tolerance(error_tolerance),
        single_layer_verify(single_layer_verify),
        log_if_failed(log_if_failed) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) {
      return failure();
    }
    Value quantized_value = op->getResult(0);
    for (Operation* quantized_op : quantized_value.getUsers()) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<Q, DQ>(quantized_op)) {
        return failure();
      }

      // If it is terminator or not quantizable or any ops form the mlir quant
      // ops dialect, we shouldn't rewrite.
      if (IsOpNotQuantizable(quantized_op)) {
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
      for (int i = 0; i < quantized_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* new_op = rewriter.createOperation(new_state);
      if (quantized_op->getNumRegions() != 0) {
        for (auto indexed_regions :
             llvm::enumerate(quantized_op->getRegions())) {
          Region& target_region = new_op->getRegion(indexed_regions.index());
          BlockAndValueMapping mapping;
          indexed_regions.value().cloneInto(&target_region, mapping);
        }
      }
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
          BoolAttr log = rewriter.getBoolAttr(log_if_failed);
          // Verify the quantized value by sending the result to the verifier.
          rewriter.create<VERIFIER>(
              quantized_op->getLoc(), new_op->getResult(i).getType(),
              new_op->getResult(i), quantized_op->getResult(i), tolerance, log);

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
  bool log_if_failed;
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
          op.getLoc(), flags, qtype.getStorageType(), qtype.getExpressedType(),
          uqtype.getScale(), uqtype.getZeroPoint() - offset,
          uqtype.getStorageTypeMin() - offset,
          uqtype.getStorageTypeMax() - offset);
    } else if (auto aqtype = qtype.template dyn_cast<
                             quant::UniformQuantizedPerAxisType>()) {
      auto zero_points = aqtype.getZeroPoints();
      llvm::SmallVector<int64_t, 4> new_zero_points(zero_points.begin(),
                                                    zero_points.end());
      for (int i = 0, e = new_zero_points.size(); i != e; ++i) {
        new_zero_points[i] -= offset;
      }
      new_qtype = quant::UniformQuantizedPerAxisType::getChecked(
          op.getLoc(), flags, qtype.getStorageType(), qtype.getExpressedType(),
          aqtype.getScales(), new_zero_points, aqtype.getQuantizedDimension(),
          aqtype.getStorageTypeMin() - offset,
          aqtype.getStorageTypeMax() - offset);
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
    if (llvm::isa<FixedOutputRangeInterface, SameScalesOpInterface>(def) ||
        def->hasTrait<OpTrait::quant::NoQuantizableResult>()) {
      return failure();
    }

    op.emitWarning("Remove trivial `rescale` op. Please fix the source graph.");

    llvm::SmallVector<Type, 4> new_output_types;
    for (auto result : def->getResults()) {
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
// of `input_type`, if the `quant_dim` is valid. On the other hand, the
// symmetry of min and max is not adjusted by this method. The QAT workflow
// should set min/max correctly (and use `narrow_range`=true, `is_signed`=true)
// if symmetric quantization is required.
TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, int quant_dim,
                              IntegerAttr num_bits, BoolAttr narrow_range,
                              bool is_signed, bool legacy_float_scale = false);

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

// Quantizes the elements in "legacy mode", where it calls TOCO's methods to
// to quantize values with float scale.
ElementsAttr QuantizeLegacy(Attribute real_value, Type tensor_type);

// Returns the quantized type for an element attribute. The quantization
// parameters in this type is based on the min and max element of the
// attribute. When the elements in the `attr` are not in floating-point, or
// the value range isn't straddling zero, an empty type is returned. The min/max
// are adjusted to be symmetric if `symmetric` flag is set to True. And
// `symmetric` can only be set to true when it is signed and narrow_range.
Type GetUniformQuantizedTypeForWeight(ElementsAttr attr, bool symmetric,
                                      unsigned num_bits, bool is_signed,
                                      bool narrow_range,
                                      bool legacy_float_scale = false);

// Returns the per channel quantized type for an element attribute.
// `quant_dim` defines the quantization axis. The channel min/max are adjusted
// to be symmetric if `symmetric` flag is set to True. And `symmetric` can only
// be set to true when it is signed and narrow_range.
Type GetUniformQuantizedPerAxisTypeForWeight(ElementsAttr attr, int quant_dim,
                                             bool symmetric, unsigned num_bits,
                                             bool is_signed, bool narrow_range,
                                             bool legacy_float_scale = false);

// Returns the quantized type of a bias input, given the quantized types of
// other operands which are multiply-accumulated (the bias is added to the
// accumulated value).
quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types,
    bool legacy_float_scale = false);

// Propagates quantization parameters across ops in this function and satisfy
// the quantization specification of the ops. This methods assumes the initial
// quantization parameters are stored as adjacent quantize and dequantize ops
// and the propagation results are materialized by inserting pairs of quantize
// and dequantize ops to this function. Set `disable_per_channel` to true to not
// use per channel quantization even the op supports it.
// Setting `infer_tensor_range` to true, to infer quantization parameters from
// the activation ops and weight constants. This is only used for post-training
// quantization.
void ApplyQuantizationParamsPropagation(mlir::FuncOp func, bool is_signed,
                                        bool disable_per_channel,
                                        OpQuantSpecGetter op_quant_spec_getter,
                                        bool infer_tensor_ranges,
                                        bool legacy_float_scale = false);

// The function might contain more stats ops than required, and it will
// introduce requantize if the calibration stats have conflicts. This method
// tries to remove all the redundant stats ops.
bool RemoveRedundantStatsOps(mlir::FuncOp func,
                             OpQuantSpecGetter op_quant_spec_getter);

// Given quantization parameters for int8, compute the quantization parameters
// for uint if it is required, and wrap the result in an UniformQuantizedType.
quant::UniformQuantizedType GetFixedOutputRange(bool is_signed, int bit_width,
                                                Type tensor_type, double scale,
                                                int64_t zero_point,
                                                int64_t storage_min = -128,
                                                int64_t storage_max = 127);

// Extrace min and max values from the DenseFPElementsAttr, and stores them into
// `mins` and `maxs`. When mins and maxs are extracted per-channel, `dim_size`
// is number of channels and `slice_size` is the size of slice per each channel.
// When `symmetric` is true, the range is expanded to [-M, M].
void ExtractMinMaxFromAttr(DenseFPElementsAttr values, int dim_size,
                           int slice_size, bool symmetric,
                           SmallVectorImpl<double>& mins,
                           SmallVectorImpl<double>& maxs);

// Returns the quantized type for the
// input_type/min/max/storag_type_width/narrow_range.
Type GetQuantizedType(Builder builder, Type input_type, ArrayRef<double> min,
                      ArrayRef<double> max, int quant_dim,
                      int storage_type_width, bool narrow_range, bool is_signed,
                      bool legacy_float_scale = false);
}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_
