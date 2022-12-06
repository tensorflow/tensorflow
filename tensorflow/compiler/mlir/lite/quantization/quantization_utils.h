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

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace quant {

// A unit attribute can be attached to the quantize/dequantize ops which are
// added by the quantization passes. These ops can be removed erased without
// losing accuracy.
constexpr char kVolatileOpAttrName[] = "volatile";

// Following attributes are used to mark ops that are not quantizable during
// debug model generation process for whole-model verify mode. If these
// attributes are attached, the upstream float/quantized ops know which ops to
// connect to, and it also prevents these ops from being copied again.
constexpr char kDebugModeOpFloatAttrName[] = "debug_float";
constexpr char kDebugModeOpQuantAttrName[] = "debug_quant";

// Used to annotate custom ops if they are quantizable.
constexpr char kQuantTraitAttrName[] = "_tfl_quant_trait";
enum QuantizationTrait { FullyQuantizable = 0, NotQuantizable = 1 };
constexpr absl::string_view QuantTraitValues[] = {"fully_quantizable",
                                                  "not_quantizable"};

constexpr double kNearZeroTolerance = 1.0e-6;

using QuantParams = mlir::quant::QuantizedType;
using QuantSpec = QuantizationSpecs;
using SignedInteger = std::pair<unsigned, unsigned>;  // bitwidth and sign
using QuantParamsForResults = llvm::SmallVector<QuantParams, 4>;
using AccumulatorScaleFunc =
    std::function<QuantParams(const std::vector<QuantParams>&, bool)>;
using BiasParamsMap =
    std::unordered_map<int, std::pair<std::vector<int>, AccumulatorScaleFunc>>;
// UniformQuantizedType GetFixedOutputRange(bool sign, int bit_width)
using GetFixedOutputRangeFunc = std::function<UniformQuantizedType(bool, int)>;
// bool RequiredSameOperandsAndResultsScale(bool sign, int $bit_width)
using RequiredSameOperandsAndResultsScaleFunc = std::function<bool(bool, int)>;
// bool RequiredSameQuantizedAxes()
using RequiredSameQuantizedAxesFunc = std::function<bool()>;

using StringSet = absl::flat_hash_set<std::string>;
using CustomMap = quant::CustomOpMap;

// Quantization spec of an op, driving the quantization algorithm.
struct OpQuantSpec {
  // Maps the operand index of a bias input to its quantization specifications,
  // including the non-bias operand indexes and the method retrieving
  // quantization parameters from list of parameters of the non-bias operands.
  // This map is empty if the op doesn't have a bias operand.
  BiasParamsMap biases_params;

  // Quantization parameters for value restricted outputs. This is the
  // "hard-coded" parameters and should be used unconditionally for the
  // quantized op. This vector is empty if the op doesn't have value restricted
  // outputs.
  llvm::DenseMap<SignedInteger, QuantParamsForResults> restricted_output_params;

  // Coefficient operand index and whether supporting per-channel quantization.
  // For QAT, this information is carried by the FakeQuant*/Quantize/Dequantize
  // ops, but post-training quantization, the quantization parameters need to be
  // inferred from the tensor content and op property. A "-1" value indicates
  // the operand doesn't support per-channel quantization.
  llvm::DenseMap<int, int> coeff_op_quant_dim;

  // Indices of quantizable operands. Biases are not included in this field,
  // the indices of biases can be found in the `biases_params`.
  absl::flat_hash_set<int> quantizable_operands;
};

// Quantization scale spec of an op. The information defined in the MLIR
// interfaces FixedOutputRangeInterface and SameOperandsAndResultsScale should
// be checked first if present.
struct OpQuantScaleSpec {
  // Whether this op has a fixed range requirement (e.g. sigmoid)
  bool has_fixed_output_range = false;
  // Whether this op should have same result and operand scales (e.g. concat)
  bool has_same_scale_requirement = false;
  // Returns the fixed output range, when has_fixed_output_range is set.
  GetFixedOutputRangeFunc fixed_output_range_func;
  // Returns whether same operands and results scales are required.
  RequiredSameOperandsAndResultsScaleFunc required_same_scale_func =
      [](bool sign, int bit_width) { return true; };
  // Returns whether operands and results must have the same quantized axis.
  RequiredSameQuantizedAxesFunc required_same_quantized_axes_func = []() {
    return true;
  };
};

// Used in TFL Numeric Verify
struct NumericVerifySpec {
  // Whether to enable numeric verification
  bool verify_numeric = false;

  // Tolerance level from the quantized value for verification. If the tolerance
  // is very small(<0.1), only the stats of the diff is displayed.
  float error_tolerance = 5.0f;

  // Whether to verify numerical correctness layer by layer or by whole model
  bool whole_model_verify = false;

  // Whether to enable log for failures
  bool log_if_failed_flag = false;
};

// Used in TFL Quantize Pass
struct QuantPassSpec {
  // Variables to control TFL Numeric Verify
  NumericVerifySpec numeric_verify_spec;

  // Variables related to quantization
  QuantSpec quant_spec;
};

// A function signature for getting the particular OpQuantSpec for the provided
// op.
typedef std::unique_ptr<OpQuantSpec> (*OpQuantSpecGetter)(Operation* op);
// A function signature for getting the particular OpQuantScaleSpec for the
// provided op.
typedef std::unique_ptr<OpQuantScaleSpec> (*OpQuantScaleSpecGetter)(
    Operation* op);

// Re-calculates scales again in float instead of simply downcasting existing
// scales.
quant::QuantizedType DownCastScale(quant::QuantizedType type,
                                   const SmallVectorImpl<double>& mins,
                                   const SmallVectorImpl<double>& maxs,
                                   Location loc);

quant::QuantizedType DownCastScale(quant::QuantizedType type, double min,
                                   double max, Location loc);

bool IsOpNotQuantizable(Operation* op);

// Specialized version of location to string for flatbuffer exported locations.
inline std::string GetTensorNameFromLoc(Location loc) {
  if (auto name_loc = loc.dyn_cast<NameLoc>()) {
    return name_loc.getName().str();
  }
  return "";
}

template <typename QuantizeOpT, typename DequantizeOpT>
struct ConvertStatsToQDQs : public OpRewritePattern<quantfork::StatisticsOp> {
  ConvertStatsToQDQs(int num_bits, bool narrow_range, bool is_signed,
                     bool legacy_float_scale, MLIRContext* context)
      : OpRewritePattern<quantfork::StatisticsOp>(context),
        num_bits(num_bits),
        narrow_range(narrow_range),
        is_signed(is_signed),
        legacy_float_scale(legacy_float_scale) {}

  LogicalResult matchAndRewrite(quantfork::StatisticsOp op,
                                PatternRewriter& rewriter) const override {
    Type expressed = op.getType().cast<ShapedType>().getElementType();
    quant::QuantizedType quant_type;
    SmallVector<double, 4> mins, maxs;

    if (op.getAxisStats().has_value()) {
      int stats_num = op.getAxisStats()->getNumElements();
      if (stats_num == 0 || stats_num % 2 != 0) return failure();
      auto stats = op.getAxisStats()->dyn_cast<DenseFPElementsAttr>();
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
      quant_type = quantfork::fakeQuantAttrsToType(
          op.getLoc(), num_bits, *op.getAxis(), mins, maxs, narrow_range,
          expressed, is_signed);
      if (legacy_float_scale) {
        quant_type = DownCastScale(quant_type, mins, maxs, op->getLoc());
      }
    } else if (auto stats =
                   op.getLayerStats().dyn_cast<DenseFPElementsAttr>()) {
      auto statValues = stats.getValues<APFloat>();
      double rmin = FloatAttr::getValueAsDouble(statValues[0]);
      double rmax = FloatAttr::getValueAsDouble(statValues[1]);
      // The default nudging implementation of mlir quant library might cause
      // clamping during inference if the calibration range isn't wide enough.
      // So here we adjust the range to include 0.0.
      rmin = std::min(rmin, 0.0);
      rmax = std::max(rmax, 0.0);
      TensorRangeSanityCheck(op, rmin, rmax);
      quant_type =
          quantfork::fakeQuantAttrsToType(op.getLoc(), num_bits, rmin, rmax,
                                          narrow_range, expressed, is_signed);
      if (legacy_float_scale) {
        quant_type = DownCastScale(quant_type, rmin, rmax, op->getLoc());
      }
    } else {
      return failure();
    }

    rewriter.setInsertionPointAfter(op.getOperation());
    Type result_type = quant_type.castFromExpressedType(op.getType());
    auto q =
        rewriter.create<QuantizeOpT>(op.getLoc(), result_type, op.getArg());
    q->setAttr(kVolatileOpAttrName, rewriter.getUnitAttr());

    auto dq = rewriter.create<DequantizeOpT>(op.getLoc(), op.getType(), q);
    op.getResult().replaceAllUsesWith(dq);
    q.getOperation()->replaceUsesOfWith(dq, op.getArg());
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
  void TensorRangeSanityCheck(quantfork::StatisticsOp op, double& min,
                              double& max) const {
    double range = std::fabs(max - min);
    if (num_bits <= 8 && range >= 10.0) {
      op.emitWarning()
          << "Tensor range is too wide to be quantized. Use tf.clip_by_value "
             "or tf.relu6 to narrow the tensor range. Range: "
          << range << ", bit width: " << num_bits;
    }
    if (std::abs(max - min) < kNearZeroTolerance) {
      op.emitWarning() << "Tensor range (" << min << ", " << max
                       << ") is too narrow and it might cause overflow. "
                          "Expanding range symmetrically by "
                       << kNearZeroTolerance;
      min -= kNearZeroTolerance;
      max += kNearZeroTolerance;
    }
  }
};

template <typename VerifierT>
bool UsedBy(Operation* op) {
  for (Operation* user : op->getUsers()) {
    if (llvm::isa_and_nonnull<VerifierT>(user)) return true;
  }
  return false;
}

template <typename VerifierT>
void CreateVerifier(Operation* quantizing_op, Operation* quantized_op,
                    PatternRewriter& rewriter, int result_idx,
                    const QuantPassSpec& quant_params) {
  rewriter.setInsertionPointAfter(quantized_op);
  FloatAttr tolerance = rewriter.getF32FloatAttr(
      quant_params.numeric_verify_spec.error_tolerance);
  BoolAttr log =
      rewriter.getBoolAttr(quant_params.numeric_verify_spec.log_if_failed_flag);
  // Verify the quantized value by sending the result to the verifier.
  rewriter.create<VerifierT>(
      quantizing_op->getLoc(), quantized_op->getResult(result_idx).getType(),
      quantized_op->getResult(result_idx), quantizing_op->getResult(result_idx),
      tolerance, log);
}

template <>
inline bool UsedBy<void>(Operation* op) {
  return false;
}

// This specialization is not going to be called, but needed for compilation.
template <>
inline void CreateVerifier<void>(Operation* quantizing_op,
                                 Operation* quantized_op,
                                 PatternRewriter& rewriter, int result_idx,
                                 const QuantPassSpec& quant_params) {}

// A base rewrite pattern which matches any N-in-M-out operations with
// quantization parameters propagated to at least one of its operands. The
// quantization parameters are annotated by the QuantizeOp/DequantizeOp pairs.
// Each matched pattern are rewritten by its quantized alternatives.
//
// The concrete pattern, extends from this base pattern, can specify whether it
// allows dynamic range quantized operands and results for the operations in the
// current context. These "DynamicRangeQuantized" operands and results don't
// have quantization parameters propagated to, so will be in float in the
// quantized results. The concrete pattern should define the following two
// functions:
//
//   bool AllowDynamicRangeQuantizedOperand(Operation *) const
//   bool AllowDynamicRangeQuantizedResult(Operation *) const
//
// Full integer quantization disallows "DynamicRangeQuantized" operands or
// results. Dynamic range quantization allows "DynamicRangeQuantized" operands
// and results.
template <typename ConcreteT, typename QuantizeOpT, typename DequantizeOpT,
          typename VerifierT, typename RootOpT = DequantizeOpT>
class QuantizationPattern : public RewritePattern {
 public:
  using BaseType = QuantizationPattern<ConcreteT, QuantizeOpT, DequantizeOpT,
                                       VerifierT, RootOpT>;

  explicit QuantizationPattern(MLIRContext* context,
                               const QuantPassSpec& quant_params)
      // Set the score to a large number so it is always preferred.
      : RewritePattern(RootOpT::getOperationName(), 300, context),
        quant_params_(quant_params) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*, 4> quantizing_ops;

    // Collect all the ops to quantize, as the user / producer of the root op.
    if (std::is_same<RootOpT, DequantizeOpT>::value) {
      if (op->getNumResults() != 1) {
        return failure();
      }
      auto users = op->getResult(0).getUsers();
      quantizing_ops.append(users.begin(), users.end());
    } else if (std::is_same<RootOpT, QuantizeOpT>::value) {
      if (op->getNumOperands() != 1) {
        return failure();
      }
      Value quantize_operand = op->getOperand(0);
      if (QuantizedType::getQuantizedElementType(quantize_operand.getType())) {
        // The input of this QuantizeOp has already been quantized, i.e.
        // rescale.
        return failure();
      }
      DenseFPElementsAttr attr;
      if (matchPattern(quantize_operand, m_Constant(&attr))) {
        // Const-> QuantizeOp pattern will be handled separately.
        return failure();
      }
      if (Operation* quantizing_op = quantize_operand.getDefiningOp()) {
        quantizing_ops.push_back(quantizing_op);
      }
    }

    tensorflow::DataType inference_type =
        quant_params_.quant_spec.inference_type;
    bool weight_only_quantization =
        quant_params_.quant_spec.weight_only_quantization;
    bool enable_verify = quant_params_.numeric_verify_spec.verify_numeric;
    bool enable_whole_model_verify =
        quant_params_.numeric_verify_spec.whole_model_verify;
    StringSet ops_blocklist = quant_params_.quant_spec.ops_blocklist;
    StringSet nodes_blocklist = quant_params_.quant_spec.nodes_blocklist;
    CustomMap custom_map = quant_params_.quant_spec.custom_map;

    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* quantizing_op : quantizing_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<QuantizeOpT, DequantizeOpT>(quantizing_op)) {
        return failure();
      }

      // If the op is terminator, not quantizable or any ops from the mlir quant
      // ops dialect, we shouldn't rewrite. In case of whole-model verify debug
      // mode, not-quantizable ops should be duplicated to keep parallel
      // float/quant model execution.
      if (quantizing_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }

      if (IsOpNotQuantizable(quantizing_op) &&
          !static_cast<const ConcreteT*>(this)->IsQuantizableCustomOp(
              quantizing_op, custom_map)) {
        if (!(enable_verify && enable_whole_model_verify)) {
          return failure();
        }
        if (quantizing_op->hasAttr(kDebugModeOpQuantAttrName) ||
            quantizing_op->hasAttr(kDebugModeOpFloatAttrName)) {
          return failure();
        }

        rewriter.setInsertionPoint(quantizing_op);
        Operation* float_op = rewriter.clone(*quantizing_op);
        quantizing_op->setAttr(kDebugModeOpQuantAttrName,
                               rewriter.getUnitAttr());
        float_op->setAttr(kDebugModeOpFloatAttrName, rewriter.getUnitAttr());
        RewireFloatModelBackbone(quantizing_op, float_op);
        return success();
      }

      // Blocklist op is checked in advance for non-dynamic range quantization
      // case.
      if (!quant_params_.quant_spec.weight_quantization &&
          (ops_blocklist.find(quantizing_op->getName().getStringRef().str()) !=
           ops_blocklist.end())) {
        return failure();
      }

      if (!nodes_blocklist.empty()) {
        if (auto name_loc = quantizing_op->getLoc().dyn_cast<NameLoc>()) {
          std::string sloc = name_loc.getName().str();
          if (!sloc.empty() &&
              (nodes_blocklist.find(sloc) != nodes_blocklist.end())) {
            return failure();
          }
        }
      }

      // An op with float inputs and outputs are expected when it's used by a
      // NumericVerify op. Skip this op.
      if (enable_verify && UsedBy<VerifierT>(quantizing_op)) {
        continue;
      }

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantizing_op->getNumOperands());
      for (auto operand : quantizing_op->getOperands()) {
        Type operand_type = operand.getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }

        auto ele_type = operand.getType().cast<TensorType>().getElementType();
        if (static_cast<const ConcreteT*>(this)
                ->AllowDynamicRangeQuantizedOperand(quantizing_op,
                                                    custom_map)) {
          auto dq_op = dyn_cast_or_null<DequantizeOpT>(operand.getDefiningOp());

          if (dq_op && inference_type == tensorflow::DT_QINT8 &&
              !static_cast<const ConcreteT*>(this)->IsWeightOnlyOp(
                  quantizing_op, ops_blocklist, weight_only_quantization,
                  custom_map)) {
            // Dynamic range quantization is applied by having QuantizeOp as an
            // input. Only int8 weight is supported for now.
            inputs.push_back(dq_op.getOperand());
          } else {
            // Otherwise, it's the case where the operand is activations or the
            // quantizing_op is non-supported/weight-only.
            inputs.push_back(operand);
          }
        } else {
          if (auto dq_op =
                  dyn_cast_or_null<DequantizeOpT>(operand.getDefiningOp())) {
            inputs.push_back(dq_op.getOperand());
          } else if (!ele_type.isF32()) {
            // If the operand is an integer tensor, then it doesn't require the
            // DequantizeOp in the pattern.
            inputs.push_back(operand);
          } else {
            return failure();
          }
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantizing_op->getNumResults());
      for (const auto& enumerated_result :
           llvm::enumerate(quantizing_op->getResults())) {
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
        // If the user is the QuantizeOp, it must be the only user.
        if (result.hasOneUse() &&
            llvm::isa<QuantizeOpT>(*result.user_begin())) {
          auto user = llvm::cast<QuantizeOpT>(*result.user_begin());
          outputs_replaced.insert(
              {user.getResult(), enumerated_result.index()});
          output_types.push_back(user.getType());
        } else if (!result_ele_type.isF32()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else if (static_cast<const ConcreteT*>(this)
                       ->AllowDynamicRangeQuantizedResult(quantizing_op,
                                                          custom_map)) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          return failure();
        }
      }

      rewriter.setInsertionPointAfter(quantizing_op);
      OperationState new_state(quantizing_op->getLoc(),
                               quantizing_op->getName().getStringRef(), inputs,
                               output_types, quantizing_op->getAttrs());
      for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* quantized_op = rewriter.create(new_state);
      if (quantizing_op->getNumRegions() != 0) {
        for (const auto& indexed_regions :
             llvm::enumerate(quantizing_op->getRegions())) {
          Region& target_region =
              quantized_op->getRegion(indexed_regions.index());
          BlockAndValueMapping mapping;
          indexed_regions.value().cloneInto(&target_region, mapping);
        }
      }
      for (auto output : outputs_replaced) {
        output.getFirst().replaceAllUsesWith(
            quantized_op->getResult(output.getSecond()));
      }

      // To verify the numericals, the original floating-point ops are
      // preserved in the graph. The result of these floating-point ops are sent
      // to a numeric verifier op as the reference.
      if (enable_verify && !std::is_same<VerifierT, void>()) {
        // For constant operands, the floating-point constant is duplicated in
        // case it is quantized.
        for (int i = 0, e = quantized_op->getNumOperands(); i < e; ++i) {
          auto def = quantized_op->getOperand(i).getDefiningOp();
          if (auto q = llvm::dyn_cast_or_null<QuantizeOpT>(def)) {
            DenseFPElementsAttr attr;
            if (!matchPattern(q.getOperand(), m_Constant(&attr))) {
              continue;
            }
            auto cst = rewriter.create<arith::ConstantOp>(
                quantized_op->getLoc(), attr);
            quantizing_op->setOperand(i, cst.getResult());
          }
        }

        for (int i = 0, e = quantized_op->getNumResults(); i < e; ++i) {
          if (!quantizing_op->getResult(i)
                   .getType()
                   .cast<ShapedType>()
                   .getElementType()
                   .isa<FloatType>()) {
            continue;
          }
          CreateVerifier<VerifierT>(quantizing_op, quantized_op, rewriter, i,
                                    quant_params_);

          if (enable_whole_model_verify) {
            RewireFloatModelBackbone(quantized_op, quantizing_op);
          }
        }
      }
    }
    return success();
  }

 private:
  // Reconnects float ops in the whole-model verify mode. Works for both
  // Quantizable ops and Unquantizable ops
  void RewireFloatModelBackbone(Operation* quantized_op,
                                Operation* float_op) const {
    for (int i = 0, e = quantized_op->getNumResults(); i < e; ++i) {
      if (!float_op->getResult(i)
               .getType()
               .cast<ShapedType>()
               .getElementType()
               .isF32()) {
        continue;
      }
      // Find the Quantize/Dequantize users of the new op results, and replace
      // the usage. Then all the floating-point ops are connected, forming a
      // separate float "backbone" model that the quantized model can be
      // compared against in parallel.
      // N.B. the return op will use this floating-point result.
      Value result;
      if (IsOpNotQuantizable(float_op)) {
        // For not quantizable ops, search for dequantize attached to the
        // quantized op of the output.
        if (Operation* quantize_op = dyn_cast_or_null<QuantizeOpT>(
                *quantized_op->getResult(i).getUsers().begin())) {
          result = quantize_op->getResult(0);
        } else {
          quantize_op->emitError()
              << "Output[" << i
              << "] is expected to have only one user [QUANTIZE]";
          return;
        }
      } else {
        result = quantized_op->getResult(i);
      }
      for (auto user : result.getUsers()) {
        // Skip the Requantize op and set the user to the following dequantize
        // op. This happens when the quantizer tries to match the scale conflict
        // with QuantizeOp - QuantizeOp(requant) - DequantizeOp triples. The
        // correct float op should be the user of the last DequantizeOp.
        if (llvm::isa<QuantizeOpT>(user)) {
          user = *user->getResult(0).getUsers().begin();
        }
        if (auto dequantize = llvm::dyn_cast<DequantizeOpT>(user)) {
          // Replace all uses, except not quantizable ops that are being used in
          // the float backbone.
          dequantize.getResult().replaceUsesWithIf(
              float_op->getResult(i), [&](OpOperand& use) {
                return !use.getOwner()->hasAttr(kDebugModeOpQuantAttrName);
              });
        }
      }
    }
  }

  QuantPassSpec quant_params_;
};

// A pattern that removes debug attributes that are annotated to ops during
// the debug model creation.
class RemoveDebugAttrPattern : public RewritePattern {
 public:
  explicit RemoveDebugAttrPattern(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

// Converts quantized tensor type with signed integer type to quantized tensor
// type with unsigned integer type.
Type ConvertSignedQuantizedToUnsigned(Type signed_tensor_type, Location loc);

// Converts quantize ops with unsigned quantized types to these with signed
// quantized types and preserves the scales.
template <typename QuantizeOpT>
struct ConvertUnsignedToSigned : public OpRewritePattern<QuantizeOpT> {
  using BaseType = ConvertUnsignedToSigned<QuantizeOpT>;
  using QType = quant::QuantizedType;

  explicit ConvertUnsignedToSigned(MLIRContext* context)
      : OpRewritePattern<QuantizeOpT>(context, 1) {}

  LogicalResult matchAndRewrite(QuantizeOpT op,
                                PatternRewriter& rewriter) const override {
    Type output_type = op.getResult().getType();
    auto qtype = QType::getQuantizedElementType(output_type);
    if (!qtype || qtype.isSigned()) return failure();

    int num_bits = qtype.getStorageTypeIntegralWidth();
    if (num_bits == 8) {
      // If storage is 8-bit, trained num bits may be less than 8 so check here.
      num_bits =
          static_cast<int>(std::ceil(std::log2(qtype.getStorageTypeMax())));
    }
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
      for (int i = 0, e = new_zero_points.size(); i < e; ++i) {
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
    rewriter.replaceOpWithNewOp<QuantizeOpT>(op, new_output_type, op.getArg());
    return success();
  }
};

// Fold Extra Requantize ops if the preceding ops has free scale requirement.
template <typename RequantizeOpT>
struct FoldTrivalRequantizeOp : public OpRewritePattern<RequantizeOpT> {
  explicit FoldTrivalRequantizeOp(MLIRContext* context)
      : OpRewritePattern<RequantizeOpT>(context, 1) {}

  LogicalResult matchAndRewrite(RequantizeOpT op,
                                PatternRewriter& rewriter) const override {
    Value pre_quantized = op->getOperand(0);
    auto pre_quantized_type =
        quant::QuantizedType::getQuantizedElementType(pre_quantized.getType());
    if (!pre_quantized_type) return failure();

    Operation* def = pre_quantized.getDefiningOp();
    if (!def) return failure();
    if (llvm::isa<FixedOutputRangeInterface, SameScalesOpInterface>(def) ||
        !def->hasTrait<OpTrait::quant::QuantizableResult>()) {
      return failure();
    }

    // This op should not clobber def, if more than one requant of this value.
    if (!pre_quantized.hasOneUse()) {
      return failure();
    }

    op.emitWarning("Remove trivial `rescale` op. Please fix the source graph.");

    llvm::SmallVector<Type, 4> new_output_types;
    for (auto result : def->getResults()) {
      if (result.hasOneUse() && *result.getUsers().begin() == op) {
        new_output_types.push_back(op.getResult().getType());
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
    Operation* new_op = rewriter.create(new_state);

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
                              bool is_signed, bool legacy_float_scale = false,
                              bool use_fake_quant_num_bits = false);

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
                                      bool legacy_float_scale = false,
                                      bool use_fake_quant_num_bits = false);

// Returns the per channel quantized type for an element attribute.
// `quant_dim` defines the quantization axis. The channel min/max are adjusted
// to be symmetric if `symmetric` flag is set to True. And `symmetric` can only
// be set to true when it is signed and narrow_range.
Type GetUniformQuantizedPerAxisTypeForWeight(
    ElementsAttr attr, int quant_dim, bool symmetric, unsigned num_bits,
    bool is_signed, bool narrow_range, bool legacy_float_scale = false,
    bool use_fake_quant_num_bits = false);

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
void ApplyQuantizationParamsPropagation(mlir::func::FuncOp func, bool is_signed,
                                        bool disable_per_channel,
                                        OpQuantSpecGetter op_quant_spec_getter,
                                        bool infer_tensor_ranges,
                                        bool legacy_float_scale = false);

void ApplyQuantizationParamsPropagation(
    mlir::func::FuncOp func, bool is_signed, bool disable_per_channel,
    OpQuantSpecGetter op_quant_spec_getter,
    OpQuantScaleSpecGetter op_quant_scale_spec_getter, bool infer_tensor_ranges,
    bool legacy_float_scale = false);

// Gets quantization scale specs (e.g. fixed output range, same result and
// operand scales) from the default quantization interfaces. The op should
// outlive returned spec for its interface methods to be properly referenced.
std::unique_ptr<OpQuantScaleSpec> GetDefaultQuantScaleSpec(Operation* op);

// The function might contain more stats ops than required, and it will
// introduce requantize if the calibration stats have conflicts. This method
// tries to remove all the redundant stats ops.
bool RemoveRedundantStatsOps(mlir::func::FuncOp func,
                             OpQuantSpecGetter op_quant_spec_getter,
                             OpQuantScaleSpecGetter op_quant_scale_spec_getter =
                                 GetDefaultQuantScaleSpec);

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
                      bool legacy_float_scale = false,
                      bool use_fake_quant_num_bits = false);
}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_UTILS_H_
