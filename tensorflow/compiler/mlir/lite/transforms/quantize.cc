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

// This transformation pass applies quantization on TFLite dialect.

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//===----------------------------------------------------------------------===//
namespace {
#define GEN_PASS_DEF_QUANTIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

enum class OpQuantizationType { kSRQ, kDRQ, kWeightOnly, kUnsupported };

static LogicalResult IsDrqTensor(Value value, Value& fq_input) {
  if (auto composite_op = llvm::dyn_cast_or_null<stablehlo::CompositeOp>(
          value.getDefiningOp())) {
    if (IsDrqFakeQuant(composite_op)) {
      fq_input = composite_op.getOperand(0);
      return success();
    }
  }
  return failure();
}
static LogicalResult HasDQParent(Value value, Value& dq_input) {
  if (auto dq_op =
          llvm::dyn_cast_or_null<DequantizeOp>(value.getDefiningOp())) {
    dq_input = dq_op.getOperand();
    return success();
  }
  return failure();
}

static OpQuantizationType GetOpQuantizationType(Operation* op) {
  // The assumption here is that the op has at least one DQ operand since the
  // pattern's root is that.

  // Indicates if an input which is not an FQ is seen.
  bool non_fq_float_input_seen = false;
  Value fq_input, dq_input;
  for (auto operand : op->getOperands()) {
    if (IsDrqTensor(operand, fq_input).succeeded()) {
      // As soon as a DRQ tensor is encountered, the op is DRQ.
      return OpQuantizationType::kDRQ;
    }

    if (HasDQParent(operand, dq_input).succeeded()) {
      // Operands with QDQ can not specify the quantization type.
      continue;
    }

    auto element_type = getElementTypeOrSelf(operand.getType());

    // Ignore non-f32 tensors when determining the quantization type.
    // Examples:
    //  - i32 operands are generally index tensors (e.g. in transpose
    // permutation)
    //  - bool operands can be condition on a select_v2
    if (element_type.isF32()) {
      non_fq_float_input_seen = true;
    }
  }
  if (non_fq_float_input_seen) {
    return OpQuantizationType::kWeightOnly;
  }

  for (auto result : op->getResults()) {
    for (auto user : result.getUsers()) {
      auto q_op = llvm::dyn_cast_or_null<QuantizeOp>(user);
      if (!q_op) {
        return OpQuantizationType::kUnsupported;
      }
    }
  }

  return OpQuantizationType::kSRQ;
}

class RemoveUnusedFQ : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (IsDrqFakeQuant(op) && op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "is not a drq fake quant op with no uses.");
  }
};

class StrictQuantizationPattern : public RewritePattern {
 public:
  using BaseType = StrictQuantizationPattern;

  explicit StrictQuantizationPattern(MLIRContext* context,
                                     const QuantPassSpec& quant_params)
      // Set the score to a large number so it is always preferred.
      : RewritePattern(DequantizeOp::getOperationName(), 300, context),
        quant_params_(quant_params) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*, 4> quantizing_ops;
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto users = op->getResult(0).getUsers();
    quantizing_ops.append(users.begin(), users.end());

    tensorflow::DataType inference_type =
        quant_params_.quant_spec.inference_type;
    bool enable_verify = quant_params_.numeric_verify_spec.verify_numeric;
    bool enable_whole_model_verify =
        quant_params_.numeric_verify_spec.whole_model_verify;
    CustomOpMap custom_map = quant_params_.quant_spec.custom_map;

    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* quantizing_op : quantizing_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<QuantizeOp, DequantizeOp>(quantizing_op)) {
        return failure();
      }

      // If the op is terminator, not quantizable or any ops from the mlir quant
      // ops dialect, we shouldn't rewrite. In case of whole-model verify debug
      // mode, not-quantizable ops should be duplicated to keep parallel
      // float/quant model execution.
      if (quantizing_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }

      if (!IsOpQuantizable(quantizing_op) &&
          !IsQuantizableCustomOp(quantizing_op, custom_map)) {
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

      // An op with float inputs and outputs are expected when it's used by a
      // NumericVerify op. Skip this op.
      if (enable_verify && UsedBy<NumericVerifyOp>(quantizing_op)) {
        continue;
      }

      auto op_quant_type = GetOpQuantizationType(quantizing_op);

      if (op_quant_type == OpQuantizationType::kUnsupported) {
        return rewriter.notifyMatchFailure(
            quantizing_op, "Unsupported quantization type for op: " +
                               quantizing_op->getName().getStringRef().str());
      }

      bool is_operand_or_result_modified = false;
      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantizing_op->getNumOperands());
      for (auto operand : quantizing_op->getOperands()) {
        Type operand_type = operand.getType();
        if (mlir::isa<NoneType>(operand_type)) {
          inputs.push_back(operand);
          continue;
        }

        if (Value dq_input; HasDQParent(operand, dq_input).succeeded()) {
          if (op_quant_type == OpQuantizationType::kWeightOnly) {
            inputs.push_back(operand);
          } else {
            // In both SRQ and DRQ cases, the DQ is fused in.
            is_operand_or_result_modified = true;
            inputs.push_back(dq_input);
          }
        } else if (Value fq_input; IsDrqTensor(operand, fq_input).succeeded()) {
          is_operand_or_result_modified = true;
          inputs.push_back(fq_input);
        } else if (auto ele_type = getElementTypeOrSelf(operand_type);
                   ele_type.isF32() || ele_type.isInteger(32) ||
                   ele_type.isInteger(1)) {
          // If it's F32 (non-weight-only and non-drq) or I32 or bool, just
          // directly add the input.
          inputs.push_back(operand);
        } else {
          return rewriter.notifyMatchFailure(
              quantizing_op,
              "unsupported operand received during quantization of : " +
                  quantizing_op->getName().getStringRef().str());
        }
      }

      Operation* quantized_op;
      if (QuantizableOpSupportsFloatOutputType(quantizing_op)) {
        rewriter.setInsertionPointAfter(quantizing_op);
        OperationState new_state(
            quantizing_op->getLoc(), quantizing_op->getName().getStringRef(),
            inputs, quantizing_op->getResultTypes(), quantizing_op->getAttrs());
        for (const auto& indexed_regions :
             llvm::enumerate(quantizing_op->getRegions())) {
          Region* target_region = new_state.addRegion();
          IRMapping mapping;
          indexed_regions.value().cloneInto(target_region, mapping);
        }
        quantized_op = rewriter.create(new_state);
        rewriter.replaceOp(quantizing_op, quantized_op);
      } else {
        // Collect all the quantized outputs and replace them by the results of
        // the new quantized op.
        llvm::SmallDenseMap<Value, int> outputs_replaced;
        SmallVector<Type, 4> output_types;
        output_types.reserve(quantizing_op->getNumResults());
        for (const auto& enumerated_result :
             llvm::enumerate(quantizing_op->getResults())) {
          Value result = enumerated_result.value();
          Type result_type = result.getType();
          // Add this to the test coverage once we create test ops with none
          // type results.
          if (mlir::isa<NoneType>(result_type)) {
            outputs_replaced.insert({result, enumerated_result.index()});
            output_types.push_back(result_type);
            continue;
          }
          Type result_ele_type = getElementTypeOrSelf(result_type);
          // If the user is the QuantizeOp, it must be the only user.
          if (result.hasOneUse() &&
              llvm::isa<QuantizeOp>(*result.user_begin()) &&
              op_quant_type == OpQuantizationType::kSRQ) {
            auto user = llvm::cast<QuantizeOp>(*result.user_begin());
            outputs_replaced.insert(
                {user.getResult(), enumerated_result.index()});
            output_types.push_back(user.getType());
            is_operand_or_result_modified = true;
          } else if (result_ele_type.isF32()) {
            outputs_replaced.insert({result, enumerated_result.index()});
            output_types.push_back(result.getType());
          } else {
            return rewriter.notifyMatchFailure(
                quantizing_op, "output of fake quantized op is not float32.");
          }
        }

        // For float16 quantization if none of the operand or result is
        // modified, replacing the op. See b/335025403.
        if (inference_type == tensorflow::DT_HALF &&
            !is_operand_or_result_modified) {
          return failure();
        }

        rewriter.setInsertionPointAfter(quantizing_op);
        OperationState new_state(
            quantizing_op->getLoc(), quantizing_op->getName().getStringRef(),
            inputs, output_types, quantizing_op->getAttrs());
        for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
          new_state.addRegion();
        }
        quantized_op = rewriter.create(new_state);
        if (quantizing_op->getNumRegions() != 0) {
          for (const auto& indexed_regions :
               llvm::enumerate(quantizing_op->getRegions())) {
            Region& target_region =
                quantized_op->getRegion(indexed_regions.index());
            IRMapping mapping;
            indexed_regions.value().cloneInto(&target_region, mapping);
          }
        }
        for (auto output : outputs_replaced) {
          output.getFirst().replaceAllUsesWith(
              quantized_op->getResult(output.getSecond()));
        }
      }

      // To verify the numericals, the original floating-point ops are
      // preserved in the graph. The result of these floating-point ops are sent
      // to a numeric verifier op as the reference.
      if (enable_verify && !std::is_same_v<NumericVerifyOp, void>) {
        // For constant operands, the floating-point constant is duplicated in
        // case it is quantized.
        for (int i = 0, e = quantized_op->getNumOperands(); i < e; ++i) {
          auto def = quantized_op->getOperand(i).getDefiningOp();
          if (auto q = llvm::dyn_cast_or_null<QuantizeOp>(def)) {
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
          if (!mlir::isa<FloatType>(mlir::getElementTypeOrSelf(
                  quantizing_op->getResult(i).getType()))) {
            continue;
          }
          CreateVerifier<NumericVerifyOp>(quantizing_op, quantized_op, rewriter,
                                          i, quant_params_);

          if (enable_whole_model_verify) {
            RewireFloatModelBackbone(quantized_op, quantizing_op);
          }
        }
      }
    }
    return success();
  }

 private:
  bool IsQuantizableCustomOp(Operation* op,
                             const CustomOpMap& custom_op_map) const {
    // In some cases, ops may need to be quantized even though their op trait is
    // not quantizable. For example, for the case of custom op various ops can
    // be categorized as cusom ops despite each of them may require different
    // behaviors. In that case, these ops can be marked in the custom map and
    // treated separately in this pass.

    auto custom_op = llvm::dyn_cast_or_null<CustomOp>(op);
    if (!custom_op) return false;

    // Custom op which is marked in the custom op map is quantizable.
    std::string op_name = custom_op.getCustomCode().str();
    return (custom_op_map.find(op_name) == custom_op_map.end()) ? false : true;
  }

  // Reconnects float ops in the whole-model verify mode. Works for both
  // Quantizable ops and Unquantizable ops
  void RewireFloatModelBackbone(Operation* quantized_op,
                                Operation* float_op) const {
    for (int i = 0, e = quantized_op->getNumResults(); i < e; ++i) {
      if (!getElementTypeOrSelf(float_op->getResult(i).getType()).isF32()) {
        continue;
      }
      // Find the Quantize/Dequantize users of the new op results, and replace
      // the usage. Then all the floating-point ops are connected, forming a
      // separate float "backbone" model that the quantized model can be
      // compared against in parallel.
      // N.B. the return op will use this floating-point result.
      Value result;
      if (!IsOpQuantizable(float_op)) {
        // For not quantizable ops, search for dequantize attached to the
        // quantized op of the output.
        if (Operation* quantize_op = dyn_cast_or_null<QuantizeOp>(
                *quantized_op->getResult(i).getUsers().begin())) {
          result = quantize_op->getResult(0);
        } else {
          quantized_op->emitError()
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
        if (llvm::isa<QuantizeOp>(user)) {
          user = *user->getResult(0).getUsers().begin();
        }
        if (auto dequantize = llvm::dyn_cast<DequantizeOp>(user)) {
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

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcreteT,
          typename RootOpT = DequantizeOp>
struct TFLQuantizationBase
    : public QuantizationPattern<ConcreteT, QuantizeOp, DequantizeOp,
                                 NumericVerifyOp, RootOpT> {
  explicit TFLQuantizationBase(MLIRContext* ctx,
                               const QuantPassSpec& quant_params)
      : QuantizationPattern<ConcreteT, QuantizeOp, DequantizeOp,
                            NumericVerifyOp, RootOpT>(ctx, quant_params) {}

  static bool IsQuantizableCustomOp(Operation* op,
                                    const CustomOpMap& custom_op_map) {
    // In some cases, ops may need to be quantized even though their op trait is
    // not quantizable. For example, for the case of custom op various ops can
    // be categorized as cusom ops despite each of them may require different
    // behaviors. In that case, these ops can be marked in the custom map and
    // treated separately in this pass.

    auto custom_op = llvm::dyn_cast_or_null<CustomOp>(op);
    if (!custom_op) return false;

    // Custom op which is marked in the custom op map is quantizable.
    std::string op_name = custom_op.getCustomCode().str();
    return (custom_op_map.find(op_name) == custom_op_map.end()) ? false : true;
  }

  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const CustomOpMap& custom_op_map) {
    // Collect the input if dynamic range quantization is on and the op supports
    // it.
    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool AllowDynamicRangeQuantizedResult(
      Operation* quantized_op, const CustomOpMap& custom_op_map) {
    // Collect the output if dynamic range quantization is on and the op
    // supports it.
    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool IsWeightOnlyOp(
      Operation* quantized_op,
      const absl::flat_hash_set<std::string>& ops_blocklist,
      const bool weight_only_quantization, const CustomOpMap& custom_op_map) {
    // Check whether the quantized_op needs to be quantized in weight-only
    // manner.
    bool is_blocklisted = false;

    if (auto custom_op = dyn_cast_or_null<CustomOp>(quantized_op)) {
      std::string custom_op_name = custom_op.getCustomCode().str();
      auto custom_map_iter = custom_op_map.find(custom_op_name);

      is_blocklisted =
          ops_blocklist.find(custom_op_name) != ops_blocklist.end();

      bool weight_only_custom_op = custom_map_iter != custom_op_map.end()
                                       ? custom_map_iter->second.is_weight_only
                                       : false;

      return is_blocklisted || weight_only_custom_op ||
             weight_only_quantization;
    } else {
      auto dynamic_range_op =
          dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op);

      const auto op_name = quantized_op->getName().getStringRef().str();
      is_blocklisted = ops_blocklist.find(op_name) != ops_blocklist.end();

      bool kernel_support =
          dynamic_range_op.GetDynamicRangeQuantKernelSupport();

      return is_blocklisted || !kernel_support || weight_only_quantization;
    }
  }
};

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFLFullQuantization
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantization> {
  explicit TFLFullQuantization(MLIRContext* ctx,
                               const QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantization>(
            ctx, quant_params) {}
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFLFullQuantizationReverse
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                                 QuantizeOp> {
  explicit TFLFullQuantizationReverse(MLIRContext* ctx,
                                      const QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                            QuantizeOp>(ctx, quant_params) {}
};

// Dynamic range quantization rewrite pattern using DQ as the root op.
struct TFLDynamicRangeQuantization
    : public TFLQuantizationBase<kDynamicRangeQuantization,
                                 TFLDynamicRangeQuantization> {
  explicit TFLDynamicRangeQuantization(MLIRContext* ctx,
                                       const QuantPassSpec& quant_params)
      : TFLQuantizationBase<kDynamicRangeQuantization,
                            TFLDynamicRangeQuantization>(ctx, quant_params) {}
};

class QuantizeConstPattern : public OpRewritePattern<QuantizeOp> {
 public:
  explicit QuantizeConstPattern(MLIRContext* context, bool legacy_float_scale)
      : OpRewritePattern<QuantizeOp>(context),
        legacy_float_scale_(legacy_float_scale) {}
  LogicalResult matchAndRewrite(QuantizeOp op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (matchPattern(op.getInput(), m_Constant(&attr))) {
      auto qtype = op.getQtypeAttr();
      Attribute quantized_attr;
      if (legacy_float_scale_) {
        quantized_attr = QuantizeLegacy(attr, qtype.getValue());
      } else {
        quantized_attr = Quantize(attr, qtype.getValue());
      }
      if (quantized_attr) {
        auto qconst_op =
            rewriter.create<QConstOp>(op.getLoc(), qtype, quantized_attr);
        if (auto volatile_attr = op->getAttr(kVolatileOpAttrName)) {
          qconst_op->setAttr(kVolatileOpAttrName, volatile_attr);
        }
        op.replaceAllUsesWith(qconst_op.getOutput());
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }

 private:
  bool legacy_float_scale_;
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public impl::QuantizePassBase<QuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)

  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() { quant_specs.inference_type = tensorflow::DT_QINT8; }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs(quant_specs) {
    enable_numeric_verify_ = quant_specs.verify_numeric;
    enable_whole_model_verify_ = quant_specs.whole_model_verify;
    enable_legacy_quantize_ = quant_specs.legacy_float_scale;
    enable_dynamic_range_quantization_ = quant_specs.weight_quantization;
    enable_weight_only_quantization_ = quant_specs.weight_only_quantization;
    qdq_conversion_mode_ =
        GetQDQQuantModeString(quant_specs.qdq_conversion_mode);
  }

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

namespace quantize_by_converter_patterns {
#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize_by_converter.inc"
}

void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  // Following updates the quant spec from the pass options since the tests
  // might have updated them.
  quant_specs.verify_numeric = enable_numeric_verify_;
  quant_specs.whole_model_verify = enable_whole_model_verify_;
  quant_specs.legacy_float_scale = enable_legacy_quantize_;
  quant_specs.weight_quantization = enable_dynamic_range_quantization_;
  quant_specs.weight_only_quantization = enable_weight_only_quantization_;
  quant_specs.qdq_conversion_mode =
      GetQDQQuantModeFromString(qdq_conversion_mode_);

  if (!ops_blocklist_flag_.empty()) {
    quant_specs.ops_blocklist = absl::flat_hash_set<std::string>(
        ops_blocklist_flag_.begin(), ops_blocklist_flag_.end());
  }

  if (!nodes_blocklist_flag_.empty()) {
    quant_specs.nodes_blocklist = absl::flat_hash_set<std::string>(
        nodes_blocklist_flag_.begin(), nodes_blocklist_flag_.end());
  }

  if (!enable_custom_op_weight_only_.empty()) {
    ParseCustomOpSpecs(enable_custom_op_weight_only_,
                       CustomOpUpdateOptions::kWeightOnly,
                       quant_specs.custom_map);
  }
  if (enable_float16_quantization_) {
    quant_specs.inference_type = tensorflow::DT_HALF;
  }

  const QuantPassSpec quant_params = {
      {quant_specs.verify_numeric, error_tolerance_,
       quant_specs.whole_model_verify, enable_log_if_failed_},
      quant_specs};

  if (quant_specs.qdq_conversion_mode == QDQConversionMode::kQDQStrict) {
    patterns.add<StrictQuantizationPattern>(ctx, quant_params);
    patterns.add<RemoveUnusedFQ, SquashDqQ, FuseDqQToRequant>(ctx);
  } else if (quant_specs.weight_quantization ||
             quant_specs.use_fake_quant_num_bits ||
             quant_specs.qdq_conversion_mode ==
                 QDQConversionMode::kQDQDynamic) {
    patterns.add<SquashDqQ, EliminateRemnantConstQDQ>(ctx);
    quantize_by_converter_patterns::populateWithGenerated(patterns);
    patterns.add<TFLDynamicRangeQuantization>(ctx, quant_params);
  } else if (quant_specs.qdq_conversion_mode == QDQConversionMode::kQDQNone) {
    patterns.add<SquashDqQ, EliminateRemnantConstQDQ>(ctx);
    quantize_by_converter_patterns::populateWithGenerated(patterns);
    patterns.add<TFLFullQuantization, TFLFullQuantizationReverse>(ctx,
                                                                  quant_params);
  } else {
    patterns.add<SquashDqQ, EliminateRemnantConstQDQ>(ctx);
    patterns.add<TFLFullQuantization, TFLFullQuantizationReverse>(ctx,
                                                                  quant_params);
  }

  (void)applyPatternsGreedily(func, std::move(patterns));

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<QuantizeConstPattern>(ctx, quant_specs.legacy_float_scale);
  if (quant_params.numeric_verify_spec.whole_model_verify) {
    patterns_2.add<RemoveDebugAttrPattern>(ctx);
  }
  (void)applyPatternsGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    const QuantizationSpecs& quant_specs,
    const absl::flat_hash_set<std::string>& ops_blocklist,
    const absl::flat_hash_set<std::string>& nodes_blocklist) {
  QuantizationSpecs updated_quant_specs;
  updated_quant_specs = quant_specs;
  // If there's new blocklists given, update quant_specs to use the new one.
  if (!ops_blocklist.empty()) {
    updated_quant_specs.ops_blocklist = ops_blocklist;
  }
  if (!nodes_blocklist.empty()) {
    updated_quant_specs.nodes_blocklist = nodes_blocklist;
  }
  return std::make_unique<QuantizePass>(updated_quant_specs);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantizePass() {
  return std::make_unique<QuantizePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    const bool verify_numeric, const bool whole_model_verify,
    const bool legacy_float_scale,
    const absl::flat_hash_set<std::string>& ops_blocklist,
    const absl::flat_hash_set<std::string>& nodes_blocklist) {
  QuantizationSpecs quant_specs;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = ops_blocklist;
  quant_specs.nodes_blocklist = nodes_blocklist;
  return std::make_unique<QuantizePass>(quant_specs);
}

}  // namespace TFL
}  // namespace mlir
